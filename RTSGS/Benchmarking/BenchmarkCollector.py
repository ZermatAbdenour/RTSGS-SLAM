from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class StageStats:
    total_ms: float = 0.0
    count: int = 0
    last_ms: float = 0.0
    max_ms: float = 0.0

    def add(self, ms: float) -> None:
        m = float(ms)
        self.total_ms += m
        self.count += 1
        self.last_ms = m
        if m > self.max_ms:
            self.max_ms = m

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count > 0 else 0.0


def _bytes_to_gb(num_bytes: float) -> float:
    return float(num_bytes) / (1024.0**3)


def _tensor_nbytes(obj) -> int:
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return int(obj.numel()) * int(obj.element_size())
        return 0
    except Exception:
        return 0


def _module_nbytes(module) -> int:
    if module is None:
        return 0
    try:
        import torch

        total = 0
        for p in module.parameters(recurse=True):
            total += int(p.numel()) * int(p.element_size())
        for b in module.buffers(recurse=True):
            if isinstance(b, torch.Tensor):
                total += int(b.numel()) * int(b.element_size())
        return int(total)
    except Exception:
        return 0


class BenchmarkCollector:
    """Thread-safe benchmark accumulator used only when `benchmark=True`.

    Tracks:
    - wall-time per stage (avg over repeats)
    - GPU memory: steady/peak + mapping peak delta
    """

    STAGE_TRACKING = "Tracking"
    STAGE_MAPPING = "Mapping per keyframe"
    STAGE_MASK2FORMER_FWD = "Mask2Former forward"
    STAGE_INSTANCE_ASSOC = "Instance association + cache"
    STAGE_RELATION_FWD = "Relation head forward"
    STAGE_END_TO_END = "End-to-end pipeline"

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)
        self._lock = threading.Lock()
        self._stages: Dict[str, StageStats] = {}

        # Trajectory lifecycle
        self.started_t = time.time()
        self.trajectory_ended = False
        self.trajectory_end_t: float | None = None
        self.finalized = False
        self.finalize_status: str = "Not finalized"

        # GPU memory summary (bytes)
        self.mask2former_weights_bytes: int = 0
        self.relation_head_ema_bytes: int = 0
        self.gaussian_map_bytes: int = 0
        self.total_steady_bytes: int = 0
        self.total_peak_bytes: int = 0
        self.mapping_activation_peak_bytes: int = 0

        # We track a "total peak" ourselves so we can safely reset PyTorch's
        # peak stats for per-stage measurements (e.g., mapping activation peak).
        self._observed_total_peak_bytes: int = 0

    def stage(self, name: str) -> StageStats:
        st = self._stages.get(name)
        if st is None:
            st = StageStats()
            self._stages[name] = st
        return st

    def record_ms(self, stage_name: str, ms: float) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.stage(stage_name).add(ms)

    def mark_trajectory_ended(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if not self.trajectory_ended:
                self.trajectory_ended = True
                self.trajectory_end_t = time.time()

    def update_mapping_activation_peak(self, peak_delta_bytes: int) -> None:
        if not self.enabled:
            return
        b = int(max(0, peak_delta_bytes))
        with self._lock:
            if b > int(self.mapping_activation_peak_bytes):
                self.mapping_activation_peak_bytes = b

    def reset_total_peak(self) -> None:
        """Reset CUDA peak counters once at start of a benchmark run."""
        if not self.enabled:
            return
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            with self._lock:
                self._observed_total_peak_bytes = 0
        except Exception:
            return

    def observe_total_peak_value(self, peak_bytes: int) -> None:
        if not self.enabled:
            return
        b = int(max(0, peak_bytes))
        with self._lock:
            if b > int(self._observed_total_peak_bytes):
                self._observed_total_peak_bytes = b

    def observe_total_peak(self) -> None:
        if not self.enabled:
            return
        try:
            import torch

            if torch.cuda.is_available():
                self.observe_total_peak_value(int(torch.cuda.max_memory_allocated()))
        except Exception:
            return

    def _compute_gaussian_map_bytes(self, pcd) -> int:
        if pcd is None:
            return 0
        total = 0
        # Core map tensors
        for name in (
            "all_points",
            "all_sh",
            "all_scales",
            "all_quaternions",
            "all_alpha",
            # segmentation-aligned buffers
            "segmentation_labels",
            "segmentation_colors",
            "segmentation_color_logits",
            "semantic_confidence",
            "semantic_alt_class",
            "semantic_alt_confidence",
            "semantic_switch_support",
            "semantic_switch_cooldown",
            "gaussian_instance_ids",
            # voxel packing keys
            "seen_keys",
        ):
            total += _tensor_nbytes(getattr(pcd, name, None))

        # Camera intrinsics/extrinsics tensors are tiny but live on GPU.
        for name in (
            "fx",
            "fy",
            "cx",
            "cy",
            "rgb_fx",
            "rgb_fy",
            "rgb_cx",
            "rgb_cy",
            "R_depth_to_rgb",
            "t_depth_to_rgb",
            "R_fix",
        ):
            total += _tensor_nbytes(getattr(pcd, name, None))

        return int(total)

    def finalize(self, pcd=None, segmenter=None, scene_graph=None) -> None:
        """Compute memory components and snapshot steady/peak totals.

        Should be called after trajectory tracking ended and background workers are idle.
        """
        if not self.enabled:
            return

        with self._lock:
            if self.finalized:
                return

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                steady = int(torch.cuda.memory_allocated())
                # Use our observed peak if available; otherwise fall back.
                with self._lock:
                    peak = int(self._observed_total_peak_bytes)
                if peak <= 0:
                    peak = int(torch.cuda.max_memory_allocated())
            else:
                steady = 0
                peak = 0

            mask_model = getattr(segmenter, "_model", None) if segmenter is not None else None
            rel_model = getattr(scene_graph, "_relation_head", None) if scene_graph is not None else None

            mask_bytes = _module_nbytes(mask_model)
            rel_bytes = _module_nbytes(rel_model)

            # Paper-style accounting: keep a separate EMA copy for the relation head.
            rel_ema_bytes = 2 * rel_bytes if rel_bytes > 0 else 0

            gauss_bytes = self._compute_gaussian_map_bytes(pcd)

            with self._lock:
                self.mask2former_weights_bytes = int(mask_bytes)
                self.relation_head_ema_bytes = int(rel_ema_bytes)
                self.gaussian_map_bytes = int(gauss_bytes)
                self.total_steady_bytes = int(steady)
                self.total_peak_bytes = int(peak)
                self.finalized = True
                self.finalize_status = "OK"
        except Exception as e:
            with self._lock:
                self.finalized = True
                self.finalize_status = f"Finalize error: {e}"

    def get_stage_snapshot(self) -> Dict[str, StageStats]:
        with self._lock:
            return {k: StageStats(**vars(v)) for k, v in self._stages.items()}

    def get_table_ix(self) -> dict:
        with self._lock:
            stages = {k: StageStats(**vars(v)) for k, v in self._stages.items()}
            status = {
                "enabled": bool(self.enabled),
                "trajectory_ended": bool(self.trajectory_ended),
                "finalized": bool(self.finalized),
                "finalize_status": str(self.finalize_status),
            }

            if self.finalized:
                mem = {
                    "mask2former_weights_gb": _bytes_to_gb(self.mask2former_weights_bytes),
                    "relation_head_ema_gb": _bytes_to_gb(self.relation_head_ema_bytes),
                    "gaussian_map_steady_gb": _bytes_to_gb(self.gaussian_map_bytes),
                    "mapping_activation_peak_gb": _bytes_to_gb(self.mapping_activation_peak_bytes),
                    "total_steady_gb": _bytes_to_gb(self.total_steady_bytes),
                    "total_peak_gb": _bytes_to_gb(self.total_peak_bytes),
                }
            else:
                mem = {
                    "mask2former_weights_gb": None,
                    "relation_head_ema_gb": None,
                    "gaussian_map_steady_gb": None,
                    "mapping_activation_peak_gb": None,
                    "total_steady_gb": None,
                    "total_peak_gb": None,
                }

        def row(name: str) -> dict:
            st = stages.get(name)
            if st is None or st.count <= 0:
                return {"stage": name, "avg_ms": None, "hz": None, "count": 0}
            avg = float(st.avg_ms)
            hz = (1000.0 / avg) if avg > 1e-9 else None
            return {"stage": name, "avg_ms": avg, "hz": hz, "count": int(st.count)}

        rows = [
            row(self.STAGE_TRACKING),
            row(self.STAGE_MAPPING),
            row(self.STAGE_MASK2FORMER_FWD),
            row(self.STAGE_INSTANCE_ASSOC),
            row(self.STAGE_RELATION_FWD),
            row(self.STAGE_END_TO_END),
        ]
        return {"status": status, "rows": rows, "memory": mem}
