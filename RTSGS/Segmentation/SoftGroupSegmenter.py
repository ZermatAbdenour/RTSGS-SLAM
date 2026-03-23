import math
import os
import sys
import threading
import time

import numpy as np
import torch
import yaml
from munch import Munch


class SoftGroupPeriodicSegmenter:
    def __init__(self, pcd, config, project_root: str):
        self.pcd = pcd
        self.config = config
        self.project_root = project_root

        self.enabled = bool(config.get("softgroup_enabled", True))
        self.interval_sec = float(config.get("softgroup_interval_sec", 5.0))
        self.min_points = int(config.get("softgroup_min_points", 5000))
        self.max_points = int(config.get("softgroup_max_points", 180000))
        self.min_confidence = float(config.get("softgroup_min_confidence", 0.1))
        self.fast_mode = bool(config.get("softgroup_fast_mode", True))

        self.cfg_path = self._abs_path(
            str(config.get("softgroup_config", "ThirdParty/SoftGroup/configs/softgroup/softgroup_scannet.yaml"))
        )
        self.ckpt_path = self._abs_path(
            str(config.get("softgroup_checkpoint", "Prototype/softgroup_scannet_spconv2.pth"))
        )

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._running = False

        self._model = None
        self._cfg = None
        self._voxelization_idx = None
        self._rng = np.random.default_rng(123)

    def _abs_path(self, p: str) -> str:
        if os.path.isabs(p):
            return p
        return os.path.join(self.project_root, p)

    def start(self):
        if not self.enabled:
            print("[SoftGroup] Disabled by config (softgroup_enabled=False).")
            return
        if not torch.cuda.is_available():
            print("[SoftGroup] CUDA unavailable. Periodic segmentation is disabled.")
            return
        if self._running:
            return
        self._running = True
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._running and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._running = False

    def _loop(self):
        # Delay first run slightly to allow map bootstrapping.
        next_t = time.time() + min(2.0, self.interval_sec)
        while not self._stop.is_set():
            now = time.time()
            wait_s = next_t - now
            if wait_s > 0:
                self._stop.wait(wait_s)
                if self._stop.is_set():
                    break

            try:
                self._run_once()
            except Exception as exc:
                print(f"[SoftGroup] Segmentation pass failed: {exc}")

            next_t = time.time() + max(0.5, self.interval_sec)

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True

        if not os.path.exists(self.cfg_path):
            print(f"[SoftGroup] Config not found: {self.cfg_path}")
            return False
        if not os.path.exists(self.ckpt_path):
            print(f"[SoftGroup] Checkpoint not found: {self.ckpt_path}")
            return False

        softgroup_root = os.path.join(self.project_root, "ThirdParty", "SoftGroup")
        if softgroup_root not in sys.path:
            sys.path.insert(0, softgroup_root)

        from softgroup.model import SoftGroup
        from softgroup.ops import voxelization_idx
        from softgroup.util import get_root_logger, load_checkpoint

        with open(self.cfg_path, "r", encoding="utf-8") as f:
            cfg = Munch.fromDict(yaml.safe_load(f))

        if self.fast_mode:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
            cfg.model.test_cfg.lvl_fusion = True

        model = SoftGroup(**cfg.model).cuda()
        model.eval()
        logger = get_root_logger()
        load_checkpoint(self.ckpt_path, logger, model)

        self._cfg = cfg
        self._model = model
        self._voxelization_idx = voxelization_idx

        print("[SoftGroup] Model loaded for periodic segmentation.")
        return True

    def _snapshot_pointcloud(self):
        with self.pcd.lock:
            if self.pcd.all_points is None or self.pcd.all_sh is None:
                return None
            points = self.pcd.all_points.detach().cpu().numpy().astype(np.float32)
            sh0 = self.pcd.all_sh[:, 0, :].detach().cpu().numpy().astype(np.float32)

        # Inverse of SH0 encoding used by PointCloud.
        rgb = 1.0 / (1.0 + np.exp(-(sh0 * 0.28209479177387814)))
        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
        return points, rgb

    def _build_batch(self, xyz: np.ndarray, rgb_01: np.ndarray):
        n_total = xyz.shape[0]
        if n_total > self.max_points:
            sampled_idx = self._rng.choice(n_total, self.max_points, replace=False)
            sampled_idx = np.sort(sampled_idx)
            xyz = xyz[sampled_idx]
            rgb_01 = rgb_01[sampled_idx]
        else:
            sampled_idx = np.arange(n_total, dtype=np.int64)

        rgb = rgb_01 * 2.0 - 1.0

        voxel_scale = int(self._cfg.data.test.voxel_cfg.scale)
        min_spatial_shape = int(self._cfg.data.test.voxel_cfg.spatial_shape[0])

        xyz_t = torch.from_numpy(xyz).to(device=self.device, dtype=torch.float32)
        theta = 0.35 * math.pi
        rot = torch.tensor(
            [
                [math.cos(theta), math.sin(theta), 0.0],
                [-math.sin(theta), math.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            device=self.device,
            dtype=torch.float32,
        )

        coord_float = xyz_t @ rot
        xyz_scaled = coord_float * float(voxel_scale)
        xyz_scaled = xyz_scaled - xyz_scaled.min(dim=0, keepdim=True).values

        feat = torch.from_numpy(rgb).to(device=self.device, dtype=torch.float32)
        n = xyz_scaled.shape[0]
        coord_cpu = xyz_scaled.to("cpu").long()
        coords_cpu = torch.cat([torch.zeros((n, 1), dtype=torch.long), coord_cpu], dim=1)

        voxel_coords, v2p_map, p2v_map = self._voxelization_idx(coords_cpu, 1)

        semantic_labels = torch.zeros((n,), dtype=torch.long, device=self.device)
        instance_labels = torch.full((n,), -100, dtype=torch.long, device=self.device)
        pt_offset_labels = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
        batch_idxs = coords_cpu[:, 0].int().to(self.device)

        spatial_shape = np.clip(coords_cpu[:, 1:].max(0)[0].numpy() + 1, min_spatial_shape, None)

        batch = {
            "scan_ids": ["rtsgs_map"],
            "batch_idxs": batch_idxs,
            "voxel_coords": voxel_coords.to(self.device),
            "p2v_map": p2v_map.to(self.device),
            "v2p_map": v2p_map.to(self.device),
            "coords_float": coord_float,
            "feats": feat,
            "semantic_labels": semantic_labels,
            "instance_labels": instance_labels,
            "pt_offset_labels": pt_offset_labels,
            "spatial_shape": spatial_shape,
            "batch_size": 1,
        }
        return batch, sampled_idx, n_total

    def _semantic_palette(self, labels: np.ndarray, num_classes: int) -> np.ndarray:
        rng = np.random.default_rng(seed=123)
        palette = rng.uniform(0.1, 0.95, size=(max(num_classes, 1), 3)).astype(np.float32)
        safe_labels = np.clip(labels.astype(np.int64), 0, palette.shape[0] - 1)
        return palette[safe_labels]

    def _encode_colors_for_shader(self, rgb_01: np.ndarray) -> np.ndarray:
        # Renderer shader outputs vec3(sigmoid(z), sigmoid(y), sigmoid(x));
        # build inverse so resulting displayed color equals rgb_01.
        c = np.clip(rgb_01, 1e-3, 1.0 - 1e-3).astype(np.float32)
        logits = np.log(c / (1.0 - c)).astype(np.float32)
        encoded = np.empty_like(logits)
        encoded[:, 0] = logits[:, 2]
        encoded[:, 1] = logits[:, 1]
        encoded[:, 2] = logits[:, 0]
        return encoded

    def _run_once(self):
        if not self._ensure_model():
            return

        snap = self._snapshot_pointcloud()
        if snap is None:
            return

        xyz, rgb = snap
        n_total = xyz.shape[0]
        if n_total < self.min_points:
            return

        prep_t0 = time.perf_counter()
        batch, sampled_idx, n_total = self._build_batch(xyz, rgb)

        with torch.inference_mode():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = self._model(batch)
            torch.cuda.synchronize()
            infer_ms = (time.perf_counter() - t0) * 1000.0

        semantic_preds = np.asarray(result["semantic_preds"], dtype=np.int64)
        if semantic_preds.shape[0] != sampled_idx.shape[0]:
            print(
                f"[SoftGroup] Size mismatch: preds={semantic_preds.shape[0]}, sampled={sampled_idx.shape[0]}."
            )
            return

        labels_full = np.full((n_total,), -1, dtype=np.int64)
        labels_full[sampled_idx] = semantic_preds

        num_classes = int(getattr(self._cfg.model, "semantic_classes", 1))
        seg_rgb = np.zeros((n_total, 3), dtype=np.float32)
        valid = labels_full >= 0
        if np.any(valid):
            seg_rgb[valid] = self._semantic_palette(labels_full[valid], num_classes)

        seg_encoded = self._encode_colors_for_shader(seg_rgb)

        pred_instances = []
        for inst in result.get("pred_instances", []):
            if float(inst.get("conf", 0.0)) > self.min_confidence:
                pred_instances.append(inst)

        total_ms = (time.perf_counter() - prep_t0) * 1000.0
        self.pcd.set_segmentation_result(
            labels_full,
            seg_rgb,
            seg_encoded,
            pred_instances=pred_instances,
            metadata={
                "timestamp": time.time(),
                "num_points_total": int(n_total),
                "num_points_segmented": int(sampled_idx.shape[0]),
                "num_instances": int(len(pred_instances)),
                "inference_ms": float(infer_ms),
                "total_ms": float(total_ms),
            },
        )

        print(
            "[SoftGroup] Updated segmentation "
            f"({sampled_idx.shape[0]}/{n_total} pts) in {infer_ms:.1f} ms "
            f"(total {total_ms:.1f} ms)."
        )
