from __future__ import annotations

from pathlib import Path
import threading
import time
import numpy as np
import torch
from imgui_bundle import imgui

from RTSGS.GaussianSplatting.GaussianSplating import GaussianSplatting
from RTSGS.GUI.ImageWidget import ImageWidget


class BenchmarkWindow:
    def __init__(
        self,
        gs: GaussianSplatting,
        dataset,
        tracker=None,
        bench=None,
        title: str = "Benchmark",
        output_root: str | None = None,
    ):
        self.title = title
        self.is_open = True
        self._running = False
        self._progress = 0.0
        self._result_count = 0
        self._avg_psnr = 0.0
        self._avg_ssim = 0.0
        self._thread = None
        self.gs = gs
        self.dataset = dataset
        self.tracker = tracker
        self.bench = bench
        self.output_root = Path(output_root) if output_root is not None else Path.cwd() / "benchmark_results"
        self._run_dir: Path | None = None
        self._save_status: str = ""
        self._has_unsaved_results: bool = False
        self._bench_metrics_rows: list[tuple[int, float, float]] = []
        self._bench_pose_rows: list[tuple[int, np.ndarray]] = []
        self._bench_pred_poses: list | None = None
        self._bench_gt_poses: list | None = None
        self._preview_lock = threading.Lock()
        self._latest_gt_rgb: np.ndarray | None = None
        self._latest_rendered_rgb: np.ndarray | None = None
        self._latest_seg_rgb: np.ndarray | None = None
        self._latest_frame_index: int = -1
        self._gt_widget = ImageWidget()
        self._rendered_widget = ImageWidget()
        self._seg_widget = ImageWidget()
        self._traj_lock = threading.Lock()
        self._traj_status = "Not computed"
        self._traj_metrics = None
        self._paper_frame_indices: list[int] = [0, 0, 0, 0, 0]
        self._paper_fig_status: str = ""
        self._paper_fig_generating = False

    def _dataset_paths(self) -> tuple[str, str]:
        rgb_path = ""
        depth_path = ""
        try:
            rgb_path = str(getattr(self.dataset, "_rgb_path", "") or "")
        except Exception:
            rgb_path = ""
        try:
            depth_path = str(getattr(self.dataset, "_depth_path", "") or "")
        except Exception:
            depth_path = ""
        return rgb_path, depth_path

    @staticmethod
    def _sanitize_tag(s: str, max_len: int = 64) -> str:
        import re

        t = str(s).strip()
        if not t:
            return "dataset"
        t = t.replace(" ", "_")
        t = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", t)
        t = re.sub(r"_+", "_", t).strip("_-")
        if not t:
            t = "dataset"
        return t[:max_len]

    def _prepare_run_dir(self) -> Path:
        """Create an output folder only when user explicitly clicks Save."""
        from pathlib import Path as _Path

        rgb_path, _ = self._dataset_paths()
        dataset_tag = self._sanitize_tag(_Path(rgb_path).name if rgb_path else "dataset")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_root / f"{timestamp}_{dataset_tag}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _save_results_to_disk(self) -> None:
        if self._running:
            self._save_status = "Cannot save while benchmark is running."
            return
        if not self._has_unsaved_results:
            self._save_status = "Nothing new to save."
            return

        rgb_path, depth_path = self._dataset_paths()

        # Snapshot state for saving.
        with self._traj_lock:
            traj_metrics = None if self._traj_metrics is None else dict(self._traj_metrics)
            traj_status = str(self._traj_status)

        metrics_rows = list(self._bench_metrics_rows)
        pose_rows = [(int(i), np.asarray(p).astype(np.float32)) for i, p in list(self._bench_pose_rows)]

        # Table IX snapshot (if available).
        table_ix = None
        if self.bench is not None:
            try:
                table_ix = self.bench.get_table_ix()
            except Exception:
                table_ix = None

        run_dir = self._prepare_run_dir()
        self._run_dir = run_dir

        # 1) Per-frame image metrics (PSNR/SSIM)
        try:
            out = run_dir / "benchmark_image_metrics.csv"
            with out.open("w", encoding="utf-8") as f:
                f.write("dataset_rgb_path,dataset_depth_path,frame_index,psnr,ssim\n")
                for fi, psnr, ssim in metrics_rows:
                    f.write(f"{rgb_path},{depth_path},{int(fi)},{float(psnr)},{float(ssim)}\n")
        except Exception as e:
            self._save_status = f"Save error (image metrics): {e}"
            return

        # 2) Trajectory metrics (ATE/RPE/etc)
        try:
            out = run_dir / "trajectory_metrics.csv"
            with out.open("w", encoding="utf-8") as f:
                f.write("dataset_rgb_path,dataset_depth_path,status,metric,value\n")
                if traj_metrics is None:
                    f.write(f"{rgb_path},{depth_path},{traj_status},,\n")
                else:
                    for k, v in traj_metrics.items():
                        f.write(f"{rgb_path},{depth_path},{traj_status},{k},{v}\n")
        except Exception as e:
            self._save_status = f"Save error (trajectory metrics): {e}"
            return

        # 3) Trajectory poses (pred + gt if available)
        try:
            gt_poses = getattr(self.dataset, "gt_poses", None)
            gt_lookup = {}
            if isinstance(gt_poses, (list, tuple)):
                for idx, g in enumerate(gt_poses):
                    if g is None:
                        continue
                    g = np.asarray(g)
                    if g.shape == (4, 4) and np.isfinite(g).all():
                        gt_lookup[int(idx)] = g.astype(np.float32)

            out = run_dir / "trajectory_poses.csv"
            with out.open("w", encoding="utf-8") as f:
                pred_cols = ",".join([f"pred_{j:02d}" for j in range(16)])
                gt_cols = ",".join([f"gt_{j:02d}" for j in range(16)])
                f.write(f"dataset_rgb_path,dataset_depth_path,frame_index,{pred_cols},{gt_cols}\n")
                for fi, pose in pose_rows:
                    pred_flat = pose.reshape(-1)
                    pred_vals = ",".join(f"{float(x):.9f}" for x in pred_flat)
                    gt = gt_lookup.get(int(fi))
                    if gt is None:
                        gt_vals = ",".join(["" for _ in range(16)])
                    else:
                        gt_vals = ",".join(f"{float(x):.9f}" for x in gt.reshape(-1))
                    f.write(f"{rgb_path},{depth_path},{int(fi)},{pred_vals},{gt_vals}\n")
        except Exception as e:
            self._save_status = f"Save error (trajectory poses): {e}"
            return

        # 4) Table IX wall-time + memory
        try:
            if isinstance(table_ix, dict):
                rows = table_ix.get("rows", []) if isinstance(table_ix.get("rows", []), list) else []
                mem = table_ix.get("memory", {}) if isinstance(table_ix.get("memory", {}), dict) else {}

                out = run_dir / "table_ix_walltime.csv"
                with out.open("w", encoding="utf-8") as f:
                    f.write("dataset_rgb_path,dataset_depth_path,stage,avg_ms,hz,count\n")
                    for r in rows:
                        if not isinstance(r, dict):
                            continue
                        stage = str(r.get("stage", ""))
                        avg_ms = r.get("avg_ms", "")
                        hz = r.get("hz", "")
                        cnt = int(r.get("count", 0) or 0)
                        f.write(f"{rgb_path},{depth_path},{stage},{avg_ms},{hz},{cnt}\n")

                out = run_dir / "table_ix_memory.csv"
                with out.open("w", encoding="utf-8") as f:
                    f.write("dataset_rgb_path,dataset_depth_path,component,gb\n")
                    for label, key in (
                        ("Mask2Former weights", "mask2former_weights_gb"),
                        ("Relation head + EMA", "relation_head_ema_gb"),
                        ("Gaussian map steady state", "gaussian_map_steady_gb"),
                        ("Activation peak during mapping", "mapping_activation_peak_gb"),
                        ("Total steady state", "total_steady_gb"),
                        ("Total peak", "total_peak_gb"),
                    ):
                        val = mem.get(key, "")
                        f.write(f"{rgb_path},{depth_path},{label},{val}\n")
        except Exception as e:
            self._save_status = f"Save error (Table IX): {e}"
            return

        self._has_unsaved_results = False
        self._save_status = f"Saved to: {run_dir}"

    @staticmethod
    def _save_image(path: Path, image: np.ndarray) -> None:
        import cv2

        if image is None:
            return
        img = np.asarray(image)
        if img.ndim != 3 or img.shape[2] != 3:
            return
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        bgr = img[..., ::-1]
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), bgr)

    @staticmethod
    def _load_image_rgb(path: Path) -> np.ndarray | None:
        import cv2

        img_rgb = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_rgb is None:
            return None
        return img_rgb

    @staticmethod
    def _fit_preview(image: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
        import cv2

        if image is None:
            return image
        img = np.asarray(image)
        if img.ndim != 3 or img.shape[2] != 3:
            return img
        h, w = img.shape[:2]
        if w <= 0 or h <= 0:
            return img
        scale = min(max_w / float(w), max_h / float(h), 1.0)
        if scale >= 1.0:
            return img
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _umeyama_align(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
        src = np.asarray(src, dtype=np.float64)
        dst = np.asarray(dst, dtype=np.float64)
        n = src.shape[0]

        mu_src = src.mean(axis=0)
        mu_dst = dst.mean(axis=0)
        X = src - mu_src
        Y = dst - mu_dst

        cov = (Y.T @ X) / float(n)
        U, D, Vt = np.linalg.svd(cov)

        S = np.eye(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S[2, 2] = -1.0

        R = U @ S @ Vt

        if with_scale:
            var_src = (X ** 2).sum() / float(n)
            scale = float((D * np.diag(S)).sum() / var_src)
        else:
            scale = 1.0

        t = mu_dst - scale * (R @ mu_src)
        aligned = (scale * (R @ src.T)).T + t
        return aligned.astype(np.float32), (scale, R.astype(np.float32), t.astype(np.float32))

    @staticmethod
    def _rotation_error_deg(R: np.ndarray) -> float:
        trace = float(np.trace(R))
        cos_angle = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    @staticmethod
    def _path_length(points: np.ndarray) -> float:
        if points is None or points.shape[0] < 2:
            return 0.0
        diffs = points[1:] - points[:-1]
        return float(np.linalg.norm(diffs, axis=1).sum())

    def _compute_trajectory_metrics(self, pred_poses, gt_poses):
        if pred_poses is None or gt_poses is None:
            return None, "Ground truth poses not available."

        n = min(len(pred_poses), len(gt_poses))
        if n < 2:
            return None, "Not enough poses to compute trajectory metrics."

        pred_list = []
        gt_list = []
        for i in range(n):
            pred = pred_poses[i]
            gt = gt_poses[i]
            if pred is None or gt is None:
                continue
            pred = np.asarray(pred, dtype=np.float64)
            gt = np.asarray(gt, dtype=np.float64)
            if pred.shape != (4, 4) or gt.shape != (4, 4):
                continue
            if not np.isfinite(pred).all() or not np.isfinite(gt).all():
                continue
            pred_list.append(pred)
            gt_list.append(gt)

        if len(pred_list) < 2:
            return None, "Not enough valid pose pairs to compute metrics."

        pred_xyz = np.stack([p[:3, 3] for p in pred_list], axis=0)
        gt_xyz = np.stack([g[:3, 3] for g in gt_list], axis=0)

        raw_err = np.linalg.norm(pred_xyz - gt_xyz, axis=1)
        ate_raw_rmse = float(np.sqrt(np.mean(raw_err ** 2)))

        pred_aligned, (scale, _, _) = self._umeyama_align(pred_xyz, gt_xyz, with_scale=True)
        aligned_err = np.linalg.norm(pred_aligned - gt_xyz, axis=1)

        ate_aligned_rmse = float(np.sqrt(np.mean(aligned_err ** 2)))
        ate_aligned_mean = float(np.mean(aligned_err))
        ate_aligned_median = float(np.median(aligned_err))
        ate_aligned_std = float(np.std(aligned_err))
        ate_aligned_max = float(np.max(aligned_err))

        trans_errs = []
        rot_errs = []
        for i in range(len(pred_list) - 1):
            try:
                pred_rel = np.linalg.inv(pred_list[i]) @ pred_list[i + 1]
                gt_rel = np.linalg.inv(gt_list[i]) @ gt_list[i + 1]
                err_rel = np.linalg.inv(gt_rel) @ pred_rel
            except np.linalg.LinAlgError:
                continue

            trans_errs.append(float(np.linalg.norm(err_rel[:3, 3])))
            rot_errs.append(self._rotation_error_deg(err_rel[:3, :3]))

        if trans_errs:
            trans_errs = np.asarray(trans_errs, dtype=np.float64)
            rpe_trans_rmse = float(np.sqrt(np.mean(trans_errs ** 2)))
            rpe_trans_mean = float(np.mean(trans_errs))
        else:
            rpe_trans_rmse = 0.0
            rpe_trans_mean = 0.0

        if rot_errs:
            rot_errs = np.asarray(rot_errs, dtype=np.float64)
            rpe_rot_rmse = float(np.sqrt(np.mean(rot_errs ** 2)))
            rpe_rot_mean = float(np.mean(rot_errs))
        else:
            rpe_rot_rmse = 0.0
            rpe_rot_mean = 0.0

        pred_len = self._path_length(pred_xyz)
        gt_len = self._path_length(gt_xyz)
        path_ratio = float(pred_len / gt_len) if gt_len > 0.0 else 0.0

        metrics = {
            "alignment_scale": float(scale),
            "num_samples": int(pred_xyz.shape[0]),
            "ate_raw_rmse_m": ate_raw_rmse,
            "ate_aligned_rmse_m": ate_aligned_rmse,
            "ate_aligned_mean_m": ate_aligned_mean,
            "ate_aligned_median_m": ate_aligned_median,
            "ate_aligned_std_m": ate_aligned_std,
            "ate_aligned_max_m": ate_aligned_max,
            "rpe_trans_rmse_m": rpe_trans_rmse,
            "rpe_trans_mean_m": rpe_trans_mean,
            "rpe_rot_rmse_deg": rpe_rot_rmse,
            "rpe_rot_mean_deg": rpe_rot_mean,
            "pred_path_len_m": pred_len,
            "gt_path_len_m": gt_len,
            "path_len_ratio": path_ratio,
        }

        return metrics, ""

    @staticmethod
    def _depth_to_colormap(depth: np.ndarray, colormap: int | None = None) -> np.ndarray:
        import cv2

        d = np.asarray(depth, dtype=np.float32)
        valid = np.isfinite(d) & (d > 0.0)
        if not valid.any():
            return np.zeros((*d.shape[:2], 3), dtype=np.uint8)
        lo, hi = float(d[valid].min()), float(d[valid].max())
        if hi - lo < 1e-6:
            hi = lo + 1.0
        normed = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
        normed[~valid] = 0.0
        u8 = (normed * 255.0).astype(np.uint8)
        if colormap is None:
            colormap = cv2.COLORMAP_TURBO
        colored = cv2.applyColorMap(u8, colormap)
        colored[~valid] = 0
        return colored

    def _generate_paper_figure(self, indices: list[int]) -> None:
        import cv2

        self._paper_fig_generating = True
        self._paper_fig_status = "Generating..."

        pairs = getattr(self.dataset, "RGBD_pairs", None)
        pred_poses = None
        if self.tracker is not None:
            pred_poses = list(getattr(self.tracker, "poses", []))
        if pairs is None or pred_poses is None or len(pred_poses) == 0:
            self._paper_fig_status = "Dataset or poses not available."
            self._paper_fig_generating = False
            return

        gt_rgbs, gt_depths, rendered_rgbs, rendered_depths, seg_rgbs = [], [], [], [], []
        target_h, target_w = None, None

        pad_px = 14  # horizontal padding between sequential images (paper-figure)

        def hconcat_with_pad(images_bgr: list[np.ndarray]) -> np.ndarray:
            imgs = [np.asarray(im) for im in images_bgr if im is not None]
            if len(imgs) == 0:
                return np.zeros((1, 1, 3), dtype=np.uint8)
            if len(imgs) == 1:
                return imgs[0]

            h = int(imgs[0].shape[0])
            for im in imgs:
                if im.ndim != 3 or im.shape[2] != 3 or int(im.shape[0]) != h:
                    raise ValueError("All images must be HxWx3 with matching height")

            pad = np.full((h, int(pad_px), 3), 255, dtype=imgs[0].dtype)
            out_parts = []
            for j, im in enumerate(imgs):
                if j > 0:
                    out_parts.append(pad)
                out_parts.append(im)
            return np.concatenate(out_parts, axis=1)

        cfg = getattr(self.dataset, "config", None)
        depth_min_cfg = 0.0
        depth_max_cfg = 0.0
        try:
            if cfg is not None:
                depth_min_cfg = float(cfg.get("depth_min", 0.0))
                depth_max_cfg = float(cfg.get("depth_max", 0.0))
        except Exception:
            depth_min_cfg = 0.0
            depth_max_cfg = 0.0

        def colorize_depth_percentile(d: np.ndarray) -> np.ndarray:
            """Turbo colormap using percentile normalization (comparison-window style)."""
            dd = np.asarray(d, dtype=np.float32)
            valid = np.isfinite(dd) & (dd > 0.0)
            if depth_min_cfg > 0.0:
                valid &= dd > depth_min_cfg
            if depth_max_cfg > 0.0:
                valid &= dd < depth_max_cfg

            img = np.zeros(dd.shape[:2], dtype=np.uint8)
            if np.any(valid):
                dv = dd[valid]
                lo = float(np.percentile(dv, 5.0))
                hi = float(np.percentile(dv, 95.0))
                if hi <= lo:
                    hi = lo + 1e-6
                norm = np.clip((dd - lo) / (hi - lo), 0.0, 1.0)
                img = (norm * 255.0).astype(np.uint8)
                img[~valid] = 0
            colored = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)
            colored[~valid] = 255
            return colored

        for idx in indices:
            if idx < 0 or idx >= len(pairs) or idx >= len(pred_poses):
                self._paper_fig_status = f"Index {idx} out of range."
                self._paper_fig_generating = False
                return

            gt_rgb = cv2.imread(pairs[idx][0], cv2.IMREAD_COLOR)
            if gt_rgb is None:
                self._paper_fig_status = f"Failed to load GT RGB for frame {idx}."
                self._paper_fig_generating = False
                return

            gt_depth_raw = None
            if pairs[idx][1] is not None:
                gt_depth_raw = cv2.imread(pairs[idx][1], cv2.IMREAD_UNCHANGED)
            if gt_depth_raw is None:
                self._paper_fig_status = f"Failed to load GT depth for frame {idx}."
                self._paper_fig_generating = False
                return

            gt_depth_f = gt_depth_raw.astype(np.float32)
            depth_scale = float(cfg.get("depth_scale", 1000.0)) if cfg is not None else 1000.0
            if depth_scale > 0:
                gt_depth_f /= depth_scale

            pose = pred_poses[idx]
            if pose is None:
                self._paper_fig_status = f"No pose for frame {idx}."
                self._paper_fig_generating = False
                return
            pose = np.asarray(pose, dtype=np.float32)

            rendered_rgb = self.gs.render_rgb_at_pose(pose)
            if rendered_rgb is None:
                self._paper_fig_status = f"Failed to render RGB for frame {idx}."
                self._paper_fig_generating = False
                return

            # Convention: we treat both GT and rendered as OpenCV BGR uint8.
            rendered_bgr = np.asarray(rendered_rgb)
            if rendered_bgr.ndim != 3 or rendered_bgr.shape[2] != 3:
                self._paper_fig_status = f"Rendered RGB has invalid shape for frame {idx}."
                self._paper_fig_generating = False
                return
            if rendered_bgr.dtype != np.uint8:
                rendered_bgr = np.clip(rendered_bgr, 0, 255).astype(np.uint8)

            rendered_depth = self.gs.render_depth_at_pose(pose)
            if rendered_depth is None:
                self._paper_fig_status = f"Failed to render depth for frame {idx}."
                self._paper_fig_generating = False
                return

            seg_rgb = None
            try:
                seg_rgb = self.gs.render_segmentation_rgb_at_pose(pose)
            except Exception:
                seg_rgb = None
            if seg_rgb is None:
                seg_rgb = np.full_like(rendered_bgr, 255, dtype=np.uint8)
            else:
                seg_rgb = np.asarray(seg_rgb)
                if seg_rgb.ndim != 3 or seg_rgb.shape[2] != 3:
                    seg_rgb = np.full_like(rendered_bgr, 255, dtype=np.uint8)
                elif seg_rgb.dtype != np.uint8:
                    seg_rgb = np.clip(seg_rgb, 0, 255).astype(np.uint8)

            if target_h is None:
                target_h, target_w = gt_rgb.shape[:2]

            gt_rgb = cv2.resize(gt_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            rendered_bgr = cv2.resize(rendered_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            gt_depth_f = cv2.resize(gt_depth_f, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            rendered_depth = cv2.resize(rendered_depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            seg_rgb = cv2.resize(seg_rgb, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

            gt_rgbs.append(gt_rgb)
            gt_depths.append(gt_depth_f)
            rendered_rgbs.append(rendered_bgr)
            rendered_depths.append(rendered_depth)
            seg_rgbs.append(seg_rgb)

        gt_depth_vis = [colorize_depth_percentile(d) for d in gt_depths]
        rd_depth_vis = [colorize_depth_percentile(d) for d in rendered_depths]

        if rendered_rgbs[0].dtype != np.uint8:
            rendered_rgbs = [np.clip(r, 0, 255).astype(np.uint8) for r in rendered_rgbs]

        row_gt_rgb = hconcat_with_pad(gt_rgbs)
        row_gt_depth = hconcat_with_pad(gt_depth_vis)
        row_rd_rgb = hconcat_with_pad(rendered_rgbs)
        row_rd_depth = hconcat_with_pad(rd_depth_vis)
        row_seg = hconcat_with_pad(seg_rgbs)

        figure = np.concatenate([row_gt_rgb, row_gt_depth, row_rd_rgb, row_rd_depth, row_seg], axis=0)

        out_dir = self.output_root / "paper_figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"paper_figure_{timestamp}.png"
        cv2.imwrite(str(out_path), figure)

        self._paper_fig_status = f"Saved: {out_path}"
        self._paper_fig_generating = False

    def _update_previews(
        self,
        frame_index: int,
        gt_rgb: np.ndarray,
        rendered_rgb: np.ndarray,
        seg_rgb: np.ndarray | None = None,
    ) -> None:
        with self._preview_lock:
            self._latest_frame_index = int(frame_index)
            self._latest_gt_rgb = np.asarray(gt_rgb).copy()
            self._latest_rendered_rgb = np.asarray(rendered_rgb).copy()
            self._latest_seg_rgb = None if seg_rgb is None else np.asarray(seg_rgb).copy()

    def _benchmark_thread(self):
        self._running = True
        self._progress = 0.0
        psnr_sum = 0.0
        ssim_sum = 0.0
        count = 0
        self._run_dir = None
        self._save_status = ""
        self._has_unsaved_results = False
        self._bench_metrics_rows = []
        self._bench_pose_rows = []
        self._bench_pred_poses = None
        self._bench_gt_poses = None
        with self._traj_lock:
            self._traj_metrics = None
            self._traj_status = "Running"

        # Require dataset to have RGBD_pairs and tracker poses.
        pairs = getattr(self.dataset, "RGBD_pairs", None)
        pred_poses = None
        if self.tracker is not None:
            pred_poses = list(getattr(self.tracker, "poses", []))
        if pairs is None or pred_poses is None or len(pred_poses) == 0:
            with self._traj_lock:
                self._traj_metrics = None
                if pairs is None:
                    self._traj_status = "RGBD pairs not available."
                else:
                    self._traj_status = "No predicted poses available."
            self._running = False
            return

        n = min(len(pairs), len(pred_poses))
        metrics_rows = []
        pose_rows = []
        for i in range(n):
            pose = pred_poses[i]
            if pose is None:
                continue
            pose = np.asarray(pose, dtype=np.float32)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                continue
            # Render predicted RGB at predicted pose
            try:
                pred = self.gs.render_rgb_at_pose(pose)
            except Exception:
                pred = None

            if pred is None:
                # skip
                continue

            try:
                seg = self.gs.render_segmentation_rgb_at_pose(pose)
            except Exception:
                seg = None
            if seg is None:
                seg = np.zeros_like(pred, dtype=np.uint8)

            # Load GT RGB (OpenCV BGR) directly from disk (no saving).
            gt_path = pairs[i][0]
            try:
                import cv2

                gt_bgr = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
            except Exception:
                gt_bgr = None
            if gt_bgr is None:
                continue

            # Convention: evaluate in OpenCV BGR uint8.
            pred_bgr = np.asarray(pred, dtype=np.uint8).copy()
            seg_bgr = np.asarray(seg, dtype=np.uint8)
            if seg_bgr.ndim == 3 and seg_bgr.shape[2] == 3:
                # Match the same channel order used for previews/metrics.
                seg_bgr = seg_bgr[..., ::-1].copy()
            else:
                seg_bgr = np.zeros_like(pred_bgr, dtype=np.uint8)

            # Ensure same size
            if gt_bgr.shape[:2] != pred_bgr.shape[:2]:
                try:
                    import cv2

                    pred_bgr = cv2.resize(pred_bgr, (gt_bgr.shape[1], gt_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
                    seg_bgr = cv2.resize(seg_bgr, (gt_bgr.shape[1], gt_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
                except Exception:
                    continue

            # `gt_bgr` / `pred_bgr` are OpenCV BGR.
            self._update_previews(i, gt_bgr, pred_bgr, seg_rgb=seg_bgr)

            # Convert to float [0,1]
            gt_f = (gt_bgr.astype(np.float32) / 255.0).clip(0.0, 1.0)
            pred_f = (pred_bgr.astype(np.float32) / 255.0).clip(0.0, 1.0)

            mse = float(np.mean((gt_f - pred_f) ** 2))
            if mse <= 0.0:
                psnr = float('inf')
            else:
                psnr = 10.0 * float(np.log10(1.0 / mse))

            # SSIM via pytorch_msssim if available
            ssim_val = 0.0
            try:
                from pytorch_msssim import ssim as _ssim
                import torch as _torch

                p = _torch.from_numpy(pred_f.transpose(2, 0, 1)).unsqueeze(0).to(_torch.float32)
                g = _torch.from_numpy(gt_f.transpose(2, 0, 1)).unsqueeze(0).to(_torch.float32)
                ssim_val = float(_ssim(p, g, data_range=1.0, size_average=True).cpu().item())
            except Exception:
                # fallback: simple luminance-based approximation
                try:
                    pred_y = 0.2126 * pred_f[..., 0] + 0.7152 * pred_f[..., 1] + 0.0722 * pred_f[..., 2]
                    gt_y = 0.2126 * gt_f[..., 0] + 0.7152 * gt_f[..., 1] + 0.0722 * gt_f[..., 2]
                    # crude local windowless SSIM approximation
                    ux = pred_y.mean()
                    uy = gt_y.mean()
                    vx = pred_y.var()
                    vy = gt_y.var()
                    cxy = float(((pred_y - ux) * (gt_y - uy)).mean())
                    C1 = (0.01 ** 2)
                    C2 = (0.03 ** 2)
                    ssim_val = ((2 * ux * uy + C1) * (2 * cxy + C2)) / ((ux * ux + uy * uy + C1) * (vx + vy + C2))
                    ssim_val = float(np.clip(ssim_val, 0.0, 1.0))
                except Exception:
                    ssim_val = 0.0

            psnr_sum += psnr if np.isfinite(psnr) else 0.0
            ssim_sum += ssim_val
            count += 1
            metrics_rows.append((i, psnr, ssim_val))
            pose_rows.append((i, pose))
            self._progress = (i + 1) / float(n)

        # finalize
        self._result_count = count
        self._avg_psnr = (psnr_sum / count) if count > 0 else 0.0
        self._avg_ssim = (ssim_sum / count) if count > 0 else 0.0
        gt_poses = getattr(self.dataset, "gt_poses", None)
        traj_metrics, traj_status = self._compute_trajectory_metrics(pred_poses, gt_poses)
        with self._traj_lock:
            self._traj_metrics = traj_metrics
            self._traj_status = traj_status if traj_status else "OK"
        # Keep results in-memory; user explicitly clicks Save to write CSVs.
        self._bench_metrics_rows = list(metrics_rows)
        self._bench_pose_rows = list(pose_rows)
        self._bench_pred_poses = list(pred_poses)
        self._bench_gt_poses = list(gt_poses) if isinstance(gt_poses, (list, tuple)) else None
        self._has_unsaved_results = True
        self._running = False

    def start_benchmark(self):
        if self._running:
            return
        self._thread = threading.Thread(target=self._benchmark_thread, daemon=True)
        self._thread.start()

    def draw(self):
        if not self.is_open:
            return

        opened, self.is_open = imgui.begin(self.title, self.is_open)
        if not opened:
            imgui.end()
            return

        if imgui.button("Start Benchmark"):
            if not self._running:
                self.start_benchmark()

        imgui.same_line()
        can_save = (not self._running) and bool(self._has_unsaved_results)
        if not can_save:
            imgui.begin_disabled(True)
        if imgui.button("Save"):
            self._save_results_to_disk()
        if not can_save:
            imgui.end_disabled()

        if self._save_status:
            imgui.same_line()
            imgui.text_wrapped(self._save_status)

        if self._running:
            imgui.same_line()
            imgui.text("Running…")

        # Progress
        if self._running:
            imgui.progress_bar(self._progress, size_arg=imgui.get_content_region_avail())
        else:
            imgui.progress_bar(1.0 if self._result_count > 0 else 0.0, size_arg=imgui.get_content_region_avail())

        # TABLE IX — runtime pipeline wall-time + memory, reported after trajectory ends.
        if self.bench is not None:
            try:
                rep = self.bench.get_table_ix()
            except Exception:
                rep = None

            if isinstance(rep, dict):
                status = rep.get("status", {}) if isinstance(rep.get("status", {}), dict) else {}
                rows = rep.get("rows", []) if isinstance(rep.get("rows", []), list) else []
                mem = rep.get("memory", {}) if isinstance(rep.get("memory", {}), dict) else {}

                imgui.separator()
                imgui.text("TABLE IX — Wall-time per stage on a single GPU")

                traj_ended = bool(status.get("trajectory_ended", False))
                finalized = bool(status.get("finalized", False))
                fin_msg = str(status.get("finalize_status", ""))
                if not traj_ended:
                    imgui.text("Status: collecting (waiting for trajectory end)")
                elif not finalized:
                    imgui.text("Status: trajectory ended, draining workers…")
                else:
                    imgui.text(f"Status: {fin_msg}")

                table_flags = (
                    imgui.TableFlags_.borders
                    | imgui.TableFlags_.row_bg
                    | imgui.TableFlags_.sizing_fixed_fit
                )
                if imgui.begin_table("table_ix_walltime", 3, flags=table_flags):
                    imgui.table_setup_column("Stage")
                    imgui.table_setup_column("ms / frame")
                    imgui.table_setup_column("Hz")
                    imgui.table_headers_row()
                    for r in rows:
                        if not isinstance(r, dict):
                            continue
                        stage = str(r.get("stage", ""))
                        avg_ms = r.get("avg_ms", None)
                        hz = r.get("hz", None)
                        imgui.table_next_row()
                        imgui.table_set_column_index(0)
                        imgui.text(stage)
                        imgui.table_set_column_index(1)
                        imgui.text("—" if avg_ms is None else f"{float(avg_ms):.3f}")
                        imgui.table_set_column_index(2)
                        imgui.text("—" if hz is None else f"{float(hz):.2f}")
                    imgui.end_table()

                imgui.separator()
                imgui.text("Component (GB)")
                if imgui.begin_table("table_ix_memory", 2, flags=table_flags):
                    imgui.table_setup_column("Component")
                    imgui.table_setup_column("GB")
                    imgui.table_headers_row()

                    def _mem_row(label: str, key: str):
                        v = mem.get(key, None)
                        imgui.table_next_row()
                        imgui.table_set_column_index(0)
                        imgui.text(label)
                        imgui.table_set_column_index(1)
                        imgui.text("—" if v is None else f"{float(v):.3f}")

                    _mem_row("Mask2Former weights", "mask2former_weights_gb")
                    _mem_row("Relation head + EMA", "relation_head_ema_gb")
                    _mem_row("Gaussian map steady state", "gaussian_map_steady_gb")
                    _mem_row("Activation peak during mapping", "mapping_activation_peak_gb")
                    _mem_row("Total steady state", "total_steady_gb")
                    _mem_row("Total peak", "total_peak_gb")
                    imgui.end_table()
            else:
                imgui.separator()
                imgui.text("TABLE IX — Wall-time per stage on a single GPU")
                imgui.text("Status: unavailable")

        imgui.separator()
        imgui.text(f"Frames evaluated: {self._result_count}")
        imgui.text(f"Avg PSNR: {self._avg_psnr:.3f}")
        imgui.text(f"Avg SSIM: {self._avg_ssim:.4f}")
        if self._run_dir is not None:
            imgui.text(f"Saved to: {self._run_dir}")

        with self._traj_lock:
            traj_metrics = None if self._traj_metrics is None else dict(self._traj_metrics)
            traj_status = str(self._traj_status)

        imgui.separator()
        imgui.text("Trajectory metrics (pred vs GT)")
        if traj_metrics is None:
            imgui.text(traj_status)
        else:
            imgui.text(f"Samples: {traj_metrics['num_samples']}")
            imgui.text(f"Alignment scale: {traj_metrics['alignment_scale']:.4f}")
            imgui.text(f"ATE raw RMSE: {traj_metrics['ate_raw_rmse_m']:.4f} m")
            imgui.text(
                "ATE aligned RMSE/mean/median/std/max: "
                f"{traj_metrics['ate_aligned_rmse_m']:.4f}/"
                f"{traj_metrics['ate_aligned_mean_m']:.4f}/"
                f"{traj_metrics['ate_aligned_median_m']:.4f}/"
                f"{traj_metrics['ate_aligned_std_m']:.4f}/"
                f"{traj_metrics['ate_aligned_max_m']:.4f} m"
            )
            imgui.text(
                "RPE trans RMSE/mean: "
                f"{traj_metrics['rpe_trans_rmse_m']:.4f}/"
                f"{traj_metrics['rpe_trans_mean_m']:.4f} m"
            )
            imgui.text(
                "RPE rot RMSE/mean: "
                f"{traj_metrics['rpe_rot_rmse_deg']:.3f}/"
                f"{traj_metrics['rpe_rot_mean_deg']:.3f} deg"
            )
            imgui.text(
                "Path length pred/gt: "
                f"{traj_metrics['pred_path_len_m']:.3f}/"
                f"{traj_metrics['gt_path_len_m']:.3f} m "
                f"(ratio {traj_metrics['path_len_ratio']:.3f})"
            )

        imgui.separator()
        imgui.text("Paper Figure")
        for fi in range(5):
            imgui.set_next_item_width(80)
            changed, val = imgui.input_int(f"##paper_idx_{fi}", self._paper_frame_indices[fi])
            if changed:
                self._paper_frame_indices[fi] = max(0, val)
            if fi < 4:
                imgui.same_line()

        if imgui.button("Generate Paper Figure") and not self._paper_fig_generating:
            indices = list(self._paper_frame_indices)
            threading.Thread(
                target=self._generate_paper_figure, args=(indices,), daemon=True
            ).start()
        if self._paper_fig_generating:
            imgui.same_line()
            imgui.text("Generating...")
        if self._paper_fig_status:
            imgui.text_wrapped(self._paper_fig_status)

        with self._preview_lock:
            gt_preview = None if self._latest_gt_rgb is None else self._latest_gt_rgb.copy()
            rendered_preview = None if self._latest_rendered_rgb is None else self._latest_rendered_rgb.copy()
            seg_preview = None if self._latest_seg_rgb is None else self._latest_seg_rgb.copy()
            frame_index = self._latest_frame_index

        if gt_preview is not None and rendered_preview is not None:
            imgui.separator()
            imgui.text(f"Current preview frame: {frame_index}")

            avail_w = float(imgui.get_content_region_avail().x)
            preview_w = max(1, int((avail_w - 8.0) * 0.5))
            preview_h = 320
            gt_preview = self._fit_preview(gt_preview, preview_w, preview_h)
            rendered_preview = self._fit_preview(rendered_preview, preview_w, preview_h)

            if seg_preview is not None:
                seg_row_w = max(1, int(avail_w))
                seg_preview = self._fit_preview(seg_preview, seg_row_w, preview_h)

            self._gt_widget.set_image_rgb(gt_preview)
            self._rendered_widget.set_image_rgb(rendered_preview)
            if seg_preview is not None:
                self._seg_widget.set_image_rgb(seg_preview)

            imgui.begin_child("benchmark_gt_preview", (preview_w, preview_h + 24), True)
            imgui.text("GT")
            self._gt_widget.draw(fit_to_window=True)
            imgui.end_child()

            imgui.same_line()

            imgui.begin_child("benchmark_rendered_preview", (preview_w, preview_h + 24), True)
            imgui.text("Rendered")
            self._rendered_widget.draw(fit_to_window=True)
            imgui.end_child()

            if seg_preview is not None:
                imgui.begin_child("benchmark_seg_preview", (seg_row_w, preview_h + 24), True)
                imgui.text("Segmentation")
                self._seg_widget.draw(fit_to_window=True)
                imgui.end_child()

        imgui.end()
