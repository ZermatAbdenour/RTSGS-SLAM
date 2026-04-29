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
    def __init__(self, gs: GaussianSplatting, dataset, title: str = "Benchmark", output_root: str | None = None):
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
        self.output_root = Path(output_root) if output_root is not None else Path.cwd() / "benchmark_results"
        self._run_dir: Path | None = None
        self._preview_lock = threading.Lock()
        self._latest_gt_rgb: np.ndarray | None = None
        self._latest_rendered_rgb: np.ndarray | None = None
        self._latest_frame_index: int = -1
        self._gt_widget = ImageWidget()
        self._rendered_widget = ImageWidget()

    def _prepare_run_dir(self) -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_root / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "pairs").mkdir(parents=True, exist_ok=True)
        return run_dir

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

    def _update_previews(self, frame_index: int, gt_rgb: np.ndarray, rendered_rgb: np.ndarray) -> None:
        with self._preview_lock:
            self._latest_frame_index = int(frame_index)
            self._latest_gt_rgb = np.asarray(gt_rgb).copy()
            self._latest_rendered_rgb = np.asarray(rendered_rgb).copy()

    def _benchmark_thread(self):
        self._running = True
        self._progress = 0.0
        psnr_sum = 0.0
        ssim_sum = 0.0
        count = 0
        self._run_dir = self._prepare_run_dir()

        # Require dataset to have gt_poses and RGBD_pairs
        pairs = getattr(self.dataset, "RGBD_pairs", None)
        poses = getattr(self.dataset, "gt_poses", None)
        if pairs is None or poses is None:
            self._running = False
            return

        n = min(len(pairs), len(poses))
        for i in range(n):
            pose = poses[i]
            # Render predicted RGB at ground-truth pose
            try:
                pred = self.gs.render_rgb_at_pose(pose)
            except Exception:
                pred = None

            if pred is None:
                # skip
                continue

            # Load the saved images back and convert from BGR to RGB
            gt_path = pairs[i][0]
            gt = self._load_image_rgb(Path(gt_path))
            if gt is None:
                continue

            # Ensure same size
            if gt.shape[:2] != pred.shape[:2]:
                try:
                    import cv2

                    pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
                except Exception:
                    continue

            pair_dir = self._run_dir / "pairs" / f"frame_{i:06d}"
            gt_path_out = pair_dir / "gt.png"
            rendered_path_out = pair_dir / "rendered.png"
            self._save_image(gt_path_out, gt)
            self._save_image(rendered_path_out, pred)

            gt_rgb = self._load_image_rgb(gt_path_out)
            rendered_rgb = self._load_image_rgb(rendered_path_out)
            if gt_rgb is None or rendered_rgb is None:
                continue

            if gt_rgb.shape[:2] != rendered_rgb.shape[:2]:
                try:
                    import cv2

                    rendered_rgb = cv2.resize(rendered_rgb, (gt_rgb.shape[1], gt_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
                except Exception:
                    continue

            self._update_previews(i, gt_rgb, rendered_rgb)

            # Convert to float [0,1]
            gt_f = (gt_rgb.astype(np.float32) / 255.0).clip(0.0, 1.0)
            pred_f = (rendered_rgb.astype(np.float32) / 255.0).clip(0.0, 1.0)

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
            self._progress = (i + 1) / float(n)

        # finalize
        self._result_count = count
        self._avg_psnr = (psnr_sum / count) if count > 0 else 0.0
        self._avg_ssim = (ssim_sum / count) if count > 0 else 0.0
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

        if self._running:
            imgui.same_line()
            imgui.text("Running…")

        # Progress
        if self._running:
            imgui.progress_bar(self._progress, size_arg=imgui.get_content_region_avail())
        else:
            imgui.progress_bar(1.0 if self._result_count > 0 else 0.0, size_arg=imgui.get_content_region_avail())

        imgui.separator()
        imgui.text(f"Frames evaluated: {self._result_count}")
        imgui.text(f"Avg PSNR: {self._avg_psnr:.3f}")
        imgui.text(f"Avg SSIM: {self._avg_ssim:.4f}")
        if self._run_dir is not None:
            imgui.text(f"Saved to: {self._run_dir}")

        with self._preview_lock:
            gt_preview = None if self._latest_gt_rgb is None else self._latest_gt_rgb.copy()
            rendered_preview = None if self._latest_rendered_rgb is None else self._latest_rendered_rgb.copy()
            frame_index = self._latest_frame_index

        if gt_preview is not None and rendered_preview is not None:
            imgui.separator()
            imgui.text(f"Current preview frame: {frame_index}")

            avail_w = float(imgui.get_content_region_avail().x)
            preview_w = max(1, int((avail_w - 8.0) * 0.5))
            preview_h = 320
            gt_preview = self._fit_preview(gt_preview, preview_w, preview_h)
            rendered_preview = self._fit_preview(rendered_preview, preview_w, preview_h)

            self._gt_widget.set_image_rgb(gt_preview)
            self._rendered_widget.set_image_rgb(rendered_preview)

            imgui.begin_child("benchmark_gt_preview", (preview_w, preview_h + 24), True)
            imgui.text("GT")
            self._gt_widget.draw(fit_to_window=True)
            imgui.end_child()

            imgui.same_line()

            imgui.begin_child("benchmark_rendered_preview", (preview_w, preview_h + 24), True)
            imgui.text("Rendered")
            self._rendered_widget.draw(fit_to_window=True)
            imgui.end_child()

        imgui.end()
