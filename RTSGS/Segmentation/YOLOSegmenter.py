import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from RTSGS.GaussianSplatting.GaussianSplating import _build_K, frustum_cull_mask


class YOLOSemanticSegmenter:
    def __init__(self, pcd, config, project_root: str):
        self.pcd = pcd
        self.config = config
        self.project_root = project_root

        self.enabled = bool(config.get("yolo_segmentation_enabled", True))
        self.model_path = self._abs_path(str(config.get("yolo_model_path", "Models/yolo26x-seg.pt")))
        self.min_confidence = float(config.get("yolo_min_confidence", 0.25))
        self.max_detections = int(config.get("yolo_max_detections", 64))
        self.max_points_per_detection = int(config.get("yolo_max_points_per_detection", 8000))
        self.semantic_voxel_size = float(config.get("semantic_voxel_size", config.get("voxel_size", 0.02)))
        self.semantic_depth_tolerance_m = float(config.get("semantic_depth_tolerance_m", 0.08))
        self.cull_near = float(config.get("semantic_cull_near", 0.05))
        self.cull_far = float(config.get("semantic_cull_far", 50.0))
        self.cull_pad_px = float(config.get("semantic_cull_pad_px", 2.0))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_str = "cuda:0" if self.device.type == "cuda" else "cpu"

        self._model = None
        self._names = []
        self._palette_t = None
        self._grid_cache = {}

        self._cache_num_points = -1
        self._gauss_keys_sorted = None
        self._gauss_idx_sorted = None

    def _abs_path(self, p: str) -> str:
        if os.path.isabs(p):
            return p
        return os.path.join(self.project_root, p)

    def start(self):
        if not self.enabled:
            print("[YOLO] Semantic segmentation disabled by config (yolo_segmentation_enabled=False).")
            return
        if not torch.cuda.is_available():
            print("[YOLO] CUDA unavailable. Semantic fusion disabled.")
            self.enabled = False
            return
        self._ensure_model()

    def stop(self):
        return

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        if not os.path.exists(self.model_path):
            print(f"[YOLO] Model not found: {self.model_path}")
            return False

        model = YOLO(self.model_path)
        if self.device.type == "cuda":
            model.to(self.device_str)

        names = model.names if hasattr(model, "names") else {}
        if isinstance(names, dict):
            max_id = max(names.keys()) if len(names) > 0 else -1
            self._names = [str(names.get(i, f"class_{i}")) for i in range(max_id + 1)]
        elif isinstance(names, (list, tuple)):
            self._names = [str(x) for x in names]
        else:
            self._names = []

        n_classes = max(1, len(self._names))
        g = torch.Generator(device=self.device)
        g.manual_seed(123)
        self._palette_t = torch.rand((n_classes, 3), generator=g, device=self.device, dtype=torch.float32) * 0.85 + 0.1

        self._model = model
        print(f"[YOLO] Loaded segmentation model: {self.model_path}")
        return True

    def _get_depth_grid(self, h: int, w: int):
        key = (h, w)
        cached = self._grid_cache.get(key, None)
        if cached is not None:
            return cached

        v, u = torch.meshgrid(
            torch.arange(h, device=self.device, dtype=torch.float32),
            torch.arange(w, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        self._grid_cache[key] = (u.reshape(-1), v.reshape(-1))
        return self._grid_cache[key]

    def _prepare_gaussian_lookup(self):
        with self.pcd.lock:
            points = self.pcd.all_points
            n = 0 if points is None else int(points.shape[0])

        if n <= 0 or points is None:
            self._cache_num_points = -1
            self._gauss_keys_sorted = None
            self._gauss_idx_sorted = None
            return n

        if self._cache_num_points == n and self._gauss_keys_sorted is not None and self._gauss_idx_sorted is not None:
            return n

        vox = torch.floor(points / self.semantic_voxel_size).to(torch.int64)
        keys = self.pcd._pack_voxels(vox)
        order = torch.argsort(keys)
        self._gauss_keys_sorted = keys[order]
        self._gauss_idx_sorted = torch.arange(n, device=self.device, dtype=torch.long)[order]
        self._cache_num_points = n
        return n

    def _match_points_to_gaussians(
        self,
        points_world: torch.Tensor,
        observed_depth: torch.Tensor | None = None,
        gauss_depth_cam: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if points_world.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=self.device)
        if self._gauss_keys_sorted is None or self._gauss_idx_sorted is None:
            return torch.empty((0,), dtype=torch.long, device=self.device)

        vox = torch.floor(points_world / self.semantic_voxel_size).to(torch.int64)
        q_keys = self.pcd._pack_voxels(vox)

        q_unique, q_inv = torch.unique(q_keys, return_inverse=True)
        left = torch.searchsorted(self._gauss_keys_sorted, q_unique, right=False)
        right = torch.searchsorted(self._gauss_keys_sorted, q_unique, right=True)

        matched_chunks = []
        use_depth_gate = observed_depth is not None and gauss_depth_cam is not None
        tol = float(self.semantic_depth_tolerance_m)

        for i in range(int(q_unique.shape[0])):
            li = int(left[i].item())
            ri = int(right[i].item())
            if ri <= li:
                continue

            cand_idx = self._gauss_idx_sorted[li:ri]
            if cand_idx.numel() == 0:
                continue

            if use_depth_gate:
                z_obs_i = observed_depth[q_inv == i]
                if z_obs_i.numel() == 0:
                    continue

                z_g = gauss_depth_cam[cand_idx]
                valid_z = torch.isfinite(z_g)
                if not torch.any(valid_z):
                    continue

                cand_idx = cand_idx[valid_z]
                z_g = z_g[valid_z]
                dz = torch.abs(z_g[:, None] - z_obs_i[None, :])
                keep = torch.amin(dz, dim=1) <= tol
                cand_idx = cand_idx[keep]
                if cand_idx.numel() == 0:
                    continue

            matched_chunks.append(cand_idx)

        if len(matched_chunks) == 0:
            return torch.empty((0,), dtype=torch.long, device=self.device)

        return torch.unique(torch.cat(matched_chunks, dim=0))

    @torch.inference_mode()
    def process_frame(self, rgb_bgr: np.ndarray, depth_raw: np.ndarray, pose_w: np.ndarray | None):
        if not self.enabled or rgb_bgr is None or depth_raw is None:
            return
        if pose_w is None:
            return
        if not self._ensure_model():
            return

        n_map = self._prepare_gaussian_lookup()
        if n_map <= 0:
            return

        with self.pcd.lock:
            means = self.pcd.all_points

        if means is None or int(means.shape[0]) == 0:
            return

        t0 = time.perf_counter()
        yolo_results = self._model.predict(
            source=rgb_bgr,
            verbose=False,
            conf=self.min_confidence,
            max_det=self.max_detections,
            device=self.device_str,
        )
        infer_ms = (time.perf_counter() - t0) * 1000.0

        pose_t = torch.from_numpy(np.asarray(pose_w, dtype=np.float32)).to(self.device)
        T_fix = torch.eye(4, device=self.device, dtype=torch.float32)
        T_fix[:3, :3] = self.pcd.R_fix
        viewmat = torch.inverse(T_fix @ pose_t).unsqueeze(0)

        K_depth = _build_K(
            fx=float(self.pcd.fx),
            fy=float(self.pcd.fy),
            cx=float(self.pcd.cx),
            cy=float(self.pcd.cy),
            device=self.device,
        ).unsqueeze(0)

        depth_t = torch.from_numpy(np.asarray(depth_raw)).to(self.device, dtype=torch.float32)
        depth_t = depth_t / float(self.pcd.depth_scale)
        h_d, w_d = depth_t.shape[:2]
        h_r, w_r = rgb_bgr.shape[:2]

        in_frustum = frustum_cull_mask(
            means_world=means,
            viewmats=viewmat,
            Ks=K_depth,
            width=w_d,
            height=h_d,
            near=self.cull_near,
            far=self.cull_far,
            pad=self.cull_pad_px,
        )

        # Keep only Gaussians that are actually visible at the measured depth.
        visible_idx = torch.where(in_frustum)[0]
        gauss_depth_cam = torch.full((n_map,), float("inf"), dtype=torch.float32, device=self.device)
        visible_depth_idx = torch.empty((0,), dtype=torch.long, device=self.device)
        if visible_idx.numel() > 0:
            Pw = means[visible_idx]
            Pw_h = torch.cat([Pw, torch.ones((Pw.shape[0], 1), device=self.device, dtype=Pw.dtype)], dim=1)
            Pc = Pw_h @ viewmat[0].T
            X, Y, Z = Pc[:, 0], Pc[:, 1], Pc[:, 2]

            in_z = Z > 1e-6
            u = self.pcd.fx * (X / Z.clamp_min(1e-12)) + self.pcd.cx
            v = self.pcd.fy * (Y / Z.clamp_min(1e-12)) + self.pcd.cy
            ui = torch.round(u).to(torch.long)
            vi = torch.round(v).to(torch.long)
            in_img = (ui >= 0) & (ui < w_d) & (vi >= 0) & (vi < h_d)

            keep = in_z & in_img
            if torch.any(keep):
                vis_k = visible_idx[keep]
                z_k = Z[keep]
                d_k = depth_t[vi[keep], ui[keep]]
                d_ok = torch.isfinite(d_k) & (d_k > 0)
                close = torch.abs(z_k - d_k) <= float(self.semantic_depth_tolerance_m)
                depth_visible = d_ok & close
                if torch.any(depth_visible):
                    visible_depth_idx = vis_k[depth_visible]
                    gauss_depth_cam[visible_depth_idx] = z_k[depth_visible]

        if len(yolo_results) == 0:
            self.pcd.fuse_semantic_observations(
                torch.empty((0,), dtype=torch.long, device=self.device),
                torch.empty((0,), dtype=torch.long, device=self.device),
                torch.empty((0,), dtype=torch.float32, device=self.device),
                visible_gaussian_indices=visible_depth_idx,
                class_names=self._names,
                class_palette=self._palette_t,
                metadata={"inference_ms": float(infer_ms), "num_detections": 0},
            )
            return

        result = yolo_results[0]
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)
        if boxes is None or masks is None or getattr(masks, "data", None) is None:
            self.pcd.fuse_semantic_observations(
                torch.empty((0,), dtype=torch.long, device=self.device),
                torch.empty((0,), dtype=torch.long, device=self.device),
                torch.empty((0,), dtype=torch.float32, device=self.device),
                visible_gaussian_indices=visible_depth_idx,
                class_names=self._names,
                class_palette=self._palette_t,
                metadata={"inference_ms": float(infer_ms), "num_detections": 0},
            )
            return

        cls_ids = boxes.cls.to(self.device, dtype=torch.long)
        confs = boxes.conf.to(self.device, dtype=torch.float32)
        mask_data = masks.data.to(self.device, dtype=torch.float32)

        if cls_ids.numel() == 0 or mask_data.shape[0] == 0:
            self.pcd.fuse_semantic_observations(
                torch.empty((0,), dtype=torch.long, device=self.device),
                torch.empty((0,), dtype=torch.long, device=self.device),
                torch.empty((0,), dtype=torch.float32, device=self.device),
                visible_gaussian_indices=visible_depth_idx,
                class_names=self._names,
                class_palette=self._palette_t,
                metadata={"inference_ms": float(infer_ms), "num_detections": 0},
            )
            return

        if int(mask_data.shape[1]) != h_r or int(mask_data.shape[2]) != w_r:
            mask_data = F.interpolate(mask_data.unsqueeze(1), size=(h_r, w_r), mode="nearest").squeeze(1)
        mask_bool = mask_data > 0.5

        u_d, v_d = self._get_depth_grid(h_d, w_d)
        z = depth_t.reshape(-1)
        valid_d = torch.isfinite(z) & (z > 0)
        if not torch.any(valid_d):
            self.pcd.fuse_semantic_observations(
                torch.empty((0,), dtype=torch.long, device=self.device),
                torch.empty((0,), dtype=torch.long, device=self.device),
                torch.empty((0,), dtype=torch.float32, device=self.device),
                visible_gaussian_indices=visible_depth_idx,
                class_names=self._names,
                class_palette=self._palette_t,
                metadata={"inference_ms": float(infer_ms), "num_detections": int(cls_ids.numel())},
            )
            return

        z = z[valid_d]
        u_d = u_d[valid_d]
        v_d = v_d[valid_d]

        x = (u_d - self.pcd.cx) * z / self.pcd.fx
        y = (v_d - self.pcd.cy) * z / self.pcd.fy
        points_d = torch.stack((x, y, z), dim=1)

        points_rgb = (self.pcd.R_depth_to_rgb @ points_d.T).T + self.pcd.t_depth_to_rgb
        z_rgb = points_rgb[:, 2]
        valid_rgb = z_rgb > 1e-6
        if not torch.any(valid_rgb):
            self.pcd.fuse_semantic_observations(
                torch.empty((0,), dtype=torch.long, device=self.device),
                torch.empty((0,), dtype=torch.long, device=self.device),
                torch.empty((0,), dtype=torch.float32, device=self.device),
                visible_gaussian_indices=visible_depth_idx,
                class_names=self._names,
                class_palette=self._palette_t,
                metadata={"inference_ms": float(infer_ms), "num_detections": int(cls_ids.numel())},
            )
            return

        points_d = points_d[valid_rgb]
        z_rgb = z_rgb[valid_rgb]

        u_rgb = self.pcd.rgb_fx * (points_rgb[valid_rgb, 0] / z_rgb) + self.pcd.rgb_cx
        v_rgb = self.pcd.rgb_fy * (points_rgb[valid_rgb, 1] / z_rgb) + self.pcd.rgb_cy

        ui = torch.round(u_rgb).to(torch.long)
        vi = torch.round(v_rgb).to(torch.long)
        in_img = (ui >= 0) & (ui < w_r) & (vi >= 0) & (vi < h_r)
        if not torch.any(in_img):
            self.pcd.fuse_semantic_observations(
                torch.empty((0,), dtype=torch.long, device=self.device),
                torch.empty((0,), dtype=torch.long, device=self.device),
                torch.empty((0,), dtype=torch.float32, device=self.device),
                visible_gaussian_indices=visible_depth_idx,
                class_names=self._names,
                class_palette=self._palette_t,
                metadata={"inference_ms": float(infer_ms), "num_detections": int(cls_ids.numel())},
            )
            return

        points_d = points_d[in_img]
        ui = ui[in_img]
        vi = vi[in_img]

        R_corr = self.pcd.R_fix @ pose_t[:3, :3]
        t_corr = (self.pcd.R_fix @ pose_t[:3, 3].unsqueeze(-1)).squeeze(-1)
        points_world = (R_corr @ points_d.T).T + t_corr

        obs_idx = []
        obs_cls = []
        obs_conf = []

        order = torch.argsort(confs, descending=True)
        for det_i in order.tolist():
            m = mask_bool[det_i, vi, ui]
            if not torch.any(m):
                continue

            sel = torch.where(m)[0]
            g_parts = []
            chunk_size = max(1, int(self.max_points_per_detection))
            for sel_chunk in torch.split(sel, chunk_size):
                pts_sel = points_world[sel_chunk]
                z_sel = points_d[sel_chunk, 2]
                g_idx_chunk = self._match_points_to_gaussians(pts_sel, observed_depth=z_sel, gauss_depth_cam=gauss_depth_cam)
                if g_idx_chunk.numel() > 0:
                    g_parts.append(g_idx_chunk)

            if len(g_parts) == 0:
                continue

            g_idx = torch.unique(torch.cat(g_parts, dim=0))
            obs_idx.append(g_idx)
            obs_cls.append(torch.full_like(g_idx, int(cls_ids[det_i].item()), dtype=torch.long, device=self.device))
            obs_conf.append(torch.full_like(g_idx, float(confs[det_i].item()), dtype=torch.float32, device=self.device))

        if len(obs_idx) == 0:
            idx_t = torch.empty((0,), dtype=torch.long, device=self.device)
            cls_t = torch.empty((0,), dtype=torch.long, device=self.device)
            conf_t = torch.empty((0,), dtype=torch.float32, device=self.device)
        else:
            idx_t = torch.cat(obs_idx, dim=0)
            cls_t = torch.cat(obs_cls, dim=0)
            conf_t = torch.cat(obs_conf, dim=0)

        total_ms = (time.perf_counter() - t0) * 1000.0
        self.pcd.fuse_semantic_observations(
            idx_t,
            cls_t,
            conf_t,
            visible_gaussian_indices=visible_depth_idx,
            class_names=self._names,
            class_palette=self._palette_t,
            metadata={
                "timestamp": time.time(),
                "inference_ms": float(infer_ms),
                "total_ms": float(total_ms),
                "num_detections": int(cls_ids.numel()),
                "num_gaussians_observed": int(idx_t.numel()),
            },
        )
