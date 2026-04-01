import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from RTSGS.GaussianSplatting.GaussianSplating import _build_K, frustum_cull_mask


class YOLOSemanticSegmenter:
    def __init__(self, pcd, config, project_root: str):
        self.pcd = pcd
        self.config = config
        self.project_root = project_root

        self.enabled = bool(config.get("yolo_segmentation_enabled", True))
        self.model_id = str(
            config.get("mask2former_model_id", config.get("yolo_model_path", "facebook/mask2former-swin-base-ade-semantic"))
        )
        self.min_confidence = float(config.get("yolo_min_confidence", 0.6))
        self.assignment_confidence_gate = float(config.get("semantic_assignment_min_confidence", 0.4))
        self.max_detections = int(config.get("yolo_max_detections", 64))
        self.semantic_depth_tolerance_m = float(config.get("semantic_depth_tolerance_m", 0.04))
        self.semantic_mask_threshold = float(config.get("semantic_mask_threshold", 0.5))
        self.semantic_mask_erode_px = int(config.get("semantic_mask_erode_px", 10))
        self.semantic_mask_pad_px = int(config.get("semantic_mask_pad_px", 0))
        self.semantic_depth_edge_reject_enabled = bool(config.get("semantic_depth_edge_reject_enabled", True))
        self.semantic_depth_edge_threshold_m = float(config.get("semantic_depth_edge_threshold_m", 0.05))
        self.cull_near = float(config.get("semantic_cull_near", 0.05))
        self.cull_far = float(config.get("semantic_cull_far", 50.0))
        self.cull_pad_px = float(config.get("semantic_cull_pad_px", 2.0))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_str = "cuda:0" if self.device.type == "cuda" else "cpu"

        self._model = None
        self._processor = None
        self._label_ids_to_fuse = set()
        self._names = []
        self._raw_label_to_local = {}
        self._palette_t = None
        self._grid_cache = {}

    def _abs_path(self, p: str) -> str:
        if os.path.isabs(p):
            return p
        return os.path.join(self.project_root, p)

    def start(self):
        if not self.enabled:
            print("[Mask2Former] Semantic segmentation disabled by config (yolo_segmentation_enabled=False).")
            return
        if not torch.cuda.is_available():
            print("[Mask2Former] CUDA unavailable. Semantic fusion disabled.")
            self.enabled = False
            return
        self._ensure_model()

    def stop(self):
        return

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        try:
            processor = AutoImageProcessor.from_pretrained(self.model_id)
            model = Mask2FormerForUniversalSegmentation.from_pretrained(self.model_id).to(self.device)
            model.eval()
        except Exception as e:
            print(f"[Mask2Former] Failed to load model '{self.model_id}': {e}")
            return False

        id2label = getattr(model.config, "id2label", {}) or {}
        raw_ids = sorted(int(k) for k in id2label.keys())
        self._raw_label_to_local = {rid: i for i, rid in enumerate(raw_ids)}
        self._names = [str(id2label[rid]) for rid in raw_ids]
        if len(self._names) == 0:
            self._names = ["class_0"]
            self._raw_label_to_local = {0: 0}

        n_classes = max(1, len(self._names))
        g = torch.Generator(device=self.device)
        g.manual_seed(123)
        self._palette_t = torch.rand((n_classes, 3), generator=g, device=self.device, dtype=torch.float32) * 0.85 + 0.1

        self._model = model
        self._processor = processor
        cfg_fuse = getattr(model.config, "label_ids_to_fuse", None)
        if cfg_fuse is None:
            self._label_ids_to_fuse = set()
        else:
            self._label_ids_to_fuse = {int(x) for x in cfg_fuse}
        print(f"[Mask2Former] Loaded segmentation model: {self.model_id}")
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

    def _update_debug_image(self, img_bgr: np.ndarray | None):
        if img_bgr is None:
            return
        try:
            img = np.asarray(img_bgr, dtype=np.uint8)
            if img.ndim != 3 or img.shape[2] != 3:
                return
            if not img.flags["C_CONTIGUOUS"]:
                img = np.ascontiguousarray(img)
            with self.pcd.lock:
                self.pcd.segmentation_debug_image_bgr = img.copy()
                self.pcd.segmentation_debug_timestamp = time.time()
        except Exception:
            return

    def _erode_mask_bool(self, mask_bool: torch.Tensor) -> torch.Tensor:
        r = max(0, int(self.semantic_mask_erode_px))
        if r <= 0 or mask_bool.numel() == 0:
            return mask_bool

        # Binary erosion via max-pooling the inverted mask.
        k = 2 * r + 1
        x = mask_bool.to(dtype=torch.float32).unsqueeze(1)
        eroded = 1.0 - F.max_pool2d(1.0 - x, kernel_size=k, stride=1, padding=r)
        return eroded.squeeze(1) > 0.5

    def _depth_edge_mask(self, depth_m: torch.Tensor) -> torch.Tensor:
        if depth_m.numel() == 0:
            return torch.zeros_like(depth_m, dtype=torch.bool)

        d = depth_m
        valid = torch.isfinite(d) & (d > 0)
        thr = float(self.semantic_depth_edge_threshold_m)

        edge = torch.zeros_like(d, dtype=torch.bool)

        if d.shape[1] > 1:
            pair_valid_x = valid[:, 1:] & valid[:, :-1]
            gx = torch.abs(d[:, 1:] - d[:, :-1])
            ex = pair_valid_x & (gx > thr)
            edge[:, 1:] |= ex
            edge[:, :-1] |= ex

        if d.shape[0] > 1:
            pair_valid_y = valid[1:, :] & valid[:-1, :]
            gy = torch.abs(d[1:, :] - d[:-1, :])
            ey = pair_valid_y & (gy > thr)
            edge[1:, :] |= ey
            edge[:-1, :] |= ey

        return edge

    def _dilate_mask_bool(self, mask_bool: torch.Tensor) -> torch.Tensor:
        r = max(0, int(self.semantic_mask_pad_px))
        if r <= 0 or mask_bool.numel() == 0:
            return mask_bool

        # Binary dilation expands mask support around edges by r pixels.
        k = 2 * r + 1
        x = mask_bool.to(dtype=torch.float32).unsqueeze(1)
        dilated = F.max_pool2d(x, kernel_size=k, stride=1, padding=r)
        return dilated.squeeze(1) > 0.5


    def _build_debug_image(self, rgb_bgr: np.ndarray, segmentation: np.ndarray, segments_info: list[dict]) -> np.ndarray:
        img = np.asarray(rgb_bgr, dtype=np.uint8)
        if img.ndim != 3 or img.shape[2] != 3:
            return img

        out = img.astype(np.float32)
        palette_np = (self._palette_t.detach().cpu().numpy() * 255.0).astype(np.float32)
        for seg in segments_info:
            seg_id = int(seg.get("id", -1))
            raw_label = int(seg.get("label_id", -1))
            local_label = self._raw_label_to_local.get(raw_label, -1)
            if local_label < 0 or local_label >= int(palette_np.shape[0]):
                continue
            mask = segmentation == seg_id
            if not np.any(mask):
                continue
            color = palette_np[local_label]
            out[mask] = 0.4 * out[mask] + 0.6 * color[None, :]
        return np.clip(out, 0, 255).astype(np.uint8)

    @torch.inference_mode()
    def process_frame(self, rgb_bgr: np.ndarray, depth_raw: np.ndarray, pose_w: np.ndarray | None):
        if not self.enabled or rgb_bgr is None or depth_raw is None:
            return
        if pose_w is None:
            return
        if not self._ensure_model():
            return

        with self.pcd.lock:
            means = self.pcd.all_points

        if means is None or int(means.shape[0]) == 0:
            return
        n_map = int(means.shape[0])

        t0 = time.perf_counter()
        rgb_np = np.asarray(rgb_bgr, dtype=np.uint8)
        rgb_np = rgb_np[..., ::-1].copy()
        inputs = self._processor(images=rgb_np, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self._model(**inputs)

        t_post0 = time.perf_counter()
        panoptic = self._processor.post_process_panoptic_segmentation(
            outputs,
            target_sizes=[(int(rgb_bgr.shape[0]), int(rgb_bgr.shape[1]))],
            label_ids_to_fuse=self._label_ids_to_fuse,
        )[0]
        segmentation_cpu_t = panoptic["segmentation"].detach()
        segmentation_t = segmentation_cpu_t.to(self.device, dtype=torch.int64)
        segmentation = segmentation_cpu_t.cpu().numpy()
        segments_info = list(panoptic.get("segments_info", []))
        post_ms = (time.perf_counter() - t_post0) * 1000.0
        infer_ms = (time.perf_counter() - t0) * 1000.0

        t_dbg0 = time.perf_counter()
        debug_img_bgr = self._build_debug_image(np.asarray(rgb_bgr, dtype=np.uint8), segmentation, segments_info)
        self._update_debug_image(debug_img_bgr)
        debug_ms = (time.perf_counter() - t_dbg0) * 1000.0

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
        visible_depth_idx = torch.empty((0,), dtype=torch.long, device=self.device)
        gauss_depth_cam = torch.full((n_map,), float("inf"), dtype=torch.float32, device=self.device)
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

        if len(segments_info) == 0:
            self.pcd.fuse_semantic_observations(
                torch.empty((0,), dtype=torch.long, device=self.device),
                torch.empty((0,), dtype=torch.long, device=self.device),
                torch.empty((0,), dtype=torch.float32, device=self.device),
                visible_gaussian_indices=visible_idx,
                class_names=self._names,
                class_palette=self._palette_t,
                pred_instances=[],
                metadata={"inference_ms": float(infer_ms), "num_detections": 0},
            )
            return

        n_low_conf_discarded = 0

        if visible_depth_idx.numel() == 0:
            self.pcd.fuse_semantic_observations(
                torch.empty((0,), dtype=torch.long, device=self.device),
                torch.empty((0,), dtype=torch.long, device=self.device),
                torch.empty((0,), dtype=torch.float32, device=self.device),
                visible_gaussian_indices=visible_idx,
                class_names=self._names,
                class_palette=self._palette_t,
                pred_instances=[],
                metadata={"inference_ms": float(infer_ms), "num_detections": int(len(segments_info))},
            )
            return

        t_geom1 = time.perf_counter()
        depth_edge = self._depth_edge_mask(depth_t) if self.semantic_depth_edge_reject_enabled else torch.zeros_like(depth_t, dtype=torch.bool)

        Pw = means[visible_depth_idx]
        Pw_h = torch.cat([Pw, torch.ones((Pw.shape[0], 1), device=self.device, dtype=Pw.dtype)], dim=1)
        Pc = Pw_h @ viewmat[0].T

        Xd, Yd, Zd = Pc[:, 0], Pc[:, 1], Pc[:, 2]
        ud = torch.round(self.pcd.fx * (Xd / Zd.clamp_min(1e-12)) + self.pcd.cx).to(torch.long)
        vd = torch.round(self.pcd.fy * (Yd / Zd.clamp_min(1e-12)) + self.pcd.cy).to(torch.long)
        in_depth = (ud >= 0) & (ud < w_d) & (vd >= 0) & (vd < h_d)
        if not torch.any(in_depth):
            self.pcd.fuse_semantic_observations(
                torch.empty((0,), dtype=torch.long, device=self.device),
                torch.empty((0,), dtype=torch.long, device=self.device),
                torch.empty((0,), dtype=torch.float32, device=self.device),
                visible_gaussian_indices=visible_idx,
                class_names=self._names,
                class_palette=self._palette_t,
                pred_instances=[],
                metadata={"inference_ms": float(infer_ms), "num_detections": int(len(segments_info))},
            )
            return

        vis_idx = visible_depth_idx[in_depth]
        Pc = Pc[in_depth]
        ud = ud[in_depth]
        vd = vd[in_depth]
        edge_ok = ~depth_edge[vd, ud]
        if not torch.any(edge_ok):
            self.pcd.fuse_semantic_observations(
                torch.empty((0,), dtype=torch.long, device=self.device),
                torch.empty((0,), dtype=torch.long, device=self.device),
                torch.empty((0,), dtype=torch.float32, device=self.device),
                visible_gaussian_indices=visible_idx,
                class_names=self._names,
                class_palette=self._palette_t,
                pred_instances=[],
                metadata={"inference_ms": float(infer_ms), "num_detections": int(len(segments_info))},
            )
            return

        vis_idx = vis_idx[edge_ok]
        Pc = Pc[edge_ok]

        points_rgb = (self.pcd.R_depth_to_rgb @ Pc[:, :3].T).T + self.pcd.t_depth_to_rgb
        z_rgb = points_rgb[:, 2]
        valid_rgb = z_rgb > 1e-6
        if not torch.any(valid_rgb):
            self.pcd.fuse_semantic_observations(
                torch.empty((0,), dtype=torch.long, device=self.device),
                torch.empty((0,), dtype=torch.long, device=self.device),
                torch.empty((0,), dtype=torch.float32, device=self.device),
                visible_gaussian_indices=visible_idx,
                class_names=self._names,
                class_palette=self._palette_t,
                pred_instances=[],
                metadata={"inference_ms": float(infer_ms), "num_detections": int(len(segments_info))},
            )
            return

        vis_idx = vis_idx[valid_rgb]
        points_rgb = points_rgb[valid_rgb]
        z_rgb = z_rgb[valid_rgb]

        ui = torch.round(self.pcd.rgb_fx * (points_rgb[:, 0] / z_rgb) + self.pcd.rgb_cx).to(torch.long)
        vi = torch.round(self.pcd.rgb_fy * (points_rgb[:, 1] / z_rgb) + self.pcd.rgb_cy).to(torch.long)
        in_img = (ui >= 0) & (ui < w_r) & (vi >= 0) & (vi < h_r)
        if not torch.any(in_img):
            self.pcd.fuse_semantic_observations(
                torch.empty((0,), dtype=torch.long, device=self.device),
                torch.empty((0,), dtype=torch.long, device=self.device),
                torch.empty((0,), dtype=torch.float32, device=self.device),
                visible_gaussian_indices=visible_idx,
                class_names=self._names,
                class_palette=self._palette_t,
                pred_instances=[],
                metadata={"inference_ms": float(infer_ms), "num_detections": int(len(segments_info))},
            )
            return

        vis_idx = vis_idx[in_img]
        ui = ui[in_img]
        vi = vi[in_img]
        seg_ids_at_gauss = segmentation_t[vi, ui]
        geom_ms = (time.perf_counter() - t_geom1) * 1000.0

        seg_info_by_id = {int(s.get("id", -1)): s for s in segments_info}
        ranked_segments = []
        obs_idx = []
        obs_cls = []
        obs_conf = []
        pred_instances = []

        t_match0 = time.perf_counter()
        for seg_id_t in torch.unique(seg_ids_at_gauss).tolist():
            seg_id = int(seg_id_t)
            if seg_id < 0:
                continue

            seg = seg_info_by_id.get(seg_id, None)
            if seg is None:
                continue

            seg_score = float(seg.get("score", 0.0))
            if seg_score < float(self.assignment_confidence_gate):
                n_low_conf_discarded += 1
                continue

            raw_label = int(seg.get("label_id", -1))
            local_class_id = self._raw_label_to_local.get(raw_label, -1)
            if local_class_id < 0:
                continue

            sel = seg_ids_at_gauss == seg_id
            if not torch.any(sel):
                continue

            g_idx = torch.unique(vis_idx[sel])
            if g_idx.numel() == 0:
                continue

            ranked_segments.append(seg)
            obs_idx.append(g_idx)
            obs_cls.append(torch.full_like(g_idx, int(local_class_id), dtype=torch.long, device=self.device))
            obs_conf.append(torch.full_like(g_idx, float(seg_score), dtype=torch.float32, device=self.device))
            pred_instances.append(
                {
                    "class_id": int(local_class_id),
                    "confidence": float(seg_score),
                    "gaussian_indices": g_idx,
                }
            )

        ranked_segments.sort(key=lambda s: float(s.get("score", 0.0)), reverse=True)
        match_ms = (time.perf_counter() - t_match0) * 1000.0

        if len(obs_idx) == 0:
            idx_t = torch.empty((0,), dtype=torch.long, device=self.device)
            cls_t = torch.empty((0,), dtype=torch.long, device=self.device)
            conf_t = torch.empty((0,), dtype=torch.float32, device=self.device)
        else:
            idx_t = torch.cat(obs_idx, dim=0)
            cls_t = torch.cat(obs_cls, dim=0)
            conf_t = torch.cat(obs_conf, dim=0)

        # Final safety gate before fusion.
        if conf_t.numel() > 0:
            keep_final = conf_t >= float(self.assignment_confidence_gate)
            idx_t = idx_t[keep_final]
            cls_t = cls_t[keep_final]
            conf_t = conf_t[keep_final]

        total_ms = (time.perf_counter() - t0) * 1000.0
        self.pcd.fuse_semantic_observations(
            idx_t,
            cls_t,
            conf_t,
            visible_gaussian_indices=visible_idx,
            class_names=self._names,
            class_palette=self._palette_t,
            pred_instances=pred_instances,
            metadata={
                "timestamp": time.time(),
                "inference_ms": float(infer_ms),
                "total_ms": float(total_ms),
                "postprocess_ms": float(post_ms),
                "debug_ms": float(debug_ms),
                "geometry_ms": float(geom_ms),
                "matching_ms": float(match_ms),
                "num_detections": int(len(segments_info)),
                "num_visible_segments": int(len(ranked_segments)),
                "num_gaussians_observed": int(idx_t.numel()),
                "assignment_confidence_gate": float(self.assignment_confidence_gate),
                "low_conf_detections_discarded": int(n_low_conf_discarded),
                "model_id": self.model_id,
            },
        )
