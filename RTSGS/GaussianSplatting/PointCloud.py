import threading

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as Rot
import concurrent.futures

class PointCloud:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Camera parameters (depth camera for geometry, rgb camera for image supervision).
        K_depth = config.get_depth_intrinsics()
        K_rgb = config.get_rgb_intrinsics()

        self.fx = torch.tensor(float(K_depth[0, 0]), device=self.device)
        self.fy = torch.tensor(float(K_depth[1, 1]), device=self.device)
        self.cx = torch.tensor(float(K_depth[0, 2]), device=self.device)
        self.cy = torch.tensor(float(K_depth[1, 2]), device=self.device)

        self.rgb_fx = torch.tensor(float(K_rgb[0, 0]), device=self.device)
        self.rgb_fy = torch.tensor(float(K_rgb[1, 1]), device=self.device)
        self.rgb_cx = torch.tensor(float(K_rgb[0, 2]), device=self.device)
        self.rgb_cy = torch.tensor(float(K_rgb[1, 2]), device=self.device)

        T_depth_to_rgb = config.get_T_depth_to_rgb()
        self.R_depth_to_rgb = torch.from_numpy(T_depth_to_rgb[:3, :3]).to(self.device).float()
        self.t_depth_to_rgb = torch.from_numpy(T_depth_to_rgb[:3, 3]).to(self.device).float()

        self.depth_scale = float(config.get("depth_scale", 1.0))
        self.voxel_size = float(config.get("voxel_size", 0.02))
        self.novelty_voxel = float(config.get("novelty_voxel", self.voxel_size))
        self.projection_depth_diff_threshold_m = float(config.get("projection_depth_diff_threshold_m", 0.10))

        # SH Parameters
        self.sh_degree = int(config.get("sh_degree", 1))
        self.num_sh_bases = (self.sh_degree + 1) ** 2
        
        # Gaussian Properties
        self.sigma_px = float(config.get("sigma_px", 4.0))
        self.sigma_z0 = float(config.get("sigma_z0", 0.005))
        self.sigma_z1 = float(config.get("sigma_z1", 0.0))
        self.alpha_init = float(config.get("alpha_init", 1.0))
        self.alpha_min = float(config.get("alpha_min", 0.01))
        self.alpha_max = float(config.get("alpha_max", 1.0))
        self.alpha_depth_scale = float(config.get("alpha_depth_scale", 0.0))

        self.all_points = None
        self.all_sh = None
        self.all_scales = None
        self.all_quaternions = None
        self.all_alpha = None

        # Segmentation outputs (aligned 1:1 with all_points).
        self.segmentation_labels = None
        self.segmentation_colors = None
        self.segmentation_color_logits = None
        self.semantic_confidence = None
        self.segmentation_instances = []
        self.segmentation_metadata = {}
        self.segmentation_version = 0
        self.semantic_decay_factor = float(config.get("semantic_decay_factor", 0.98))
        self.semantic_min_confidence = float(config.get("semantic_min_confidence", 0.2))
        self.semantic_ema_alpha = float(config.get("semantic_ema_alpha", 1.0))

        # Voxel Packing
        self._pack_offset = int(config.get("pack_offset", 1_000_000))
        self._pack_base = 2 * self._pack_offset + 1
        self.seen_keys = torch.empty((0,), dtype=torch.int64, device=self.device)
        self.pixel_subsample = float(config.get("pixel_subsample", 1.0))

        # Rotation Fix
        ax, ay = np.radians(-90), np.radians(180)
        Rx = torch.tensor([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
        Ry = torch.tensor([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
        self.R_fix = (Ry @ Rx).to(self.device).float()

        # --- Async Handling ---
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.is_processing = False
        self.lock = threading.Lock()
        
    def _pack_voxels(self, vox_xyz: torch.Tensor) -> torch.Tensor:
        off, base = self._pack_offset, self._pack_base
        return (vox_xyz[:, 0] + off) * (base**2) + (vox_xyz[:, 1] + off) * base + (vox_xyz[:, 2] + off)

    @torch.no_grad()
    def _make_gaussians_batch(self, points_cam, R_world_from_cam, z):
        N = points_cam.shape[0]
        sigma_z = self.sigma_z0 + self.sigma_z1 * (z * z)
        sigma_x, sigma_y = (self.sigma_px / self.fx) * z, (self.sigma_px / self.fy) * z
        scales = torch.log(torch.clamp(torch.stack([sigma_x, sigma_y, sigma_z], dim=-1), min=0.01))

        # Ensure R_world_from_cam is a batch of matrices even for a single pose
        if R_world_from_cam.dim() == 2:
            R_world_from_cam = R_world_from_cam.unsqueeze(0).expand(N, -1, -1)

        quats = torch.from_numpy(Rot.from_matrix(R_world_from_cam.cpu().numpy()).as_quat()).to(self.device).float()

        a = self.alpha_init * torch.exp(-z / self.alpha_depth_scale) if self.alpha_depth_scale > 0.0 else torch.full((N,), float(self.alpha_init), device=self.device)
        a = torch.logit(torch.clamp(a, self.alpha_min, self.alpha_max)).unsqueeze(1)
        return scales, quats, a

    @torch.no_grad()
    def novelty_filter_fast_with_gaussians(self, points, colors, voxel):
        if points.numel() == 0: return None
        vox = torch.floor(points / voxel).to(torch.int64)
        keys = self._pack_voxels(vox)
        keys_unique, inv_indices = torch.unique(keys, return_inverse=True)
        
        first_idx = torch.full(keys_unique.shape, points.shape[0], dtype=torch.long, device=self.device)
        first_idx.scatter_reduce_(0, inv_indices, torch.arange(points.shape[0], device=self.device), "amin", include_self=False)

        is_new = torch.ones_like(keys_unique, dtype=torch.bool) if self.seen_keys.numel() == 0 else ~torch.isin(keys_unique, self.seen_keys)
        if not is_new.any(): return None

        self.seen_keys = torch.cat([self.seen_keys, keys_unique[is_new]])
        return points[first_idx[is_new]], colors[first_idx[is_new]], first_idx[is_new]

    @torch.no_grad()
    def voxel_filter_with_gaussians(self, points, sh, scales, quats, alpha, voxel):
        if points.numel() == 0: return points, sh, scales, quats, alpha
        vox = torch.floor(points / voxel).to(torch.int64)
        _, inverse = torch.unique(vox, dim=0, return_inverse=True)
        num = int(inverse.max().item()) + 1
        counts = torch.zeros((num, 1), device=self.device, dtype=points.dtype)
        counts.scatter_add_(0, inverse[:, None], torch.ones((points.shape[0], 1), device=self.device))

        def reduce_mean(tensor, dim_size):
            shape = (num,) + tensor.shape[1:]
            out = torch.zeros(shape, device=self.device, dtype=tensor.dtype)
            idx = inverse.view(-1, *([1] * (len(tensor.shape) - 1))).expand_as(tensor)
            out.scatter_add_(0, idx, tensor)
            div = counts.view(-1, *([1] * (len(tensor.shape) - 1)))
            return out / div

        return reduce_mean(points, 3), reduce_mean(sh, 3), reduce_mean(scales, 3), \
               F.normalize(reduce_mean(quats, 4), p=2, dim=1), reduce_mean(alpha, 1)

    @torch.no_grad()
    def process_single_keyframe(self, rgb_np, depth_np, pose_np, rendered_depth_np=None):
        """Processes a single keyframe: Image + Depth + Pose -> Filtered Gaussians."""
        rgb = torch.from_numpy(rgb_np).to(self.device).float() / 255.0
        depth = torch.from_numpy(depth_np).to(self.device).float() / self.depth_scale
        pose = torch.from_numpy(pose_np).to(self.device).float()

        H_d, W_d = depth.shape
        H_r, W_r = rgb.shape[:2]
        z_raw = depth.reshape(-1)
        mask = z_raw > 0

        # Optionally skip points that are too close to the last rendered depth.
        if rendered_depth_np is not None:
            rendered_depth = torch.from_numpy(np.asarray(rendered_depth_np)).to(self.device).float()
            if rendered_depth.shape == depth.shape:
                rendered_raw = rendered_depth.reshape(-1)
                rendered_valid = torch.isfinite(rendered_raw) & (rendered_raw > 0)
                close_to_render = torch.abs(z_raw - rendered_raw) < self.projection_depth_diff_threshold_m
                mask &= ~(rendered_valid & close_to_render)

        if self.pixel_subsample < 1.0:
            mask &= (torch.rand(z_raw.shape, device=self.device) < self.pixel_subsample)
        
        indices = torch.where(mask)[0]
        if indices.numel() == 0: return None

        z_f = z_raw[indices]
        u_d = (indices % W_d).float()
        v_d = (indices // W_d).float()
        points_cam = torch.stack([
            (u_d - self.cx) * z_f / self.fx,
            (v_d - self.cy) * z_f / self.fy,
            z_f,
        ], dim=1)

        # Reproject depth-camera points into RGB image for robust color lookup.
        points_rgb_cam = (self.R_depth_to_rgb @ points_cam.T).T + self.t_depth_to_rgb
        z_rgb = points_rgb_cam[:, 2]
        valid_rgb = z_rgb > 1e-6
        if not torch.any(valid_rgb):
            return None

        u_rgb = self.rgb_fx * (points_rgb_cam[:, 0] / z_rgb) + self.rgb_cx
        v_rgb = self.rgb_fy * (points_rgb_cam[:, 1] / z_rgb) + self.rgb_cy

        ui = torch.round(u_rgb).long()
        vi = torch.round(v_rgb).long()
        valid_rgb &= (ui >= 0) & (ui < W_r) & (vi >= 0) & (vi < H_r)
        if not torch.any(valid_rgb):
            return None

        points_cam = points_cam[valid_rgb]
        z_f = z_f[valid_rgb]
        ui = ui[valid_rgb]
        vi = vi[valid_rgb]

        R_corr = self.R_fix @ pose[:3, :3]
        t_corr = (self.R_fix @ pose[:3, 3].unsqueeze(-1)).squeeze(-1)

        points_world = (R_corr @ points_cam.T).T + t_corr
        colors = rgb[vi, ui]

        res = self.novelty_filter_fast_with_gaussians(points_world, colors, self.novelty_voxel)
        if res is None: return None
        
        pts, cols, k_idx = res
        
        # SH conversion
        sh_full = torch.zeros((pts.shape[0], self.num_sh_bases, 3), device=self.device)
        sh_full[:, 0, :] = torch.logit(torch.clamp(cols, 0.001, 0.999)) / 0.28209479177387814

        # Attribute generation
        scales, quats, alpha = self._make_gaussians_batch(points_cam[k_idx], R_corr, z_f[k_idx])
        return self.voxel_filter_with_gaussians(pts, sh_full, scales, quats, alpha, self.voxel_size)

    def update_async(self, rgb_np, depth_np, pose_np, rendered_depth_np=None):
        """Non-blocking call to process and add a keyframe to the map."""
        if self.is_processing:
            return False # Busy processing previous frame
        
        self.is_processing = True

        def _task():
            try:
                new_data = self.process_single_keyframe(rgb_np, depth_np, pose_np, rendered_depth_np)
                if new_data is not None:
                    self._merge_data(new_data)
            finally:
                self.is_processing = False

        self.executor.submit(_task)
        return True

    def _merge_data(self, new_data):
        """Thread-safe concatenation of new Gaussians."""
        with self.lock: # Protect the update
            if self.all_points is None:
                self.all_points, self.all_sh, self.all_scales, self.all_quaternions, self.all_alpha = new_data
            else:
                self.all_points = torch.cat([self.all_points, new_data[0]])
                self.all_sh = torch.cat([self.all_sh, new_data[1]])
                self.all_scales = torch.cat([self.all_scales, new_data[2]])
                self.all_quaternions = torch.cat([self.all_quaternions, new_data[3]])
                self.all_alpha = torch.cat([self.all_alpha, new_data[4]])

            # Keep segmentation buffers aligned after map growth.
            if self.segmentation_labels is not None:
                num_new = int(new_data[0].shape[0])
                if num_new > 0:
                    self.segmentation_labels = torch.cat(
                        [
                            self.segmentation_labels,
                            torch.full((num_new,), -1, dtype=torch.long, device=self.device),
                        ],
                        dim=0,
                    )
            if self.segmentation_colors is not None:
                num_new = int(new_data[0].shape[0])
                if num_new > 0:
                    self.segmentation_colors = torch.cat(
                        [
                            self.segmentation_colors,
                            torch.zeros((num_new, 3), dtype=torch.float32, device=self.device),
                        ],
                        dim=0,
                    )
            if self.segmentation_color_logits is not None:
                num_new = int(new_data[0].shape[0])
                if num_new > 0:
                    self.segmentation_color_logits = torch.cat(
                        [
                            self.segmentation_color_logits,
                            torch.zeros((num_new, 3), dtype=torch.float32, device=self.device),
                        ],
                        dim=0,
                    )
            if self.semantic_confidence is not None:
                num_new = int(new_data[0].shape[0])
                if num_new > 0:
                    self.semantic_confidence = torch.cat(
                        [
                            self.semantic_confidence,
                            torch.zeros((num_new,), dtype=torch.float32, device=self.device),
                        ],
                        dim=0,
                    )

    def get_map(self):
        return self.all_points, self.all_sh, self.all_scales, self.all_quaternions, self.all_alpha

    def set_segmentation_result(
        self,
        labels_np,
        colors_np,
        color_logits_np,
        pred_instances=None,
        metadata=None,
    ):
        with self.lock:
            labels_t = torch.from_numpy(np.asarray(labels_np, dtype=np.int64)).to(self.device)
            colors_t = torch.from_numpy(np.asarray(colors_np, dtype=np.float32)).to(self.device)
            logits_t = torch.from_numpy(np.asarray(color_logits_np, dtype=np.float32)).to(self.device)

            n_map = 0 if self.all_points is None else int(self.all_points.shape[0])
            if n_map > 0 and labels_t.shape[0] != n_map:
                return False

            self.segmentation_labels = labels_t
            self.segmentation_colors = colors_t
            self.segmentation_color_logits = logits_t
            self.semantic_confidence = torch.where(labels_t >= 0, torch.ones_like(labels_t, dtype=torch.float32), torch.zeros_like(labels_t, dtype=torch.float32))
            self.segmentation_instances = list(pred_instances) if pred_instances is not None else []
            self.segmentation_metadata = dict(metadata) if metadata is not None else {}
            self.segmentation_version += 1
            return True

    @staticmethod
    def _encode_colors_for_shader_torch(rgb_01: torch.Tensor) -> torch.Tensor:
        c = torch.clamp(rgb_01, 1e-3, 1.0 - 1e-3)
        logits = torch.log(c / (1.0 - c))
        encoded = torch.empty_like(logits)
        encoded[:, 0] = logits[:, 2]
        encoded[:, 1] = logits[:, 1]
        encoded[:, 2] = logits[:, 0]
        return encoded

    def _ensure_semantic_buffers(self, n_map: int):
        if self.segmentation_labels is None or int(self.segmentation_labels.shape[0]) != n_map:
            self.segmentation_labels = torch.full((n_map,), -1, dtype=torch.long, device=self.device)
        if self.semantic_confidence is None or int(self.semantic_confidence.shape[0]) != n_map:
            self.semantic_confidence = torch.zeros((n_map,), dtype=torch.float32, device=self.device)
        if self.segmentation_colors is None or int(self.segmentation_colors.shape[0]) != n_map:
            self.segmentation_colors = torch.zeros((n_map, 3), dtype=torch.float32, device=self.device)
        if self.segmentation_color_logits is None or int(self.segmentation_color_logits.shape[0]) != n_map:
            self.segmentation_color_logits = torch.zeros((n_map, 3), dtype=torch.float32, device=self.device)

    def fuse_semantic_observations(
        self,
        gaussian_indices: torch.Tensor,
        class_ids: torch.Tensor,
        confidences: torch.Tensor,
        visible_gaussian_indices: torch.Tensor | None = None,
        class_names=None,
        class_palette=None,
        metadata=None,
        pred_instances=None,
    ):
        with self.lock:
            n_map = 0 if self.all_points is None else int(self.all_points.shape[0])
            if n_map <= 0:
                return False

            self._ensure_semantic_buffers(n_map)

            idx = gaussian_indices.to(self.device, dtype=torch.long).reshape(-1)
            cls = class_ids.to(self.device, dtype=torch.long).reshape(-1)
            conf = confidences.to(self.device, dtype=torch.float32).reshape(-1)
            conf = torch.clamp(conf, 0.0, 1.0)

            valid = (idx >= 0) & (idx < n_map)
            idx = idx[valid]
            cls = cls[valid]
            conf = conf[valid]

            detected_mask = torch.zeros((n_map,), dtype=torch.bool, device=self.device)
            visible_mask = torch.zeros((n_map,), dtype=torch.bool, device=self.device)

            if visible_gaussian_indices is not None:
                vis_idx = visible_gaussian_indices.to(self.device, dtype=torch.long).reshape(-1)
                vis_idx = vis_idx[(vis_idx >= 0) & (vis_idx < n_map)]
                if vis_idx.numel() > 0:
                    visible_mask[vis_idx] = True
            else:
                visible_mask[:] = True

            if idx.numel() > 0:
                order = torch.argsort(conf, descending=True)
                idx = idx[order]
                cls = cls[order]
                conf = conf[order]

                keep = torch.ones_like(idx, dtype=torch.bool)
                if idx.numel() > 1:
                    keep[1:] = idx[1:] != idx[:-1]

                idx = idx[keep]
                cls = cls[keep]
                conf = conf[keep]

                old_cls = self.segmentation_labels[idx]
                old_conf = self.semantic_confidence[idx]

                ema_alpha = float(np.clip(self.semantic_ema_alpha, 0.0, 1.0))
                obs_conf = ema_alpha * conf + (1.0 - ema_alpha) * old_conf

                same_cls = old_cls == cls
                reinforced = old_conf + obs_conf * (1.0 - old_conf)

                penalized = old_conf * (1.0 - obs_conf)
                switched = obs_conf > penalized

                new_cls = torch.where(same_cls | switched, cls, old_cls)
                new_conf = torch.where(same_cls, reinforced, torch.where(switched, obs_conf, penalized))

                self.segmentation_labels[idx] = new_cls
                self.semantic_confidence[idx] = torch.clamp(new_conf, 0.0, 1.0)
                detected_mask[idx] = True

            if n_map > 0:
                decay_factor = float(np.clip(self.semantic_decay_factor, 0.0, 1.0))
                self.semantic_confidence[visible_mask & (~detected_mask)] *= decay_factor

            min_conf = float(max(0.0, self.semantic_min_confidence))
            low = self.semantic_confidence < min_conf
            self.segmentation_labels[low] = -1
            self.semantic_confidence[low] = 0.0

            if class_palette is None:
                n_classes = int(max(1, (torch.max(self.segmentation_labels).item() + 1) if torch.any(self.segmentation_labels >= 0) else 1))
                g = torch.Generator(device=self.device)
                g.manual_seed(123)
                palette_t = torch.rand((n_classes, 3), generator=g, device=self.device, dtype=torch.float32) * 0.85 + 0.1
            else:
                if isinstance(class_palette, torch.Tensor):
                    palette_t = class_palette.to(self.device, dtype=torch.float32)
                else:
                    palette_t = torch.from_numpy(np.asarray(class_palette, dtype=np.float32)).to(self.device)
                if palette_t.ndim != 2 or palette_t.shape[1] != 3:
                    palette_t = torch.zeros((1, 3), dtype=torch.float32, device=self.device)

            colors = torch.zeros((n_map, 3), dtype=torch.float32, device=self.device)
            valid_cls = self.segmentation_labels >= 0
            if torch.any(valid_cls):
                safe_cls = torch.clamp(self.segmentation_labels[valid_cls], 0, int(palette_t.shape[0]) - 1)
                base = palette_t[safe_cls]
                conf_w = self.semantic_confidence[valid_cls].unsqueeze(1)
                colors[valid_cls] = base * conf_w

            logits = self._encode_colors_for_shader_torch(colors)

            self.segmentation_colors = colors
            self.segmentation_color_logits = logits
            self.segmentation_instances = list(pred_instances) if pred_instances is not None else []

            md = dict(self.segmentation_metadata) if isinstance(self.segmentation_metadata, dict) else {}
            if metadata is not None:
                md.update(dict(metadata))

            if class_names is not None:
                md["class_names"] = [str(x) for x in class_names]
            if palette_t is not None:
                md["class_palette"] = palette_t.detach().cpu().numpy().tolist()

            md["num_points_total"] = int(n_map)
            md["num_points_segmented"] = int(torch.count_nonzero(valid_cls).item())
            md["num_classes_present"] = int(torch.unique(self.segmentation_labels[valid_cls]).numel()) if torch.any(valid_cls) else 0

            self.segmentation_metadata = md
            self.segmentation_version += 1
            return True