import threading
import time

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

        gs = config.gaussian_splatting
        self.depth_scale = float(config.camera.depth_scale)
        self.voxel_size = float(gs.voxel_size)
        self.novelty_voxel = float(gs.novelty_voxel)
        self.projection_depth_diff_threshold_m = float(gs.projection_depth_diff_threshold_m)

        self.sh_degree = int(gs.sh_degree)
        self.num_sh_bases = (self.sh_degree + 1) ** 2

        self.sigma_px = float(gs.sigma_px)
        self.sigma_z0 = float(gs.sigma_z0)
        self.sigma_z1 = float(gs.sigma_z1)
        self.alpha_init = float(gs.alpha_init)
        self.alpha_min = float(gs.alpha_min)
        self.alpha_max = float(gs.alpha_max)
        self.alpha_depth_scale = float(gs.alpha_depth_scale)

        self.depth_min = float(gs.depth_min)
        self.depth_max = float(gs.depth_max)
        self.depth_edge_threshold_m = float(gs.depth_edge_threshold_m)

        self.all_points = None
        self.all_sh = None
        self.all_scales = None
        self.all_quaternions = None
        self.all_alpha = None

        self.gaussian_birth_iter = None
        self._birth_counter = 0

        # Segmentation outputs (aligned 1:1 with all_points).
        self.segmentation_labels = None
        self.segmentation_colors = None
        self.segmentation_color_logits = None
        self.semantic_confidence = None
        self.semantic_alt_class = None
        self.semantic_alt_confidence = None
        self.semantic_switch_support = None
        self.semantic_switch_cooldown = None
        self.gaussian_instance_ids = None
        self.segmentation_instances = []
        self.segmentation_metadata = {}
        self.segmentation_version = 0
        self.segmentation_debug_image_bgr = None
        self.segmentation_debug_timestamp = 0.0
        seg = config.segmentation
        self.semantic_decay_factor = float(seg.decay_factor)
        self.semantic_challenger_decay_factor = float(seg.challenger_decay_factor)
        self.semantic_min_confidence = float(seg.min_semantic_confidence)
        self.semantic_ema_alpha = float(seg.ema_alpha)
        self.semantic_switch_margin = float(seg.switch_margin)
        self.semantic_switch_support_frames = int(seg.switch_support_frames)
        self.semantic_switch_cooldown_frames = int(seg.switch_cooldown_frames)
        self.semantic_assign_min_confidence = float(seg.assignment_min_confidence)

        inst = config.instance
        self.instance_min_points = int(inst.min_points)
        self.instance_iou_gate = float(inst.iou_gate)
        self.instance_center_gate_m = float(inst.center_gate_m)
        self.instance_overlap_gate = float(inst.overlap_gate)
        self.instance_partial_overlap_gate = float(inst.partial_overlap_gate)
        self.instance_bbox_expand_ratio = float(inst.bbox_expand_ratio)
        self.instance_merge_min_ratio = float(inst.merge_min_ratio)
        self.instance_max_missed_frames = int(inst.max_missed_frames)
        self.instance_bbox_quantile = float(inst.bbox_quantile)
        self.instance_bbox_merge_iou = float(inst.bbox_merge_iou)
        self.instance_bbox_merge_center_gate_m = float(inst.bbox_merge_center_gate_m)
        self._instance_next_id = 1
        self._instance_frame_idx = 0
        self._instance_track_meta: dict[int, dict] = {}

        # Real-time scene graph outputs.
        self.scene_graph_state = {}
        self.scene_graph_version = 0
        self.scene_graph_last_error = ""

        # Voxel Packing
        self._pack_offset = int(gs.pack_offset)
        self._pack_base = 2 * self._pack_offset + 1
        self.seen_keys = torch.empty((0,), dtype=torch.int64, device=self.device)
        self.pixel_subsample = float(gs.pixel_subsample)

        # Rotation Fix
        ax, ay = np.radians(-90), np.radians(180)
        Rx = torch.tensor([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
        Ry = torch.tensor([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
        self.R_fix = (Ry @ Rx).to(self.device).float()

        # --- Async Handling ---
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.is_processing = False
        self.lock = threading.Lock()
        self.rendered_depth_provider = None

        # Optional benchmark collector (set by System when benchmark=True)
        self.benchmark = None

    def set_rendered_depth_provider(self, provider):
        """Set callable provider: provider(pose_4x4_np) -> depth_m (H,W) or None."""
        self.rendered_depth_provider = provider
        
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
    def rebuild_seen_keys(self):
        """Recompute voxel occupancy from surviving Gaussians so pruned regions can be re-populated."""
        if self.all_points is None or self.all_points.shape[0] == 0:
            self.seen_keys = torch.empty((0,), dtype=torch.int64, device=self.device)
            return
        vox = torch.floor(self.all_points / self.novelty_voxel).to(torch.int64)
        keys = self._pack_voxels(vox)
        self.seen_keys = torch.unique(keys)

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

        valid_depth = (depth > self.depth_min) & (depth < self.depth_max) & torch.isfinite(depth)

        if self.depth_edge_threshold_m > 0.0 and H_d >= 3 and W_d >= 3:
            padded = F.pad(depth.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate')[0, 0]
            max_neighbor_diff = torch.stack([
                torch.abs(depth - padded[:-2, 1:-1]),
                torch.abs(depth - padded[2:, 1:-1]),
                torch.abs(depth - padded[1:-1, :-2]),
                torch.abs(depth - padded[1:-1, 2:]),
            ]).max(dim=0).values
            valid_depth &= max_neighbor_diff < self.depth_edge_threshold_m

        depth = torch.where(valid_depth, depth, torch.zeros_like(depth))
        z_raw = depth.reshape(-1)
        mask = z_raw > 0

        # Prefer a fresh rendered depth from the current keyframe pose.
        rendered_depth_src = None
        if self.rendered_depth_provider is not None:
            try:
                rendered_depth_src = self.rendered_depth_provider(np.asarray(pose_np, dtype=np.float32))
            except Exception:
                rendered_depth_src = None

        # Backward-compatible fallback if no provider is set.
        if rendered_depth_src is None:
            rendered_depth_src = rendered_depth_np

        # Optionally skip points that are too close to the rendered depth.
        if rendered_depth_src is not None:
            rendered_depth = torch.from_numpy(np.asarray(rendered_depth_src)).to(self.device).float()
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
                bench = getattr(self, "benchmark", None)
                do_bench = bool(getattr(bench, "enabled", False))
                alloc0 = None
                if do_bench:
                    try:
                        import torch

                        if torch.cuda.is_available():
                            # Preserve run-wide peak before doing any per-stage reset.
                            try:
                                bench.observe_total_peak()
                            except Exception:
                                pass
                            torch.cuda.synchronize()
                            # Reset peak stats so mapping peak is mapping-specific.
                            torch.cuda.reset_peak_memory_stats()
                            torch.cuda.synchronize()
                            alloc0 = int(torch.cuda.memory_allocated())
                    except Exception:
                        alloc0 = None

                t0 = time.perf_counter() if do_bench else None
                new_data = self.process_single_keyframe(rgb_np, depth_np, pose_np, rendered_depth_np)
                if new_data is not None:
                    self._merge_data(new_data)

                if do_bench and t0 is not None:
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    try:
                        from RTSGS.Benchmarking.BenchmarkCollector import BenchmarkCollector

                        bench.record_ms(BenchmarkCollector.STAGE_MAPPING, float(dt_ms))
                    except Exception:
                        pass

                    # Track mapping activation peak as a peak-alloc delta over the baseline.
                    try:
                        import torch

                        if torch.cuda.is_available() and alloc0 is not None:
                            torch.cuda.synchronize()
                            peak_map = int(torch.cuda.max_memory_allocated())
                            try:
                                bench.observe_total_peak_value(peak_map)
                            except Exception:
                                pass
                            bench.update_mapping_activation_peak(int(max(0, peak_map - int(alloc0))))
                    except Exception:
                        pass
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

            num_new = int(new_data[0].shape[0])
            self._birth_counter += 1
            new_birth = torch.full(
                (num_new,), self._birth_counter, dtype=torch.int32, device=self.device
            )
            if self.gaussian_birth_iter is None:
                self.gaussian_birth_iter = new_birth
            else:
                self.gaussian_birth_iter = torch.cat([self.gaussian_birth_iter, new_birth])

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
            if self.semantic_alt_class is not None:
                num_new = int(new_data[0].shape[0])
                if num_new > 0:
                    self.semantic_alt_class = torch.cat(
                        [
                            self.semantic_alt_class,
                            torch.full((num_new,), -1, dtype=torch.long, device=self.device),
                        ],
                        dim=0,
                    )
            if self.semantic_alt_confidence is not None:
                num_new = int(new_data[0].shape[0])
                if num_new > 0:
                    self.semantic_alt_confidence = torch.cat(
                        [
                            self.semantic_alt_confidence,
                            torch.zeros((num_new,), dtype=torch.float32, device=self.device),
                        ],
                        dim=0,
                    )
            if self.semantic_switch_support is not None:
                num_new = int(new_data[0].shape[0])
                if num_new > 0:
                    self.semantic_switch_support = torch.cat(
                        [
                            self.semantic_switch_support,
                            torch.zeros((num_new,), dtype=torch.uint8, device=self.device),
                        ],
                        dim=0,
                    )
            if self.semantic_switch_cooldown is not None:
                num_new = int(new_data[0].shape[0])
                if num_new > 0:
                    self.semantic_switch_cooldown = torch.cat(
                        [
                            self.semantic_switch_cooldown,
                            torch.zeros((num_new,), dtype=torch.uint8, device=self.device),
                        ],
                        dim=0,
                    )
            if self.gaussian_instance_ids is not None:
                num_new = int(new_data[0].shape[0])
                if num_new > 0:
                    self.gaussian_instance_ids = torch.cat(
                        [
                            self.gaussian_instance_ids,
                            torch.full((num_new,), -1, dtype=torch.long, device=self.device),
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
            self.semantic_alt_class = torch.full_like(labels_t, -1, dtype=torch.long)
            self.semantic_alt_confidence = torch.zeros_like(labels_t, dtype=torch.float32)
            self.semantic_switch_support = torch.zeros_like(labels_t, dtype=torch.uint8)
            self.semantic_switch_cooldown = torch.zeros_like(labels_t, dtype=torch.uint8)
            if pred_instances is not None:
                self.segmentation_instances = list(pred_instances)
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
        def _resize_or_init_1d(x, fill_value, dtype):
            if x is None:
                return torch.full((n_map,), fill_value, dtype=dtype, device=self.device)
            cur = int(x.shape[0])
            if cur == n_map:
                return x
            if cur < n_map:
                tail = torch.full((n_map - cur,), fill_value, dtype=dtype, device=self.device)
                return torch.cat([x, tail], dim=0)
            return x[:n_map]

        def _resize_or_init_2d(x, fill_value, dtype):
            if x is None:
                return torch.full((n_map, 3), fill_value, dtype=dtype, device=self.device)
            cur = int(x.shape[0])
            if cur == n_map:
                return x
            if cur < n_map:
                tail = torch.full((n_map - cur, 3), fill_value, dtype=dtype, device=self.device)
                return torch.cat([x, tail], dim=0)
            return x[:n_map]

        self.segmentation_labels = _resize_or_init_1d(self.segmentation_labels, -1, torch.long)
        self.semantic_confidence = _resize_or_init_1d(self.semantic_confidence, 0.0, torch.float32)
        self.semantic_alt_class = _resize_or_init_1d(self.semantic_alt_class, -1, torch.long)
        self.semantic_alt_confidence = _resize_or_init_1d(self.semantic_alt_confidence, 0.0, torch.float32)
        self.semantic_switch_support = _resize_or_init_1d(self.semantic_switch_support, 0, torch.uint8)
        self.semantic_switch_cooldown = _resize_or_init_1d(self.semantic_switch_cooldown, 0, torch.uint8)
        self.gaussian_instance_ids = _resize_or_init_1d(self.gaussian_instance_ids, -1, torch.long)
        self.segmentation_colors = _resize_or_init_2d(self.segmentation_colors, 0.0, torch.float32)
        self.segmentation_color_logits = _resize_or_init_2d(self.segmentation_color_logits, 0.0, torch.float32)

    def _instance_bbox(self, indices_t: torch.Tensor):
        if indices_t.numel() == 0 or self.all_points is None:
            return None
        pts = self.all_points[indices_t]
        if pts.numel() == 0:
            return None
        q = float(np.clip(self.instance_bbox_quantile, 0.0, 0.2))
        if q > 0.0 and int(pts.shape[0]) >= 8:
            lo = torch.tensor(q, device=self.device, dtype=pts.dtype)
            hi = torch.tensor(1.0 - q, device=self.device, dtype=pts.dtype)
            bmin = torch.quantile(pts, lo, dim=0)
            bmax = torch.quantile(pts, hi, dim=0)
        else:
            bmin = torch.amin(pts, dim=0)
            bmax = torch.amax(pts, dim=0)
        center = 0.5 * (bmin + bmax)
        return bmin, bmax, center

    def _instance_obb(self, indices_t: torch.Tensor):
        if indices_t.numel() == 0 or self.all_points is None:
            return None
        pts = self.all_points[indices_t]
        if pts.numel() == 0:
            return None

        n = int(pts.shape[0])
        centroid = torch.mean(pts, dim=0)
        centered = pts - centroid

        # PCA frame for oriented extents.
        if n >= 3:
            cov = (centered.T @ centered) / float(max(n - 1, 1))
            evals, evecs = torch.linalg.eigh(cov)
            order = torch.argsort(evals, descending=True)
            R = evecs[:, order]
            if torch.det(R) < 0:
                R[:, 2] = -R[:, 2]
        else:
            R = torch.eye(3, device=self.device, dtype=torch.float32)

        local = centered @ R
        q = float(np.clip(self.instance_bbox_quantile, 0.0, 0.2))
        if q > 0.0 and n >= 8:
            lo = torch.tensor(q, device=self.device, dtype=pts.dtype)
            hi = torch.tensor(1.0 - q, device=self.device, dtype=pts.dtype)
            local_min = torch.quantile(local, lo, dim=0)
            local_max = torch.quantile(local, hi, dim=0)
        else:
            local_min = torch.amin(local, dim=0)
            local_max = torch.amax(local, dim=0)

        axes_lengths = (local_max - local_min).clamp_min(1e-6)
        local_center = 0.5 * (local_min + local_max)
        obb_centroid = centroid + (R @ local_center)

        # Use the shortest OBB axis as dominant normal proxy.
        n_idx = int(torch.argmin(axes_lengths).item())
        dominant_normal = R[:, n_idx]

        return {
            "centroid": obb_centroid,
            "axesLengths": axes_lengths,
            "normalizedAxes": R.reshape(-1),
            "dominantNormal": dominant_normal,
        }

    @staticmethod
    def _bbox_iou_3d(a_min, a_max, b_min, b_max) -> float:
        inter_min = torch.maximum(a_min, b_min)
        inter_max = torch.minimum(a_max, b_max)
        inter = torch.clamp(inter_max - inter_min, min=0.0)
        inter_vol = float((inter[0] * inter[1] * inter[2]).item())
        if inter_vol <= 0.0:
            return 0.0
        va = torch.clamp(a_max - a_min, min=0.0)
        vb = torch.clamp(b_max - b_min, min=0.0)
        vol_a = float((va[0] * va[1] * va[2]).item())
        vol_b = float((vb[0] * vb[1] * vb[2]).item())
        denom = vol_a + vol_b - inter_vol
        if denom <= 1e-12:
            return 0.0
        return inter_vol / denom

    def _rebuild_instances_snapshot(self):
        if self.all_points is None or self.gaussian_instance_ids is None:
            self.segmentation_instances = []
            return

        out = []
        ids = torch.unique(self.gaussian_instance_ids)
        ids = ids[ids >= 0]
        for inst_id_t in ids:
            inst_id = int(inst_id_t.item())
            idx = torch.where(self.gaussian_instance_ids == inst_id_t)[0]
            if idx.numel() == 0:
                continue
            lbl = self.segmentation_labels[idx]
            valid_lbl = lbl[lbl >= 0]
            if valid_lbl.numel() == 0:
                cls_id = -1
            else:
                cls_id = int(torch.mode(valid_lbl).values.item())

            box = self._instance_bbox(idx)
            if box is None:
                continue
            bmin, bmax, center = box
            obb = self._instance_obb(idx)

            conf = self.semantic_confidence[idx]
            score = float(torch.mean(conf).item()) if conf.numel() > 0 else 0.0

            out.append(
                {
                    "instance_id": inst_id,
                    "class_id": cls_id,
                    "num_points": int(idx.numel()),
                    "score": score,
                    "bbox_min": bmin.detach().cpu().numpy().tolist(),
                    "bbox_max": bmax.detach().cpu().numpy().tolist(),
                    "center": center.detach().cpu().numpy().tolist(),
                    "obb": {
                        "centroid": obb["centroid"].detach().cpu().numpy().tolist() if obb is not None else center.detach().cpu().numpy().tolist(),
                        "axesLengths": obb["axesLengths"].detach().cpu().numpy().tolist() if obb is not None else (bmax - bmin).detach().cpu().numpy().tolist(),
                        "normalizedAxes": obb["normalizedAxes"].detach().cpu().numpy().tolist() if obb is not None else torch.eye(3, device=self.device, dtype=torch.float32).reshape(-1).detach().cpu().numpy().tolist(),
                    },
                    "dominantNormal": obb["dominantNormal"].detach().cpu().numpy().tolist() if obb is not None else [0.0, 0.0, 1.0],
                }
            )

        # Carry tracking metadata (missed_frames, last_seen_frame) across rebuilds.
        track_meta = getattr(self, "_instance_track_meta", {}) or {}
        for inst_dict in out:
            iid = int(inst_dict["instance_id"])
            meta = track_meta.get(iid)
            if meta is not None:
                inst_dict["missed_frames"] = int(meta.get("missed_frames", 0))
                inst_dict["last_seen_frame"] = int(meta.get("last_seen_frame", self._instance_frame_idx))
            else:
                inst_dict["missed_frames"] = 0
                inst_dict["last_seen_frame"] = int(self._instance_frame_idx)

        out.sort(key=lambda x: x["instance_id"])
        self.segmentation_instances = out

    def _update_instances_from_predictions(self, pred_instances, n_map: int):
        """Multi-instance tracking: associate each candidate to an existing instance
        of the same class via 3D bbox IoU + center distance, otherwise spawn a new id.
        Drops instances unseen for `instance_max_missed_frames` frames."""

        if self.gaussian_instance_ids is None:
            return

        if not isinstance(self.segmentation_instances, list):
            self.segmentation_instances = []

        self._instance_frame_idx += 1
        cur_frame = int(self._instance_frame_idx)

        # Snapshot of existing instances keyed by id (used for spatial matching).
        existing: dict[int, dict] = {}
        for inst in self.segmentation_instances:
            iid = int(inst.get("instance_id", -1))
            cls_id = int(inst.get("class_id", -1))
            if iid < 0:
                continue
            bmin_list = inst.get("bbox_min", None)
            bmax_list = inst.get("bbox_max", None)
            if not isinstance(bmin_list, (list, tuple)) or not isinstance(bmax_list, (list, tuple)):
                continue
            if len(bmin_list) != 3 or len(bmax_list) != 3:
                continue
            ibmin = torch.tensor(bmin_list, device=self.device, dtype=torch.float32)
            ibmax = torch.tensor(bmax_list, device=self.device, dtype=torch.float32)
            existing[iid] = {
                "instance_id": iid,
                "class_id": cls_id,
                "bmin": ibmin,
                "bmax": ibmax,
                "center": 0.5 * (ibmin + ibmax),
                "missed_frames": int(inst.get("missed_frames", 0)),
                "last_seen_frame": int(inst.get("last_seen_frame", cur_frame)),
            }

        # Filter and shape candidates.
        candidates = []
        raw_candidates = pred_instances if isinstance(pred_instances, list) else []
        for cand in raw_candidates:
            if not isinstance(cand, dict) or "gaussian_indices" not in cand:
                continue
            idx = cand["gaussian_indices"]
            if not isinstance(idx, torch.Tensor):
                idx = torch.as_tensor(idx, device=self.device)
            idx = idx.to(self.device, dtype=torch.long).reshape(-1)
            idx = idx[(idx >= 0) & (idx < n_map)]
            if idx.numel() == 0:
                continue
            idx = torch.unique(idx)
            if int(idx.numel()) < int(self.instance_min_points):
                continue
            cls_id = int(cand.get("class_id", -1))
            if cls_id < 0:
                continue
            box = self._instance_bbox(idx)
            if box is None:
                continue
            bmin, bmax, center = box
            candidates.append(
                {
                    "cand": cand,
                    "idx": idx,
                    "cls_id": cls_id,
                    "score": float(cand.get("confidence", 0.0)),
                    "bmin": bmin.to(dtype=torch.float32),
                    "bmax": bmax.to(dtype=torch.float32),
                    "center": center.to(dtype=torch.float32),
                }
            )

        # Highest-confidence candidates pick first.
        candidates.sort(key=lambda item: item["score"], reverse=True)

        iou_gate = float(max(0.0, self.instance_iou_gate))
        center_gate = float(max(0.0, self.instance_center_gate_m))
        matched_ids: set[int] = set()

        overlap_gate = float(max(0.0, self.instance_overlap_gate))

        for c in candidates:
            cls_id = c["cls_id"]
            cand_idx = c["idx"]
            cand_size = int(cand_idx.numel())
            best_iid = -1

            # PRIMARY: 3D Gaussian-overlap matching. If this detection's points are
            # already assigned to an existing instance, merge with it — even across
            # classes (Mask2Former labels can flip on a single object frame-to-frame).
            prior_ids = self.gaussian_instance_ids[cand_idx]
            assigned_mask = prior_ids >= 0
            if torch.any(assigned_mask):
                unique_ids, counts = torch.unique(prior_ids[assigned_mask], return_counts=True)
                # Largest existing-instance footprint inside this candidate.
                top = int(torch.argmax(counts).item())
                cand_overlap_iid = int(unique_ids[top].item())
                cand_overlap_count = int(counts[top].item())
                cand_overlap_frac = cand_overlap_count / max(cand_size, 1)
                if cand_overlap_iid not in matched_ids and cand_overlap_frac >= overlap_gate:
                    best_iid = cand_overlap_iid

            # FALLBACK: bbox IoU + center-distance, restricted to same class.
            if best_iid < 0:
                best_score = -1.0
                for iid, ex in existing.items():
                    if iid in matched_ids or int(ex["class_id"]) != cls_id:
                        continue
                    iou = self._bbox_iou_3d(ex["bmin"], ex["bmax"], c["bmin"], c["bmax"])
                    dist = float(torch.linalg.norm(ex["center"] - c["center"]).item())
                    iou_ok = iou >= iou_gate
                    center_ok = (center_gate > 0.0) and (dist <= center_gate)
                    if not (iou_ok or center_ok):
                        continue
                    norm_dist = min(dist / max(center_gate, 1e-6), 1.0) if center_gate > 0.0 else 1.0
                    match_score = iou + (1.0 - norm_dist) * 0.25
                    if match_score > best_score:
                        best_score = match_score
                        best_iid = iid

            if best_iid >= 0:
                assigned = best_iid
            else:
                assigned = int(self._instance_next_id)
                self._instance_next_id += 1
                existing[assigned] = {
                    "instance_id": assigned,
                    "class_id": cls_id,
                    "bmin": c["bmin"],
                    "bmax": c["bmax"],
                    "center": c["center"],
                    "missed_frames": 0,
                    "last_seen_frame": cur_frame,
                }
            matched_ids.add(assigned)

            # Overwrite gaussian assignments inside this candidate's mask.
            self.gaussian_instance_ids[c["idx"]] = int(assigned)
            c["cand"]["instance_id"] = int(assigned)

            ex = existing[assigned]
            ex["bmin"] = torch.minimum(ex["bmin"], c["bmin"])
            ex["bmax"] = torch.maximum(ex["bmax"], c["bmax"])
            ex["center"] = 0.5 * (ex["bmin"] + ex["bmax"])

        # TTL: increment missed_frames on unmatched, drop those past the budget.
        max_missed = int(self.instance_max_missed_frames)
        for iid in list(existing.keys()):
            ex = existing[iid]
            if iid in matched_ids:
                ex["missed_frames"] = 0
                ex["last_seen_frame"] = cur_frame
            else:
                ex["missed_frames"] = int(ex.get("missed_frames", 0)) + 1
                if ex["missed_frames"] > max_missed:
                    drop = self.gaussian_instance_ids == iid
                    if torch.any(drop):
                        self.gaussian_instance_ids[drop] = -1
                    del existing[iid]

        # Tracking metadata persists across snapshot rebuilds.
        self._instance_track_meta = {
            iid: {"missed_frames": ex["missed_frames"], "last_seen_frame": ex["last_seen_frame"]}
            for iid, ex in existing.items()
        }

        self._rebuild_instances_snapshot()

    def _merge_overlapping_instances(self):
        if self.gaussian_instance_ids is None:
            return
        if not isinstance(self.segmentation_instances, list) or len(self.segmentation_instances) < 2:
            return

        instances = list(self.segmentation_instances)
        merged_any = True
        while merged_any:
            merged_any = False
            n = len(instances)
            i = 0
            while i < n:
                a = instances[i]
                j = i + 1
                while j < n:
                    b = instances[j]
                    if int(a.get("class_id", -1)) != int(b.get("class_id", -1)):
                        j += 1
                        continue

                    a_min = torch.tensor(a.get("bbox_min", [0.0, 0.0, 0.0]), device=self.device, dtype=torch.float32)
                    a_max = torch.tensor(a.get("bbox_max", [0.0, 0.0, 0.0]), device=self.device, dtype=torch.float32)
                    b_min = torch.tensor(b.get("bbox_min", [0.0, 0.0, 0.0]), device=self.device, dtype=torch.float32)
                    b_max = torch.tensor(b.get("bbox_max", [0.0, 0.0, 0.0]), device=self.device, dtype=torch.float32)

                    iou = self._bbox_iou_3d(a_min, a_max, b_min, b_max)
                    a_center = 0.5 * (a_min + a_max)
                    b_center = 0.5 * (b_min + b_max)
                    center_dist = float(torch.linalg.norm(a_center - b_center).item())

                    # Only merge highly-overlapping bboxes of the same class — otherwise
                    # distinct instances (two chairs near each other) would collapse.
                    should_merge = iou >= float(self.instance_bbox_merge_iou)
                    if not should_merge:
                        j += 1
                        continue

                    keep_id = min(int(a.get("instance_id", -1)), int(b.get("instance_id", -1)))
                    drop_id = max(int(a.get("instance_id", -1)), int(b.get("instance_id", -1)))
                    if keep_id < 0 or drop_id < 0:
                        j += 1
                        continue

                    self.gaussian_instance_ids[self.gaussian_instance_ids == drop_id] = keep_id

                    a["bbox_min"] = torch.minimum(a_min, b_min).detach().cpu().numpy().tolist()
                    a["bbox_max"] = torch.maximum(a_max, b_max).detach().cpu().numpy().tolist()
                    a["center"] = (0.5 * (torch.tensor(a["bbox_min"], device=self.device) + torch.tensor(a["bbox_max"], device=self.device))).detach().cpu().numpy().tolist()
                    a["score"] = max(float(a.get("score", 0.0)), float(b.get("score", 0.0)))
                    a["missed_frames"] = min(int(a.get("missed_frames", 0)), int(b.get("missed_frames", 0)))
                    a["last_seen_frame"] = max(int(a.get("last_seen_frame", 0)), int(b.get("last_seen_frame", 0)))

                    instances.pop(j)
                    n -= 1
                    merged_any = True
                i += 1

        self.segmentation_instances = instances

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

            # Hard safety gate: never assign classes below detector confidence floor.
            keep_conf = conf >= float(self.semantic_assign_min_confidence)
            idx = idx[keep_conf]
            cls = cls[keep_conf]
            conf = conf[keep_conf]

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
                # If frustum visibility is unknown, avoid global forgetting.
                visible_mask[:] = False

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

                ema_alpha = float(np.clip(self.semantic_ema_alpha, 0.0, 1.0))
                p_cls = self.segmentation_labels[idx]
                p_conf = self.semantic_confidence[idx]
                c_cls = self.semantic_alt_class[idx]
                c_conf = self.semantic_alt_confidence[idx]
                support = self.semantic_switch_support[idx].to(torch.int16)
                cooldown = self.semantic_switch_cooldown[idx].to(torch.int16)

                same_primary = cls == p_cls
                p_obs = ema_alpha * conf + (1.0 - ema_alpha) * p_conf
                p_conf_same = p_conf + p_obs * (1.0 - p_conf)

                challenger_same = cls == c_cls
                c_obs = ema_alpha * conf + (1.0 - ema_alpha) * c_conf
                c_conf_diff = torch.where(challenger_same, c_conf + c_obs * (1.0 - c_conf), conf)
                c_cls_diff = torch.where(challenger_same, c_cls, cls)

                # Softer decay on conflict to avoid rapid class oscillation.
                p_conf_diff = p_conf * (1.0 - 0.15 * conf)
                margin = float(max(0.0, self.semantic_switch_margin))
                switch_conf_gate = float(max(0.0, self.semantic_assign_min_confidence))

                strength_ready = c_conf_diff > (p_conf_diff + margin)
                support_thr = int(max(1, self.semantic_switch_support_frames))
                support_next = torch.where(
                    same_primary,
                    torch.zeros_like(support),
                    torch.where(challenger_same, torch.clamp(support + 1, 0, 255), torch.ones_like(support)),
                )
                enough_support = support_next >= support_thr
                can_switch = cooldown <= 0
                has_switch_conf = c_conf_diff >= switch_conf_gate
                switch = (~same_primary) & (
                    ((p_cls < 0) & has_switch_conf)
                    | (strength_ready & enough_support & can_switch & has_switch_conf)
                )

                new_p_cls = torch.where(same_primary, p_cls, torch.where(switch, c_cls_diff, p_cls))
                new_p_conf = torch.where(same_primary, p_conf_same, torch.where(switch, c_conf_diff, p_conf_diff))

                chall_decay = float(np.clip(self.semantic_challenger_decay_factor, 0.0, 1.0))
                c_conf_same_primary = c_conf * chall_decay
                c_cls_same_primary = c_cls

                c_cls_switched = p_cls
                c_conf_switched = p_conf_diff * 0.5

                c_cls_no_switch = c_cls_diff
                c_conf_no_switch = c_conf_diff

                new_c_cls = torch.where(
                    same_primary,
                    c_cls_same_primary,
                    torch.where(switch, c_cls_switched, c_cls_no_switch),
                )
                new_c_conf = torch.where(
                    same_primary,
                    c_conf_same_primary,
                    torch.where(switch, c_conf_switched, c_conf_no_switch),
                )

                cooldown_frames = int(max(0, self.semantic_switch_cooldown_frames))
                cooldown_next = torch.where(
                    switch,
                    torch.full_like(cooldown, cooldown_frames),
                    torch.clamp(cooldown - 1, min=0),
                )
                support_final = torch.where(
                    switch | same_primary,
                    torch.zeros_like(support_next),
                    support_next,
                )

                self.segmentation_labels[idx] = new_p_cls
                self.semantic_confidence[idx] = torch.clamp(new_p_conf, 0.0, 1.0)
                self.semantic_alt_class[idx] = torch.where(new_c_conf > 0.0, new_c_cls, torch.full_like(new_c_cls, -1))
                self.semantic_alt_confidence[idx] = torch.clamp(new_c_conf, 0.0, 1.0)
                self.semantic_switch_support[idx] = torch.clamp(support_final, 0, 255).to(torch.uint8)
                self.semantic_switch_cooldown[idx] = torch.clamp(cooldown_next, 0, 255).to(torch.uint8)
                detected_mask[idx] = True

            # Temporal consistency: if a labeled point is not classified this frame,
            # decay its confidence instead of dropping the class immediately.
            if n_map > 0:
                decay = float(np.clip(self.semantic_decay_factor, 0.0, 1.0))
                chall_decay = float(np.clip(self.semantic_challenger_decay_factor, 0.0, 1.0))
                # Forget only points inside current frustum scope.
                forget_mask = (self.segmentation_labels >= 0) & (~detected_mask) & visible_mask
                self.semantic_confidence[forget_mask] = self.semantic_confidence[forget_mask] * decay
                self.semantic_alt_confidence[forget_mask] = self.semantic_alt_confidence[forget_mask] * chall_decay
                # Decrease cooldown globally for visible unlabeled updates.
                cool_vis = visible_mask & (~detected_mask)
                self.semantic_switch_cooldown[cool_vis] = torch.clamp(
                    self.semantic_switch_cooldown[cool_vis].to(torch.int16) - 1,
                    min=0,
                ).to(torch.uint8)

            min_conf = float(max(0.0, self.semantic_min_confidence))
            promote = (
                (self.segmentation_labels >= 0)
                & (self.semantic_confidence < min_conf)
                & (self.semantic_alt_class >= 0)
                & (self.semantic_alt_confidence >= float(max(min_conf, self.semantic_assign_min_confidence)))
            )
            if torch.any(promote):
                self.segmentation_labels[promote] = self.semantic_alt_class[promote]
                self.semantic_confidence[promote] = self.semantic_alt_confidence[promote]
                self.semantic_alt_class[promote] = -1
                self.semantic_alt_confidence[promote] = 0.0
                self.semantic_switch_support[promote] = 0
                self.semantic_switch_cooldown[promote] = 0

            low = self.semantic_confidence < min_conf
            self.segmentation_labels[low] = -1
            self.semantic_confidence[low] = 0.0
            self.semantic_alt_class[low] = -1
            self.semantic_alt_confidence[low] = 0.0
            self.semantic_switch_support[low] = 0
            self.semantic_switch_cooldown[low] = 0

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
            if pred_instances is not None:
                bench = getattr(self, "benchmark", None)
                do_bench = bool(getattr(bench, "enabled", False))
                if do_bench:
                    t_assoc0 = time.perf_counter()
                    self._update_instances_from_predictions(pred_instances, n_map)
                    assoc_ms = (time.perf_counter() - t_assoc0) * 1000.0
                    try:
                        from RTSGS.Benchmarking.BenchmarkCollector import BenchmarkCollector

                        bench.record_ms(BenchmarkCollector.STAGE_INSTANCE_ASSOC, float(assoc_ms))
                    except Exception:
                        pass
                else:
                    self._update_instances_from_predictions(pred_instances, n_map)
            elif not isinstance(self.segmentation_instances, list):
                self.segmentation_instances = []

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
            md["num_instances"] = int(len(self.segmentation_instances))
            md["temporal_model"] = "primary_challenger_hysteresis"

            self.segmentation_metadata = md
            self.segmentation_version += 1
            return True