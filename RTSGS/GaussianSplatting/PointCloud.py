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

        # Camera Parameters
        self.fx = torch.tensor(float(config.get("fx")), device=self.device)
        self.fy = torch.tensor(float(config.get("fy")), device=self.device)
        self.cx = torch.tensor(float(config.get("cx")), device=self.device)
        self.cy = torch.tensor(float(config.get("cy")), device=self.device)

        self.depth_scale = float(config.get("depth_scale", 1.0))
        self.voxel_size = float(config.get("voxel_size", 0.02))
        self.novelty_voxel = float(config.get("novelty_voxel", self.voxel_size))

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
    def process_single_keyframe(self, rgb_np, depth_np, pose_np):
        """Processes a single keyframe: Image + Depth + Pose -> Filtered Gaussians."""
        rgb = torch.from_numpy(rgb_np).to(self.device).float() / 255.0
        depth = torch.from_numpy(depth_np).to(self.device).float() / self.depth_scale
        pose = torch.from_numpy(pose_np).to(self.device).float()

        H, W = depth.shape
        z_raw = depth.reshape(-1)
        mask = z_raw > 0
        if self.pixel_subsample < 1.0:
            mask &= (torch.rand(z_raw.shape, device=self.device) < self.pixel_subsample)
        
        indices = torch.where(mask)[0]
        if indices.numel() == 0: return None

        z_f = z_raw[indices]
        points_cam = torch.stack([((indices % W).float() - self.cx) * z_f / self.fx,
                                  ((indices // W).float() - self.cy) * z_f / self.fy,
                                  z_f], dim=1)

        R_corr = self.R_fix @ pose[:3, :3]
        t_corr = (self.R_fix @ pose[:3, 3].unsqueeze(-1)).squeeze(-1)

        points_world = (R_corr @ points_cam.T).T + t_corr
        colors = rgb.reshape(-1, 3)[indices]

        res = self.novelty_filter_fast_with_gaussians(points_world, colors, self.novelty_voxel)
        if res is None: return None
        
        pts, cols, k_idx = res
        
        # SH conversion
        sh_full = torch.zeros((pts.shape[0], self.num_sh_bases, 3), device=self.device)
        sh_full[:, 0, :] = torch.logit(torch.clamp(cols, 0.001, 0.999)) / 0.28209479177387814

        # Attribute generation
        scales, quats, alpha = self._make_gaussians_batch(points_cam[k_idx], R_corr, z_f[k_idx])
        return self.voxel_filter_with_gaussians(pts, sh_full, scales, quats, alpha, self.voxel_size)

    def update_async(self, rgb_np, depth_np, pose_np):
        """Non-blocking call to process and add a keyframe to the map."""
        if self.is_processing:
            return False # Busy processing previous frame
        
        self.is_processing = True

        def _task():
            try:
                new_data = self.process_single_keyframe(rgb_np, depth_np, pose_np)
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

    def get_map(self):
        return self.all_points, self.all_sh, self.all_scales, self.all_quaternions, self.all_alpha