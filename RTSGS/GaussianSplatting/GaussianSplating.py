import torch
import torch.nn.functional as F
from gsplat import rendering, spherical_harmonics
import numpy as np
from pytorch_msssim import ssim
import time  # Necessary for rate limiting

class GaussianSplatting:
    def __init__(self, pcd, dataset, tracker, learning_rate=4e-3, max_steps_per_sec=1, downsample_factor=1.0):
        self.pcd = pcd
        self.dataset = dataset
        self.device = pcd.device
        self.base_lr = learning_rate
        self.tracker = tracker
        self.width, self.height = tracker.config.get('width'), tracker.config.get('height')
        
        # --- Downsampling Attributes ---
        self.downsample_factor = downsample_factor
        self.train_width = int(self.width / downsample_factor)
        self.train_height = int(self.height / downsample_factor)
        
        self.num_points_optimized = 0
        self.optimizer = None
        self.iteration_count = 0

        # --- Rate Limiting ---
        self.max_steps_per_sec = max_steps_per_sec
        self.last_step_time = 0.0
        self.min_interval = 1.0 / max_steps_per_sec if max_steps_per_sec > 0 else 0

        # --- Densification Hyperparameters ---
        self.densify_start_iter = 100
        self.densify_interval = 300
        self.grad_threshold = 0.0000002
        self.xys_grad_norm = None
        self.vis_counts = None

    def _setup_optimizer(self):
        if self.pcd.all_points is None: return

        attrs = ["all_sh", "all_scales", "all_quaternions", "all_alpha"]
        for attr in attrs:
            val = getattr(self.pcd, attr)
            if not isinstance(val, torch.nn.Parameter):
                setattr(self.pcd, attr, torch.nn.Parameter(val.detach().requires_grad_(True)))

        params = [
            {'params': [self.pcd.all_sh], 'lr': self.base_lr * 3.0, "name": "sh"},
            {'params': [self.pcd.all_scales], 'lr': self.base_lr * 3.0, "name": "scales"},
            {'params': [self.pcd.all_quaternions], 'lr': self.base_lr * 1, "name": "quats"},
            {'params': [self.pcd.all_alpha], 'lr': self.base_lr, "name": "alphas"},
        ]
        self.optimizer = torch.optim.Adam(params)
        self.num_points_optimized = self.pcd.all_points.shape[0]

        self.xys_grad_norm = torch.zeros(self.num_points_optimized, device=self.device)
        self.vis_counts = torch.zeros(self.num_points_optimized, device=self.device)

    def densify(self):
        avg_grads = self.xys_grad_norm / (self.vis_counts + 1e-7)
        avg_grads[torch.isnan(avg_grads)] = 0.0
        
        mask = avg_grads >= self.grad_threshold
        num_to_add = mask.sum().item() 
        
        if num_to_add == 0: 
            return

        print(f"\033[92m[Iter {self.iteration_count}] Densifying: {num_to_add} points. "
              f"Total: {self.pcd.all_points.shape[0] + num_to_add}\033[0m")

        with torch.no_grad():
            new_points = self.pcd.all_points[mask].clone()
            new_sh = self.pcd.all_sh[mask].clone()
            new_quats = self.pcd.all_quaternions[mask].clone()
            
            new_scales = torch.full_like(self.pcd.all_scales[mask], -4.0)
            new_alphas = torch.full_like(self.pcd.all_alpha[mask], 0.0)

            self.pcd.all_points = torch.cat([self.pcd.all_points.detach(), new_points.detach()], dim=0)
            self.pcd.all_sh = torch.cat([self.pcd.all_sh.detach(), new_sh.detach()], dim=0)
            self.pcd.all_scales = torch.cat([self.pcd.all_scales.detach(), new_scales.detach()], dim=0)
            self.pcd.all_quaternions = torch.cat([self.pcd.all_quaternions.detach(), new_quats.detach()], dim=0)
            self.pcd.all_alpha = torch.cat([self.pcd.all_alpha.detach(), new_alphas.detach()], dim=0)

        self._setup_optimizer()

    def training_step(self):
        # --- Rate Limit Check ---
        current_time = time.time()
        if current_time - self.last_step_time < self.min_interval:
            return 0.0

        if self.pcd.all_points is None or not self.tracker.keyframes_poses:
            return 0.0

        self.iteration_count += 1
        self.last_step_time = current_time # Update timestamp
        
        if self.optimizer is None or self.pcd.all_points.shape[0] != self.num_points_optimized:
            self._setup_optimizer()

        self.optimizer.zero_grad()
        
        sample_idx = np.random.choice(len(self.tracker.keyframes_poses), min(2, len(self.tracker.keyframes_poses)), replace=False)
        
        # --- Downsample Ground Truth ---
        gt_rgbs_full = torch.stack([torch.from_numpy(self.dataset.rgb_keyframes[i]).to(self.device).float() / 255.0 for i in sample_idx])
        # [B, H, W, 3] -> [B, 3, H, W] for interpolate
        gt_rgbs = F.interpolate(gt_rgbs_full.permute(0, 3, 1, 2), size=(self.train_height, self.train_width), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        
        T_fix = torch.eye(4, device=self.device)
        T_fix[:3, :3] = self.pcd.R_fix

        viewmats = []
        cam_centers = []
        for i in sample_idx:
            pose = torch.from_numpy(self.tracker.keyframes_poses[i]).to(self.device).float()
            w2c = torch.inverse(T_fix @ pose)
            viewmats.append(w2c)
            cam_centers.append((T_fix @ pose)[:3, 3])

        viewmats = torch.stack(viewmats)
        cam_centers = torch.stack(cam_centers)

        # --- Scale Intrinsics for Downsampling ---
        K = torch.eye(3, device=self.device)
        K[0,0], K[1,1] = self.pcd.fx / self.downsample_factor, self.pcd.fy / self.downsample_factor
        K[0,2], K[1,2] = self.pcd.cx / self.downsample_factor, self.pcd.cy / self.downsample_factor
        Ks = K.unsqueeze(0).expand(len(sample_idx), -1, -1)

        means = self.pcd.all_points
        dirs = means.unsqueeze(0) - cam_centers.unsqueeze(1)
        dirs = F.normalize(dirs, dim=-1)

        sh_coeffs = self.pcd.all_sh.unsqueeze(0).expand(len(sample_idx), -1, -1, -1)
        colors_pre_activation = spherical_harmonics(self.pcd.sh_degree, dirs, sh_coeffs)
        colors = torch.sigmoid(colors_pre_activation)

        # --- Rasterize at Downsampled Resolution ---
        rendered_rgb, _, info = rendering.rasterization(
            means=self.pcd.all_points,
            quats=F.normalize(self.pcd.all_quaternions, p=2, dim=-1),
            scales=torch.exp(self.pcd.all_scales), 
            opacities=torch.sigmoid(self.pcd.all_alpha).squeeze(-1),
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=self.train_width,
            height=self.train_height,
        )

        info["means2d"].retain_grad()

        l1_loss = F.l1_loss(rendered_rgb, gt_rgbs)
        ssim_val = ssim(rendered_rgb.permute(0, 3, 1, 2), gt_rgbs.permute(0, 3, 1, 2), data_range=1.0)
        total_loss = 0.8 * l1_loss + 0.2 * (1.0 - ssim_val)
        
        if total_loss > 0:
            total_loss.backward()

            with torch.no_grad():
                grads_2d = info["means2d"].grad
                v_norms = torch.norm(grads_2d[:, :2], dim=-1)
                gi_ids = info["gaussian_ids"].long() 
                
                self.xys_grad_norm.scatter_add_(0, gi_ids, v_norms)
                self.vis_counts.scatter_add_(0, gi_ids, torch.ones_like(v_norms))

            self.optimizer.step()

        if self.iteration_count > self.densify_start_iter and self.iteration_count % self.densify_interval == 0:
            self.densify()
                
        return total_loss.item()