import time
import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rendering, spherical_harmonics
from pytorch_msssim import ssim


class TokenBucket:
    """
    Token-bucket rate limiter.

    rate: tokens added per second
    burst: max tokens that can accumulate (allows small bursts)
    """

    def __init__(self, rate: float, burst: float = 1.0):
        self.rate = float(rate)
        self.burst = float(burst)
        self.tokens = 0.0
        self.last_t = time.time()

    def allow(self, cost: float = 1.0) -> bool:
        """Return True if we can spend `cost` tokens now, else False."""
        if self.rate <= 0:
            return False

        now = time.time()
        dt = now - self.last_t
        self.last_t = now

        # accrue tokens
        self.tokens = min(self.burst, self.tokens + dt * self.rate)

        # spend tokens if available
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False


def _build_K(fx: float, fy: float, cx: float, cy: float, device: torch.device) -> torch.Tensor:
    K = torch.eye(3, device=device, dtype=torch.float32)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    return K


@torch.no_grad()
def frustum_cull_mask(
    means_world: torch.Tensor,      # (N,3)
    viewmats: torch.Tensor,         # (B,4,4) world->cam
    Ks: torch.Tensor,               # (B,3,3)
    width: int,
    height: int,
    near: float = 0.01,
    far: float = 1e6,
    pad: float = 2.0,               # pixels padding (keep a little outside to avoid popping)
) -> torch.Tensor:
    """
    Conservative frustum culling on GPU.
    Returns mask (N,) marking gaussians that are inside ANY of the B views.

    We ignore gaussian size here (fast). If you want more conservative culling, increase pad.
    """
    device = means_world.device
    N = means_world.shape[0]
    B = viewmats.shape[0]

    ones = torch.ones((N, 1), device=device, dtype=means_world.dtype)
    Pw = torch.cat([means_world, ones], dim=-1)  # (N,4)

    Pc = torch.matmul(Pw.unsqueeze(0), viewmats.transpose(1, 2))  # (B,N,4)

    X = Pc[..., 0]
    Y = Pc[..., 1]
    Z = Pc[..., 2]

    in_z = (Z > near) & (Z < far)

    fx = Ks[:, 0, 0].unsqueeze(1)
    fy = Ks[:, 1, 1].unsqueeze(1)
    cx = Ks[:, 0, 2].unsqueeze(1)
    cy = Ks[:, 1, 2].unsqueeze(1)

    invZ = torch.reciprocal(Z.clamp_min(1e-12))
    u = fx * (X * invZ) + cx
    v = fy * (Y * invZ) + cy

    in_u = (u >= -pad) & (u <= (width - 1 + pad))
    in_v = (v >= -pad) & (v <= (height - 1 + pad))

    in_view = in_z & in_u & in_v   # (B,N)
    mask = in_view.any(dim=0)      # (N,)
    return mask


class GaussianSplatting:
    def __init__(self, pcd, dataset, tracker, learning_rate=7e-3, max_steps_per_sec=10, downsample_factor=2.0):
        self.pcd = pcd
        self.dataset = dataset
        self.device = pcd.device
        self.base_lr = learning_rate
        self.tracker = tracker
        self.width, self.height = tracker.config.get('width'), tracker.config.get('height')

        self.downsample_factor = downsample_factor
        self.train_width = int(self.width / downsample_factor)
        self.train_height = int(self.height / downsample_factor)

        self.num_points_optimized = 0
        self.optimizer = None
        self.iteration_count = 0

        # Token-bucket limiter: smooth "max_steps_per_sec"
        self.step_limiter = TokenBucket(rate=max_steps_per_sec, burst=2.0)

        self.densify_start_iter = 100
        self.densify_interval = 300
        self.grad_threshold = 0.0000002
        self.xys_grad_norm = None
        self.vis_counts = None

        # Culling params (tune)
        self.cull_near = 0.05
        self.cull_far = 50.0
        self.cull_pad_px = 4.0
        self.min_culled_points = 2048  # avoid pathological tiny sets

    def _setup_optimizer(self):
        if self.pcd.all_points is None:
            return

        attrs = ["all_sh", "all_scales", "all_quaternions", "all_alpha"]
        for attr in attrs:
            val = getattr(self.pcd, attr)
            if not isinstance(val, torch.nn.Parameter):
                setattr(self.pcd, attr, torch.nn.Parameter(val.detach().requires_grad_(True)))

        params = [
            {'params': [self.pcd.all_sh], 'lr': self.base_lr * 3.0, "name": "sh"},
            {'params': [self.pcd.all_scales], 'lr': self.base_lr * 3.0, "name": "scales"},
            {'params': [self.pcd.all_quaternions], 'lr': self.base_lr * 1.0, "name": "quats"},
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

        print(
            f"\033[92m[Iter {self.iteration_count}] Densifying: {num_to_add} points. "
            f"Total: {self.pcd.all_points.shape[0] + num_to_add}\033[0m"
        )

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
        # Smoothly limit optimization steps to ~max_steps_per_sec
        if not self.step_limiter.allow(cost=1.0):
            return 0.0

        # no lock (as requested)
        if self.pcd.all_points is None or not self.tracker.keyframes_poses:
            return 0.0

        self.iteration_count += 1

        if self.optimizer is None or self.pcd.all_points.shape[0] != self.num_points_optimized:
            self._setup_optimizer()
            if self.optimizer is None:
                return 0.0

        self.optimizer.zero_grad(set_to_none=True)

        means_all = self.pcd.all_points
        sh_all = self.pcd.all_sh
        scales_all = self.pcd.all_scales
        quats_all = self.pcd.all_quaternions
        alpha_all = self.pcd.all_alpha
        sh_degree = self.pcd.sh_degree
        R_fix = self.pcd.R_fix

        kf_count = len(self.tracker.keyframes_poses)
        b = 2 if kf_count >= 2 else 1
        sample_idx = np.random.choice(kf_count, b, replace=False)
        gt_rgb_np = [self.dataset.rgb_keyframes[i] for i in sample_idx]
        poses_np = [self.tracker.keyframes_poses[i].copy() for i in sample_idx]

        # 1) GT (downsample)
        gt_rgbs_full = torch.stack(
            [torch.from_numpy(img).to(self.device).float().mul_(1.0 / 255.0) for img in gt_rgb_np]
        )
        gt_rgbs = F.interpolate(
            gt_rgbs_full.permute(0, 3, 1, 2),
            size=(self.train_height, self.train_width),
            mode='bilinear',
            align_corners=False
        ).permute(0, 2, 3, 1)

        # 2) Camera mats
        T_fix = torch.eye(4, device=self.device)
        T_fix[:3, :3] = R_fix

        viewmats = []
        cam_centers = []
        for p_np in poses_np:
            pose = torch.from_numpy(p_np).to(self.device).float()
            w2c = torch.inverse(T_fix @ pose)
            viewmats.append(w2c)
            cam_centers.append((T_fix @ pose)[:3, 3])

        viewmats = torch.stack(viewmats, dim=0)
        cam_centers = torch.stack(cam_centers, dim=0)

        K = _build_K(
            fx=float(self.pcd.fx / self.downsample_factor),
            fy=float(self.pcd.fy / self.downsample_factor),
            cx=float(self.pcd.cx / self.downsample_factor),
            cy=float(self.pcd.cy / self.downsample_factor),
            device=self.device,
        )
        Ks = K.unsqueeze(0).expand(b, -1, -1)

        # FRUSTUM CULL (before SH + rasterization)
        mask = frustum_cull_mask(
            means_world=means_all,
            viewmats=viewmats,
            Ks=Ks,
            width=self.train_width,
            height=self.train_height,
            near=self.cull_near,
            far=self.cull_far,
            pad=self.cull_pad_px,
        )

        # Ensure we don't end up with too few points
        n_keep = int(mask.sum().item())
        if n_keep < self.min_culled_points:
            means = means_all
            sh = sh_all
            scales = scales_all
            quats = quats_all
            alpha = alpha_all
            culled_idx = None
        else:
            culled_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
            means = means_all[culled_idx]
            sh = sh_all[culled_idx]
            scales = scales_all[culled_idx]
            quats = quats_all[culled_idx]
            alpha = alpha_all[culled_idx]

        # 4) SH colors on culled set
        dirs = means.unsqueeze(0) - cam_centers.unsqueeze(1)
        dirs = F.normalize(dirs, dim=-1)

        sh_coeffs = sh.unsqueeze(0).expand(b, -1, -1, -1)
        colors = torch.sigmoid(spherical_harmonics(sh_degree, dirs, sh_coeffs))

        # 5) Rasterization on culled set
        rendered_rgb, _, info = rendering.rasterization(
            means=means,
            quats=F.normalize(quats, p=2, dim=-1),
            scales=torch.exp(scales),
            opacities=torch.sigmoid(alpha).squeeze(-1),
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=self.train_width,
            height=self.train_height,
        )

        # 6) Loss/backward
        l1_loss = F.l1_loss(rendered_rgb, gt_rgbs)
        ssim_val = ssim(
            rendered_rgb.permute(0, 3, 1, 2),
            gt_rgbs.permute(0, 3, 1, 2),
            data_range=1.0
        )
        total_loss = 0.8 * l1_loss + 0.2 * (1.0 - ssim_val)

        if total_loss > 0:
            total_loss.backward()
            self.optimizer.step()

        # 7) Densify
        if self.iteration_count > self.densify_start_iter and self.iteration_count % self.densify_interval == 0:
            self.densify()

        return float(total_loss.detach().item())