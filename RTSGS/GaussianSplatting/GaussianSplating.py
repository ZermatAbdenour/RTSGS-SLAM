import time
from collections import deque
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
        self.tokens = self.burst
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
    def __init__(self, pcd, dataset, tracker, learning_rate=4e-4, max_steps_per_sec=1000, downsample_factor=1.0):
        self.pcd = pcd
        self.dataset = dataset
        self.device = pcd.device
        self.base_lr = learning_rate
        self.tracker = tracker
        self.rgb_width, self.rgb_height = tracker.config.get_rgb_size()
        self.depth_width, self.depth_height = tracker.config.get_depth_size()

        self.downsample_factor = downsample_factor
        self.train_width = int(self.rgb_width / downsample_factor)
        self.train_height = int(self.rgb_height / downsample_factor)

        self.num_points_optimized = 0
        self.optimizer = None
        self.iteration_count = 0

        # Loss and optimization knobs
        self.points_lr_mult = float(self.tracker.config.get('gs_points_lr_mult', 0.3))
        self.depth_loss_weight = float(self.tracker.config.get('gs_depth_loss_weight', 0.1))
        self.depth_huber_delta = float(self.tracker.config.get('gs_depth_huber_delta', 0.05))

        # Token-bucket limiter: smooth "max_steps_per_sec"
        self.step_limiter = TokenBucket(rate=max_steps_per_sec, burst=100.0)

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
        # Step rate tracking (timestamps of recent optimizer steps)
        self._step_timestamps = deque()
        self.steps_per_sec = 0.0

    @torch.no_grad()
    def render_depth_at_pose(self, pose_np: np.ndarray):
        """
        Render expected depth (meters) for a given camera pose (world-from-camera 4x4).
        Returns depth map (H, W) float32 on CPU, or None when rendering is not possible.
        """
        if self.pcd.all_points is None:
            return None

        if pose_np is None:
            return None

        means = self.pcd.all_points
        quats = self.pcd.all_quaternions
        scales = self.pcd.all_scales
        alpha = self.pcd.all_alpha

        if means is None or means.shape[0] == 0:
            return None

        pose = torch.from_numpy(np.asarray(pose_np, dtype=np.float32)).to(self.device)

        T_fix = torch.eye(4, device=self.device, dtype=torch.float32)
        T_fix[:3, :3] = self.pcd.R_fix
        viewmat = torch.inverse(T_fix @ pose).unsqueeze(0)

        K = _build_K(
            fx=float(self.pcd.fx),
            fy=float(self.pcd.fy),
            cx=float(self.pcd.cx),
            cy=float(self.pcd.cy),
            device=self.device,
        ).unsqueeze(0)

        ones = torch.ones((means.shape[0], 3), device=self.device, dtype=means.dtype)

        try:
            rendered, _, _ = rendering.rasterization(
                means=means,
                quats=F.normalize(quats, p=2, dim=-1),
                scales=torch.exp(scales),
                opacities=torch.sigmoid(alpha).squeeze(-1),
                colors=ones,
                viewmats=viewmat,
                Ks=K,
                width=self.depth_width,
                height=self.depth_height,
                render_mode="ED",
            )
        except Exception:
            return None

        # ED may return (..., 1) or (...,) depending on gsplat version.
        if rendered.ndim == 4 and rendered.shape[-1] >= 1:
            depth = rendered[0, ..., 0]
        elif rendered.ndim == 3:
            depth = rendered[0, ...]
        else:
            return None

        depth = depth.detach().to(torch.float32).cpu().numpy()
        depth[~np.isfinite(depth)] = 0.0
        depth[depth < 0.0] = 0.0
        return depth

    @torch.no_grad()
    def render_rgb_at_pose(self, pose_np: np.ndarray):
        """
        Render RGB image (uint8 HxWx3) for a given camera pose (world-from-camera 4x4).
        Returns numpy uint8 array or None on failure.
        """
        if self.pcd.all_points is None:
            return None

        if pose_np is None:
            return None

        means = self.pcd.all_points
        quats = self.pcd.all_quaternions
        scales = self.pcd.all_scales
        alpha = self.pcd.all_alpha
        sh = self.pcd.all_sh

        if means is None or means.shape[0] == 0:
            return None

        pose = torch.from_numpy(np.asarray(pose_np, dtype=np.float32)).to(self.device)

        T_fix = torch.eye(4, device=self.device, dtype=torch.float32)
        T_fix[:3, :3] = self.pcd.R_fix
        viewmat = torch.inverse(T_fix @ pose).unsqueeze(0)

        K = _build_K(
            fx=float(self.pcd.rgb_fx),
            fy=float(self.pcd.rgb_fy),
            cx=float(self.pcd.rgb_cx),
            cy=float(self.pcd.rgb_cy),
            device=self.device,
        ).unsqueeze(0)

        try:
            img, _, _ = rendering.rasterization(
                means=means,
                quats=F.normalize(quats, p=2, dim=-1),
                scales=torch.exp(scales),
                opacities=torch.sigmoid(alpha).squeeze(-1),
                colors=(torch.sigmoid(spherical_harmonics(self.pcd.sh_degree, (means - ((T_fix @ pose)[:3,3]).to(self.device)), sh)) if sh is not None else torch.ones((means.shape[0],3), device=self.device)),
                viewmats=viewmat,
                Ks=K,
                width=self.rgb_width,
                height=self.rgb_height,
                render_mode="RGB",
            )
        except Exception:
            return None

        try:
            out = (img[0].clamp_(0.0, 1.0).mul_(255.0)).to(torch.uint8).cpu().numpy()
            return out
        except Exception:
            return None

    @torch.no_grad()
    def render_segmentation_rgb_at_pose(self, pose_np: np.ndarray):
        """
        Render a segmentation-colored point cloud image for a given camera pose.

        Uses per-point colors from `pcd.segmentation_colors` (Nx3 float in [0,1]) and
        the SAME RGB camera intrinsics + resolution as `render_rgb_at_pose`, so the
        FOV matches the camera-side renders.

        Returns uint8 HxWx3 (RGB), or None if segmentation is unavailable.
        """
        if self.pcd.all_points is None:
            return None
        if pose_np is None:
            return None

        means = self.pcd.all_points
        quats = self.pcd.all_quaternions
        scales = self.pcd.all_scales
        alpha = self.pcd.all_alpha

        if means is None or means.shape[0] == 0:
            return None

        seg_colors = getattr(self.pcd, "segmentation_colors", None)
        if seg_colors is None:
            return None
        if torch.is_tensor(seg_colors):
            seg_colors_t = seg_colors
        else:
            seg_colors_t = torch.from_numpy(np.asarray(seg_colors, dtype=np.float32))
        if seg_colors_t.ndim != 2 or seg_colors_t.shape[1] != 3:
            return None
        if int(seg_colors_t.shape[0]) != int(means.shape[0]):
            return None

        pose = torch.from_numpy(np.asarray(pose_np, dtype=np.float32)).to(self.device)

        T_fix = torch.eye(4, device=self.device, dtype=torch.float32)
        T_fix[:3, :3] = self.pcd.R_fix
        viewmat = torch.inverse(T_fix @ pose).unsqueeze(0)

        K = _build_K(
            fx=float(self.pcd.rgb_fx),
            fy=float(self.pcd.rgb_fy),
            cx=float(self.pcd.rgb_cx),
            cy=float(self.pcd.rgb_cy),
            device=self.device,
        ).unsqueeze(0)

        seg_colors_t = seg_colors_t.to(self.device, dtype=torch.float32).clamp(0.0, 1.0)

        try:
            img, _, _ = rendering.rasterization(
                means=means,
                quats=F.normalize(quats, p=2, dim=-1),
                scales=torch.exp(scales),
                opacities=torch.sigmoid(alpha).squeeze(-1),
                colors=seg_colors_t,
                viewmats=viewmat,
                Ks=K,
                width=self.rgb_width,
                height=self.rgb_height,
                render_mode="RGB",
            )
        except Exception:
            return None

        try:
            out = (img[0].clamp_(0.0, 1.0).mul_(255.0)).to(torch.uint8).cpu().numpy()
            return out
        except Exception:
            return None

    @torch.no_grad()
    def _compute_visibility_signature(self, pose_np: np.ndarray):
        if self.pcd.all_points is None or self.pcd.all_points.shape[0] == 0:
            return None

        if pose_np is None:
            return None

        pose = torch.from_numpy(np.asarray(pose_np, dtype=np.float32)).to(self.device)

        T_fix = torch.eye(4, device=self.device, dtype=torch.float32)
        T_fix[:3, :3] = self.pcd.R_fix
        viewmat = torch.inverse(T_fix @ pose).unsqueeze(0)

        K = _build_K(
            fx=float(self.pcd.rgb_fx / self.downsample_factor),
            fy=float(self.pcd.rgb_fy / self.downsample_factor),
            cx=float(self.pcd.rgb_cx / self.downsample_factor),
            cy=float(self.pcd.rgb_cy / self.downsample_factor),
            device=self.device,
        ).unsqueeze(0)

        mask = frustum_cull_mask(
            means_world=self.pcd.all_points.detach(),
            viewmats=viewmat,
            Ks=K,
            width=self.train_width,
            height=self.train_height,
            near=self.cull_near,
            far=self.cull_far,
            pad=self.cull_pad_px,
        )

        return mask.detach().cpu().numpy()

    def _ensure_keyframe_covis_masks(self):
        if not hasattr(self.tracker, "keyframes_covis_masks"):
            self.tracker.keyframes_covis_masks = []

        kf_masks = self.tracker.keyframes_covis_masks
        kf_count = len(self.tracker.keyframes_poses)

        if self.pcd.all_points is None or kf_count == 0:
            return

        if len(kf_masks) > kf_count:
            del kf_masks[kf_count:]

        start = len(kf_masks)
        if start >= kf_count:
            return

        for kf_idx in range(start, kf_count):
            mask = self._compute_visibility_signature(self.tracker.keyframes_poses[kf_idx])
            kf_masks.append(mask)

    @staticmethod
    def _covisibility_score(mask_a, mask_b) -> float:
        if mask_a is None or mask_b is None:
            return 0.0

        if torch.is_tensor(mask_a):
            mask_a = mask_a.detach().cpu().numpy()
        if torch.is_tensor(mask_b):
            mask_b = mask_b.detach().cpu().numpy()

        min_len = min(mask_a.shape[0], mask_b.shape[0])
        if min_len <= 0:
            return 0.0

        a = mask_a[:min_len]
        b = mask_b[:min_len]

        inter = int(np.count_nonzero(a & b))
        if inter == 0:
            return 0.0

        count_a = int(np.count_nonzero(a))
        count_b = int(np.count_nonzero(b))
        denom = min(count_a, count_b)
        if denom <= 0:
            return 0.0

        return float(inter) / float(denom)

    def _setup_optimizer(self):
        if self.pcd.all_points is None:
            return

        old_optimizer = self.optimizer
        old_num_points = int(self.num_points_optimized)

        attrs = ["all_points","all_sh", "all_scales", "all_quaternions", "all_alpha"]
        for attr in attrs:
            val = getattr(self.pcd, attr)
            if not isinstance(val, torch.nn.Parameter):
                setattr(self.pcd, attr, torch.nn.Parameter(val.detach().requires_grad_(True)))

        params = [
            {'params': [self.pcd.all_points], 'lr': self.base_lr * self.points_lr_mult, "name": "points"},
            {'params': [self.pcd.all_sh], 'lr': self.base_lr * 3.0, "name": "sh"},
            {'params': [self.pcd.all_scales], 'lr': self.base_lr * 2.0, "name": "scales"},
            {'params': [self.pcd.all_quaternions], 'lr': self.base_lr * 2.0, "name": "quats"},
            {'params': [self.pcd.all_alpha], 'lr': self.base_lr, "name": "alphas"},
        ]
        self.optimizer = torch.optim.Adam(params)

        # Preserve Adam momentum for pre-existing gaussians when the map grows.
        if old_optimizer is not None:
            old_groups = {g.get("name", f"g{i}"): g for i, g in enumerate(old_optimizer.param_groups)}
            for i, new_group in enumerate(self.optimizer.param_groups):
                gname = new_group.get("name", f"g{i}")
                old_group = old_groups.get(gname)
                if old_group is None or len(old_group.get("params", [])) == 0:
                    continue
                if len(new_group.get("params", [])) == 0:
                    continue

                old_param = old_group["params"][0]
                new_param = new_group["params"][0]

                old_state = old_optimizer.state.get(old_param, None)
                if old_state is None:
                    continue

                new_state = self.optimizer.state[new_param]

                if "step" in old_state:
                    old_step = old_state["step"]
                    new_state["step"] = old_step.clone() if torch.is_tensor(old_step) else old_step

                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    old_tensor = old_state.get(key, None)
                    if old_tensor is None or (not torch.is_tensor(old_tensor)):
                        continue

                    new_tensor = torch.zeros_like(new_param.data)

                    if old_tensor.shape == new_tensor.shape:
                        new_tensor.copy_(old_tensor)
                    elif old_tensor.ndim == new_tensor.ndim and old_tensor.ndim > 0 and old_tensor.shape[1:] == new_tensor.shape[1:]:
                        n_copy = min(int(old_tensor.shape[0]), int(new_tensor.shape[0]))
                        if n_copy > 0:
                            new_tensor[:n_copy].copy_(old_tensor[:n_copy])

                    new_state[key] = new_tensor

        self.num_points_optimized = self.pcd.all_points.shape[0]
        if self.xys_grad_norm is None or self.vis_counts is None:
            self.xys_grad_norm = torch.zeros(self.num_points_optimized, device=self.device)
            self.vis_counts = torch.zeros(self.num_points_optimized, device=self.device)
        else:
            new_grad_norm = torch.zeros(self.num_points_optimized, device=self.device)
            new_vis_counts = torch.zeros(self.num_points_optimized, device=self.device)
            n_copy = min(old_num_points, self.num_points_optimized)
            if n_copy > 0:
                new_grad_norm[:n_copy] = self.xys_grad_norm[:n_copy]
                new_vis_counts[:n_copy] = self.vis_counts[:n_copy]
            self.xys_grad_norm = new_grad_norm
            self.vis_counts = new_vis_counts

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

        # Always include the latest keyframe; optionally add a covisible older one.
        self._ensure_keyframe_covis_masks()
        latest_idx = kf_count - 1
        sample_idx = np.array([latest_idx], dtype=np.int64)

        if kf_count > 1:
            kf_masks = getattr(self.tracker, "keyframes_covis_masks", [])
            if len(kf_masks) > latest_idx:
                latest_mask = kf_masks[latest_idx]
                if latest_mask is not None:
                    candidates = []
                    for old_idx in range(latest_idx):
                        if old_idx >= len(kf_masks):
                            break
                        score = self._covisibility_score(latest_mask, kf_masks[old_idx])
                        if score > 0.0:
                            candidates.append(old_idx)
                    if candidates:
                        old_idx = np.random.choice(candidates, 1, replace=False)
                        sample_idx = np.concatenate(
                            (np.array([latest_idx], dtype=np.int64), old_idx.astype(np.int64))
                        )

        b = len(sample_idx)
        gt_rgb_np = [self.dataset.rgb_keyframes[i] for i in sample_idx]
        can_use_depth = (
            self.depth_loss_weight > 0.0
            and len(self.dataset.depth_keyframes) > 0
            and max(sample_idx) < len(self.dataset.depth_keyframes)
        )
        gt_depth_np = [self.dataset.depth_keyframes[i] for i in sample_idx] if can_use_depth else None
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

        gt_depths = None
        if gt_depth_np is not None:
            gt_depths_full = torch.stack(
                [torch.from_numpy(dep).to(self.device).float().div_(float(self.pcd.depth_scale)) for dep in gt_depth_np]
            )
            gt_depths = F.interpolate(
                gt_depths_full.unsqueeze(1),
                size=(self.train_height, self.train_width),
                mode='nearest'
            ).squeeze(1)

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
            fx=float(self.pcd.rgb_fx / self.downsample_factor),
            fy=float(self.pcd.rgb_fy / self.downsample_factor),
            cx=float(self.pcd.rgb_cx / self.downsample_factor),
            cy=float(self.pcd.rgb_cy / self.downsample_factor),
            device=self.device,
        )
        Ks = K.unsqueeze(0).expand(b, -1, -1)

        # 3) Frustum culling
        with torch.no_grad():
            cull_mask = frustum_cull_mask(
                means_world=means_all.detach(),
                viewmats=viewmats.detach(),
                Ks=Ks.detach(),
                width=self.train_width,
                height=self.train_height,
                near=self.cull_near,
                far=self.cull_far,
                pad=self.cull_pad_px,
            )

            # Avoid too small set
            if cull_mask.sum() < self.min_culled_points:
                cull_mask = torch.ones_like(cull_mask, dtype=torch.bool)

        idx = torch.where(cull_mask)[0]

        means = means_all[idx]
        sh = sh_all[idx]
        scales = scales_all[idx]
        quats = quats_all[idx]
        alpha = alpha_all[idx]

        # 4) SH colors on culled set
        dirs = means.unsqueeze(0) - cam_centers.unsqueeze(1)
        dirs = F.normalize(dirs, dim=-1)

        sh_coeffs = sh.unsqueeze(0).expand(b, -1, -1, -1)
        colors = torch.sigmoid(spherical_harmonics(sh_degree, dirs, sh_coeffs))

        # 5) Rasterization on culled set

        rendered, _, info = rendering.rasterization(
            means=means,
            quats=F.normalize(quats, p=2, dim=-1),
            scales=torch.exp(scales),
            opacities=torch.sigmoid(alpha).squeeze(-1),
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=self.train_width,
            height=self.train_height,
            render_mode="RGB+ED",
        )


        rendered_rgb = rendered[..., :3]
        rendered_depth = rendered[..., 3] if rendered.shape[-1] > 3 else None

        # 6) Loss/backward
        l1_loss = F.l1_loss(rendered_rgb, gt_rgbs)
        # ssim_val = ssim(
        #     rendered_rgb.permute(0, 3, 1, 2),
        #     gt_rgbs.permute(0, 3, 1, 2),
        #     data_range=1.0
        # )
        rgb_loss = l1_loss #+ 0.2 * (1.0 - ssim_val)

        depth_loss = torch.zeros((), device=self.device)
        if gt_depths is not None and rendered_depth is not None:
            valid = (gt_depths > 0.0) & torch.isfinite(gt_depths) & torch.isfinite(rendered_depth) & (rendered_depth > 0.0)
            if valid.any():
                depth_loss = F.smooth_l1_loss(
                    rendered_depth[valid],
                    gt_depths[valid],
                    beta=self.depth_huber_delta,
                )

        total_loss = rgb_loss + self.depth_loss_weight * depth_loss

        if total_loss > 0:
            total_loss.backward()
            self.optimizer.step()
            # record a performed optimization step timestamp and compute recent rate
       
            now = time.time()
            self._step_timestamps.append(now)
            # drop older than 1s
            cutoff = now - 1.0
            while self._step_timestamps and self._step_timestamps[0] < cutoff:
                self._step_timestamps.popleft()
            self.steps_per_sec = float(len(self._step_timestamps))
        # 7) Densify
        if self.iteration_count > self.densify_start_iter and self.iteration_count % self.densify_interval == 0:
            self.densify()

        return float(total_loss.detach().item())