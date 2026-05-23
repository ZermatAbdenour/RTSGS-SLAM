import time
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rendering, spherical_harmonics
from gsplat.strategy import DefaultStrategy
from gsplat.strategy.ops import duplicate, split


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


class AbsoluteScaleStrategy(DefaultStrategy):
    """DefaultStrategy with an absolute split threshold instead of scene_scale-relative.

    In incremental SLAM the Gaussians are initialised with small, depth-dependent
    scales.  The default ``grow_scale3d * scene_scale`` threshold grows with the
    map and almost always exceeds the actual Gaussian sizes, so splitting never
    fires.  This subclass replaces that check with a fixed world-space threshold.
    """

    split_scale_abs: float = 0.01
    grow_min_opa: float = 0.5

    def __init__(self, split_scale_abs: float = 0.01, grow_min_opa: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.split_scale_abs = split_scale_abs
        self.grow_min_opa = grow_min_opa

    @torch.no_grad()
    def _grow_gs(self, params, optimizers, state, step):
        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        device = grads.device

        is_grad_high = grads > self.grow_grad2d
        is_confident = torch.sigmoid(params["opacities"]) >= self.grow_min_opa
        is_grad_high = is_grad_high & is_confident
        is_small = (
            torch.exp(params["scales"]).max(dim=-1).values
            <= self.split_scale_abs
        )
        is_dupli = is_grad_high & is_small
        n_dupli = is_dupli.sum().item()

        is_large = ~is_small
        is_split = is_grad_high & is_large
        if step < self.refine_scale2d_stop_iter:
            if "radii" in state and state["radii"] is not None:
                is_split = is_split | (state["radii"] > self.grow_scale2d)
        n_split = is_split.sum().item()

        if n_dupli > 0:
            duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)

        is_split = torch.cat(
            [is_split, torch.zeros(n_dupli, dtype=torch.bool, device=device)]
        )

        if n_split > 0:
            split(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_split,
                revised_opacity=self.revised_opacity,
            )
        return n_dupli, n_split


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


_AUX_BUFFERS = [
    "gaussian_birth_iter",
    "segmentation_labels",
    "segmentation_colors",
    "segmentation_color_logits",
    "semantic_confidence",
    "semantic_alt_class",
    "semantic_alt_confidence",
    "semantic_switch_support",
    "semantic_switch_cooldown",
    "gaussian_instance_ids",
]


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
        self.params = None
        self.optimizers = None
        self.iteration_count = 0

        # Loss and optimization knobs
        gs = self.tracker.config.gaussian_splatting
        self.points_lr_mult = float(gs.points_lr_mult)
        self.depth_loss_weight = float(gs.depth_loss_weight)

        # Token-bucket limiter: smooth "max_steps_per_sec"
        self.step_limiter = TokenBucket(rate=max_steps_per_sec, burst=100.0)

        # Densification and pruning with absolute split threshold for SLAM.
        self.strategy = AbsoluteScaleStrategy(
            split_scale_abs=0.01,
            prune_opa=0.02,
            grow_grad2d=0.0002,
            grow_scale3d=0.01,
            grow_scale2d=0.05,
            prune_scale3d=0.1,
            prune_scale2d=0.15,
            refine_start_iter=100,
            refine_stop_iter=10_000_000,
            reset_every=3000,
            refine_every=100,
            absgrad=False,
            verbose=True,
        )
        self.strategy_state = None

        # Culling params (for covisibility and training)
        self.cull_near = 0.05
        self.cull_far = 50.0
        self.cull_pad_px = 4.0
        self.min_culled_points = 2048
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

        with self.pcd.lock:
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
    def render_reliable_depth_at_pose(
        self,
        pose_np: np.ndarray,
        max_scale: float = 0.05,
        min_opacity: float = 0.3,
        min_maturity: int = 2,
    ):
        if self.pcd.all_points is None or pose_np is None:
            return None

        with self.pcd.lock:
            means = self.pcd.all_points
            quats = self.pcd.all_quaternions
            scales = self.pcd.all_scales
            alpha = self.pcd.all_alpha
            birth = self.pcd.gaussian_birth_iter
            birth_counter = self.pcd._birth_counter

        if means is None or means.shape[0] == 0:
            return None

        opacities = torch.sigmoid(alpha).squeeze(-1)
        world_scales = torch.exp(scales).max(dim=-1).values

        reliable = (
            (world_scales <= max_scale)
            & (opacities >= min_opacity)
        )
        if birth is not None:
            reliable = reliable & ((birth_counter - birth) >= min_maturity)

        idx = torch.where(reliable)[0]
        if idx.shape[0] == 0:
            return None

        means = means[idx]
        quats = quats[idx]
        scales = scales[idx]
        opacities = opacities[idx]

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
                opacities=opacities,
                colors=ones,
                viewmats=viewmat,
                Ks=K,
                width=self.depth_width,
                height=self.depth_height,
                render_mode="ED",
            )
        except Exception:
            return None

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

        with self.pcd.lock:
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

        with self.pcd.lock:
            means = self.pcd.all_points
            quats = self.pcd.all_quaternions
            scales = self.pcd.all_scales
            alpha = self.pcd.all_alpha
            seg_colors = getattr(self.pcd, "segmentation_colors", None)

        if means is None or means.shape[0] == 0:
            return None

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

    def _compute_scene_scale(self):
        means = self.pcd.all_points
        if means is None or means.shape[0] < 2:
            return 1.0
        with torch.no_grad():
            mn = means.min(dim=0).values
            mx = means.max(dim=0).values
            return max(float((mx - mn).norm().item()), 1e-6)

    def _setup_optimizer(self):
        if self.pcd.all_points is None:
            return

        old_params = self.params
        old_optimizers = self.optimizers
        old_strategy_state = self.strategy_state

        # Snapshot pcd under the lock so _merge_data cannot interleave and
        # cause size mismatches between the cloned tensors.
        with self.pcd.lock:
            alpha = self.pcd.all_alpha
            if alpha.ndim == 2 and alpha.shape[-1] == 1:
                alpha = alpha.squeeze(-1)

            self.params = {
                "means": torch.nn.Parameter(self.pcd.all_points.detach().clone()),
                "sh": torch.nn.Parameter(self.pcd.all_sh.detach().clone()),
                "scales": torch.nn.Parameter(self.pcd.all_scales.detach().clone()),
                "quats": torch.nn.Parameter(self.pcd.all_quaternions.detach().clone()),
                "opacities": torch.nn.Parameter(alpha.detach().clone()),
            }
            n_snap = self.params["means"].shape[0]

            # Snapshot aux buffers while still under the lock.
            aux_snaps = {}
            for buf_name in _AUX_BUFFERS:
                buf = getattr(self.pcd, buf_name, None)
                if buf is not None and torch.is_tensor(buf) and buf.shape[0] == n_snap:
                    aux_snaps[buf_name] = buf.detach().clone()

        lr_map = {
            "means": self.base_lr * self.points_lr_mult,
            "sh": self.base_lr * 3.0,
            "scales": self.base_lr * 2.0,
            "quats": self.base_lr * 2.0,
            "opacities": self.base_lr,
        }
        self.optimizers = {
            name: torch.optim.Adam([self.params[name]], lr=lr)
            for name, lr in lr_map.items()
        }

        # Carry Adam momentum for pre-existing gaussians across map growth
        if old_optimizers is not None and old_params is not None:
            for name in lr_map:
                if name not in old_optimizers or name not in old_params:
                    continue
                old_opt = old_optimizers[name]
                old_p = old_params[name]
                old_state = old_opt.state.get(old_p)
                if old_state is None:
                    continue

                new_opt = self.optimizers[name]
                new_p = self.params[name]
                new_state = new_opt.state.setdefault(new_p, {})

                if "step" in old_state:
                    new_state["step"] = old_state["step"]

                for key in ("exp_avg", "exp_avg_sq"):
                    old_t = old_state.get(key)
                    if old_t is None or not torch.is_tensor(old_t):
                        continue
                    new_t = torch.zeros_like(new_p.data)
                    n_copy = min(old_t.shape[0], new_t.shape[0])
                    if n_copy > 0:
                        new_t[:n_copy].copy_(old_t[:n_copy])
                    new_state[key] = new_t

        # Extend strategy state instead of resetting it
        scene_scale = self._compute_scene_scale()
        new_n = self.params["means"].shape[0]

        if old_strategy_state is not None and old_strategy_state.get("grad2d") is not None:
            self.strategy_state = {"scene_scale": scene_scale}
            for key in ("grad2d", "count", "radii"):
                old_t = old_strategy_state.get(key)
                if old_t is None or not torch.is_tensor(old_t):
                    if key == "radii":
                        continue
                    self.strategy_state[key] = torch.zeros(new_n, device=self.device)
                    continue
                new_t = torch.zeros(new_n, device=self.device)
                n_copy = min(old_t.shape[0], new_n)
                if n_copy > 0:
                    new_t[:n_copy].copy_(old_t[:n_copy])
                self.strategy_state[key] = new_t
        else:
            self.strategy_state = self.strategy.initialize_state(scene_scale=scene_scale)

        # Put auxiliary pcd buffers in strategy_state so the strategy ops
        # (split/duplicate/remove) keep them aligned with the gaussians.
        for buf_name, buf_t in aux_snaps.items():
            self.strategy_state[buf_name] = buf_t

        self.num_points_optimized = new_n
        self._sync_pcd_from_params()

    def _sync_pcd_from_params(self):
        """Write params (and aux buffers from strategy_state) back to pcd.

        Holds pcd.lock so that background threads (segmenter, scene-graph)
        never see half-updated attribute sizes.
        """
        if self.params is None:
            return
        with self.pcd.lock:
            self.pcd.all_points = self.params["means"]
            self.pcd.all_sh = self.params["sh"]
            self.pcd.all_scales = self.params["scales"]
            self.pcd.all_quaternions = self.params["quats"]
            opacities = self.params["opacities"]
            self.pcd.all_alpha = opacities.unsqueeze(-1) if opacities.ndim == 1 else opacities
            if self.strategy_state is not None:
                for buf_name in _AUX_BUFFERS:
                    if buf_name in self.strategy_state:
                        v = self.strategy_state[buf_name]
                        if isinstance(v, torch.Tensor):
                            setattr(self.pcd, buf_name, v)

    def training_step(self):
        if not self.step_limiter.allow(cost=1.0):
            return 0.0

        if self.pcd.all_points is None or not self.tracker.keyframes_poses:
            return 0.0

        self.iteration_count += 1

        current_count = self.pcd.all_points.shape[0]
        if self.params is None or current_count != self.num_points_optimized:
            self._setup_optimizer()
            if self.params is None:
                return 0.0

        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=True)

        means = self.params["means"]
        sh = self.params["sh"]
        scales = self.params["scales"]
        quats = self.params["quats"]
        opacities = self.params["opacities"]
        sh_degree = self.pcd.sh_degree
        R_fix = self.pcd.R_fix

        kf_count = len(self.tracker.keyframes_poses)

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

        # Frustum culling (keeps gradients flowing to full params via indexing)
        with torch.no_grad():
            cull_mask = frustum_cull_mask(
                means_world=means.detach(),
                viewmats=viewmats.detach(),
                Ks=Ks.detach(),
                width=self.train_width,
                height=self.train_height,
                near=self.cull_near,
                far=self.cull_far,
                pad=self.cull_pad_px,
            )
            if cull_mask.sum() < self.min_culled_points:
                cull_mask = torch.ones_like(cull_mask, dtype=torch.bool)

        idx = torch.where(cull_mask)[0]
        means_c = means[idx]
        sh_c = sh[idx]
        scales_c = scales[idx]
        quats_c = quats[idx]
        opacities_c = opacities[idx]

        dirs = means_c.unsqueeze(0) - cam_centers.unsqueeze(1)
        dirs = F.normalize(dirs, dim=-1)
        sh_coeffs = sh_c.unsqueeze(0).expand(b, -1, -1, -1)
        colors = torch.sigmoid(spherical_harmonics(sh_degree, dirs, sh_coeffs))

        rendered, _, info = rendering.rasterization(
            means=means_c,
            quats=F.normalize(quats_c, p=2, dim=-1),
            scales=torch.exp(scales_c),
            opacities=torch.sigmoid(opacities_c),
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=self.train_width,
            height=self.train_height,
            render_mode="RGB+ED",
        )

        # Remap packed gaussian_ids from culled-local to global indices
        # so the strategy accumulates stats for the correct gaussians
        if "gaussian_ids" in info and info["gaussian_ids"] is not None:
            info["gaussian_ids"] = idx[info["gaussian_ids"]]

        self.strategy.step_pre_backward(
            self.params, self.optimizers, self.strategy_state,
            self.iteration_count, info,
        )

        rendered_rgb = rendered[..., :3]
        rendered_depth = rendered[..., 3] if rendered.shape[-1] > 3 else None

        if gt_depths is not None:
            valid = (gt_depths > 0.0) & torch.isfinite(gt_depths)
            rgb_mask = valid.unsqueeze(-1).expand_as(rendered_rgb)
            if rgb_mask.any():
                rgb_loss = F.l1_loss(rendered_rgb[rgb_mask], gt_rgbs[rgb_mask])
            else:
                rgb_loss = F.l1_loss(rendered_rgb, gt_rgbs)

            depth_loss = torch.zeros((), device=self.device)
            if rendered_depth is not None:
                dvalid = valid & torch.isfinite(rendered_depth) & (rendered_depth > 0.0)
                if dvalid.any():
                    rd = rendered_depth[dvalid]
                    gd = gt_depths[dvalid]
                    per_pixel = F.l1_loss(rd, gd, reduction='none')
                    w = 1.0 / (gd * gd).clamp(min=0.01)
                    depth_loss = (per_pixel * w).sum() / w.sum()
        else:
            rgb_loss = F.l1_loss(rendered_rgb, gt_rgbs)
            depth_loss = torch.zeros((), device=self.device)

        total_loss = rgb_loss + self.depth_loss_weight * depth_loss

        if total_loss > 0:
            total_loss.backward()

            # Track whether the strategy replaces params (densify/prune).
            # If it does, we must sync new params back to pcd.
            # If it doesn't, the optimizer updates params in-place and
            # pcd already sees the changes (pcd attrs ARE the param objects).
            means_before = self.params["means"]

            self.strategy.step_post_backward(
                self.params, self.optimizers, self.strategy_state,
                self.iteration_count, info, packed=True,
            )

            for opt in self.optimizers.values():
                opt.step()

            if means_before is not self.params["means"]:
                self._sync_pcd_from_params()
                self.num_points_optimized = self.params["means"].shape[0]
                self.pcd.rebuild_seen_keys()

            now = time.time()
            self._step_timestamps.append(now)
            cutoff = now - 1.0
            while self._step_timestamps and self._step_timestamps[0] < cutoff:
                self._step_timestamps.popleft()
            self.steps_per_sec = float(len(self._step_timestamps))

        return float(total_loss.detach().item())