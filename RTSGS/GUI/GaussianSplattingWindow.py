import numpy as np
import torch
from gsplat import rendering, spherical_harmonics
from imgui_bundle import imgui

from RTSGS.GaussianSplatting.PointCloud import PointCloud
from RTSGS.GaussianSplatting.Renderer.Camera import Camera
from RTSGS.GUI.ImageWidget import ImageWidget

@torch.no_grad()
def _fill_viewmat_and_camcenter_from_view_np(
    view_np: np.ndarray,
    device: torch.device,
    out_viewmat: torch.Tensor,
    flip: torch.Tensor,
) -> torch.Tensor:
    """
    Returns cam_center (3,) torch on device and fills out_viewmat (4,4) in-place.
    """
    V = np.asarray(view_np, dtype=np.float32)  # view_col_major
    R_np = V[:3, :3]  # (3,3)
    t_np = V[3, :3]   # (3,)

    # Convert (still incurs CPU->GPU copy on cuda)
    R = torch.from_numpy(R_np).to(device=device, non_blocking=True)
    t = torch.from_numpy(t_np).to(device=device, non_blocking=True)

    # out_viewmat = flip @ V, where V built from R_row,t_row with R_row.t()
    out_viewmat.zero_()
    out_viewmat[3, 3] = 1.0
    out_viewmat[:3, :3].copy_(R.t())
    out_viewmat[:3, 3].copy_(t)
    out_viewmat.copy_(flip @ out_viewmat)

    # cam_center = -R^T * t (R is row-rotation here)
    cam_center = -(R.t() @ t)
    return cam_center


@torch.no_grad()
def _normalize_rows_inplace(x: torch.Tensor) -> torch.Tensor:
    # x: (N,3) -> x /= ||x|| with rsqrt; clamp to avoid inf
    inv = torch.rsqrt((x * x).sum(dim=-1, keepdim=True).clamp_min_(1e-12))
    x.mul_(inv)
    return x


@torch.no_grad()
def _sigmoid_inplace(x: torch.Tensor) -> torch.Tensor:
    # sigmoid(x) in-place: 1 / (1 + exp(-x))
    x.neg_().exp_().add_(1.0).reciprocal_()
    return x


MODE_FREE_ROAMING = 0
MODE_LAST_POSE = 1
MODE_EXTRAPOLATED = 2
_MODE_LABELS = ["Free Roaming", "Last Pose", "Extrapolated Pose"]


class GaussianSplattingWindow:
    def __init__(self, pcd: PointCloud, camera: Camera, title: str = "GSplat Renderer", tracker=None):
        self.pcd = pcd
        self.camera = camera
        self.title = title
        self.tracker = tracker
        self.camera_mode = MODE_FREE_ROAMING

        self.device = getattr(pcd, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        self.image_widget = ImageWidget(np.zeros((camera.height, camera.width, 3), dtype=np.uint8))
        self.is_open = True

        # GUI render path should use RGB intrinsics.
        fx_src = getattr(pcd, "rgb_fx", pcd.fx)
        fy_src = getattr(pcd, "rgb_fy", pcd.fy)
        self.fx = float(fx_src.item()) if torch.is_tensor(fx_src) else float(fx_src)
        self.fy = float(fy_src.item()) if torch.is_tensor(fy_src) else float(fy_src)
        self.default_scale = 0.01

        # cached tiny tensors
        self._flip = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], device=self.device, dtype=torch.float32))
        self._viewmat = torch.empty((4, 4), device=self.device, dtype=torch.float32)

        self._K = None
        self._K_w = -1
        self._K_h = -1

        self._default_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device, dtype=torch.float32)
        self._default_opacity = torch.tensor(1.0, device=self.device, dtype=torch.float32)

        self._T_fix = torch.eye(4, device=self.device, dtype=torch.float32)
        self._T_fix[:3, :3] = pcd.R_fix

        self._rgb_fx = float(getattr(pcd, "rgb_fx", self.fx))
        self._rgb_fy = float(getattr(pcd, "rgb_fy", self.fy))
        self._rgb_cx = float(getattr(pcd, "rgb_cx", 0))
        self._rgb_cy = float(getattr(pcd, "rgb_cy", 0))

        # references to latest pcd buffers
        self._xyz = None
        self._sh = None
        self._quats = None
        self._scales = None
        self._alpha = None

        # scratch buffers (reused if N matches)
        self._dirs = None        # (N,3)
        self._colors = None      # (N,3) only used when no SH
        self._ones_rgb = None    # cached ones (N,3) when no SH
        self._s_default = None   # (N,3) default scales
        self._o_default = None   # (N,)  default opacities
        self._q_default = None   # (N,4) default quats

        self._last_N = -1

    def _ensure_intrinsics(self):
        w, h = self.camera.width, self.camera.height
        if self._K is None or w != self._K_w or h != self._K_h:
            self._K = torch.tensor(
                [[self.fx, 0.0, w * 0.5], [0.0, self.fy, h * 0.5], [0.0, 0.0, 1.0]],
                device=self.device,
                dtype=torch.float32,
            ).unsqueeze(0)
            native_w = self._rgb_cx * 2.0
            native_h = self._rgb_cy * 2.0
            sx = w / native_w if native_w > 0 else 1.0
            sy = h / native_h if native_h > 0 else 1.0
            self._tracker_K = torch.tensor(
                [[self._rgb_fx * sx, 0.0, self._rgb_cx * sx],
                 [0.0, self._rgb_fy * sy, self._rgb_cy * sy],
                 [0.0, 0.0, 1.0]],
                device=self.device, dtype=torch.float32,
            ).unsqueeze(0)
            self._K_w, self._K_h = w, h

    def _ensure_scratch(self, N: int, like: torch.Tensor):
        if N == self._last_N and self._dirs is not None:
            return

        # Allocate once per N change
        self._dirs = torch.empty((N, 3), device=like.device, dtype=torch.float32)
        self._ones_rgb = torch.ones((N, 3), device=like.device, dtype=torch.float32)

        # Defaults (avoid creating every frame)
        self._q_default = self._default_quat.expand(N, 4)
        self._s_default = torch.full((N, 3), self.default_scale, device=like.device, dtype=torch.float32)
        self._o_default = self._default_opacity.expand(N)

        self._last_N = N

    def _get_tracker_pose(self):
        if self.tracker is None:
            return None
        poses = self.tracker.poses
        if not poses:
            return None
        if self.camera_mode == MODE_LAST_POSE:
            return poses[-1]
        if self.camera_mode == MODE_EXTRAPOLATED:
            return self._pull_back_pose(poses[-1])
        return None

    @torch.no_grad()
    def _viewmat_from_tracker_pose(self, pose_np: np.ndarray):
        pose = torch.from_numpy(np.asarray(pose_np, dtype=np.float32)).to(self.device)
        world_cam = self._T_fix @ pose
        viewmat = torch.inverse(world_cam)
        cam_center = world_cam[:3, 3]
        return viewmat, cam_center

    @staticmethod
    def _pull_back_pose(pose_np: np.ndarray, distance: float = 0.5) -> np.ndarray:
        pose = pose_np.astype(np.float64).copy()
        forward = pose[:3, 2]
        pose[:3, 3] -= forward * distance
        return pose.astype(np.float32)

    def _pull_latest_buffers(self) -> bool:
        p = self.pcd
        # Snapshot all attributes under the lock so background threads
        # (_merge_data, _sync_pcd_from_params) cannot change sizes mid-read.
        with p.lock:
            xyz = getattr(p, "all_points", None)
            if xyz is None or xyz.numel() == 0:
                return False

            xyz = xyz.detach()
            sh = getattr(p, "all_sh", None)
            sh = sh.detach() if sh is not None else None

            q = getattr(p, "all_quaternions", None)
            q = q.detach() if q is not None else None

            s = getattr(p, "all_scales", None)
            s = s.detach() if s is not None else None

            a = getattr(p, "all_alpha", None)
            a = a.detach() if a is not None else None

            seg_filter = getattr(p, "segmentation_class_filter", None)
            labels = getattr(p, "segmentation_labels", None) if seg_filter is not None else None

        if seg_filter is not None and labels is not None and int(labels.shape[0]) == int(xyz.shape[0]):
            labels = labels.detach()
            if isinstance(seg_filter, (list, tuple, set)):
                keep = torch.zeros_like(labels, dtype=torch.bool)
                for cls_id in seg_filter:
                    try:
                        keep |= (labels == int(cls_id))
                    except Exception:
                        continue
            else:
                try:
                    keep = labels == int(seg_filter)
                except Exception:
                    keep = None

            if keep is not None:
                xyz = xyz[keep]
                if sh is not None and int(sh.shape[0]) == int(keep.shape[0]):
                    sh = sh[keep]
                if q is not None and int(q.shape[0]) == int(keep.shape[0]):
                    q = q[keep]
                if s is not None and int(s.shape[0]) == int(keep.shape[0]):
                    s = s[keep]
                if a is not None and int(a.shape[0]) == int(keep.shape[0]):
                    a = a[keep]

        self._xyz = xyz
        self._sh = sh
        self._quats = q
        self._scales = s
        self._alpha = a
        return True

    @torch.no_grad()
    def _render_with_gsplat(self) -> np.ndarray:
        self._ensure_intrinsics()

        xyz = self._xyz
        N = xyz.shape[0]
        self._ensure_scratch(N, xyz)

        tracker_pose = self._get_tracker_pose()
        if tracker_pose is not None and self.camera_mode != MODE_FREE_ROAMING:
            viewmat, cam_center = self._viewmat_from_tracker_pose(tracker_pose)
            self._viewmat.copy_(viewmat)
        else:
            self.camera.update_view()
            cam_center = _fill_viewmat_and_camcenter_from_view_np(
                self.camera.view, self.device, self._viewmat, self._flip
            )

        # Colors
        sh = self._sh
        if sh is not None:
            # dirs = normalize(xyz - cam_center) using scratch buffer
            self._dirs.copy_(xyz)
            self._dirs.sub_(cam_center)
            _normalize_rows_inplace(self._dirs)

            c = spherical_harmonics(self.pcd.sh_degree, self._dirs, sh)
            c = _sigmoid_inplace(c)
            c = c.contiguous()
        else:
            c = self._ones_rgb

        # Quats
        q = self._quats
        if q is not None:
            # normalize without extra alloc
            inv = torch.rsqrt((q * q).sum(dim=-1, keepdim=True).clamp_min_(1e-12))
            q = q * inv
        else:
            q = self._q_default

        # Scales
        sc = self._scales
        if sc is not None:
            s = torch.exp(sc)
        else:
            s = self._s_default

        # Opacity
        al = self._alpha
        if al is not None:
            o = torch.sigmoid(al).squeeze(-1)
        else:
            o = self._o_default

        use_tracker = tracker_pose is not None and self.camera_mode != MODE_FREE_ROAMING
        img, _, _ = rendering.rasterization(
            means=xyz,
            quats=q,
            scales=s,
            opacities=o,
            colors=c,
            viewmats=self._viewmat.unsqueeze(0),
            Ks=self._tracker_K if use_tracker else self._K,
            width=self.camera.width,
            height=self.camera.height,
            render_mode="RGB",
        )

        return (img[0].clamp_(0.0, 1.0).mul_(255.0)).to(torch.uint8).cpu().numpy()

    def draw(self, delta_time: float):
        if not self.is_open:
            return

        expanded, self.is_open = imgui.begin(self.title, self.is_open)
        if not expanded:
            imgui.end()
            return

        imgui.set_next_item_width(180)
        changed, new_mode = imgui.combo("Camera Mode", self.camera_mode, _MODE_LABELS)
        if changed:
            self.camera_mode = new_mode

        avail = imgui.get_content_region_avail()
        w, h = int(avail.x), int(avail.y)
        if (w != self.camera.width) or (h != self.camera.height):
            if w > 0 and h > 0:
                self.camera.update_resolution(w, h)

        if self.camera_mode == MODE_FREE_ROAMING:
            self.camera.process_window_input(imgui.is_window_hovered(), imgui.is_window_focused(), delta_time)

        if self._pull_latest_buffers():
            rgb = self._render_with_gsplat()
            self.image_widget.set_image_rgb(rgb)
            self.image_widget.draw(fit_to_window=True)
        else:
            imgui.text("Waiting for PointCloud data...")

        imgui.end()