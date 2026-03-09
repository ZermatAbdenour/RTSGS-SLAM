from RTSGS.DataLoader.DataLoader import DataLoader
from RTSGS.Tracker.Tracker import Tracker
import numpy as np
import cv2
import torch

from RTSGS.GUI.ImageWidget import ImageWidget
from imgui_bundle import imgui, implot


class ICPORBTracker(Tracker):
    """
    ORB-based tracker with GPU ICP refinement implemented in PyTorch.
    Projective point-to-point ICP aligns the previous depth to the current depth
    using intrinsics and the PnP estimate as the initial transform.

    Returns the global pose of the current frame (4x4).
    """

    def __init__(self, dataset: DataLoader, config, Orb_features=3000):
        super().__init__()
        self.dataset = dataset

        # intrinsics
        self.fx, self.fy = config.get("fx"), config.get("fy")
        self.cx, self.cy = config.get("cx"), config.get("cy")
        self.K = config.get_camera_intrinsics()

        self.depth_scale = config.get("depth_scale")
        self.orb = cv2.ORB_create(nfeatures=Orb_features)

        # BF matcher for ratio test KNN
        self._bf_knn = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Global poses history (same initial pose as your SimpleORBTracker)
        self.poses = [dataset.gt_poses[0]]

        # keyframe thresholds
        self.alpha = config.get("kf_translation", 0.05)
        self.theta = config.get("kf_rotation", 5.0 * np.pi / 180.0)

        self.last_kf_pose = None

        # Previous frame caches
        self.prev_rgb = None
        self.prev_depth_m = None  # depth in meters

        # Visualization
        self.viz_img = None
        self.img_window = None

        self.show_matching_window = True
        self.show_comparison_window = False

        # PyTorch device
        cuda_index = int(config.get("cuda_device", 0))
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{cuda_index}")
        else:
            self.device = torch.device("cpu")

    def track_frame(self, rgb, depth=None, use_grayscale: bool = False):
        """
        ORB pose via PnP, then GPU ICP refinement (PyTorch) on RGB-D.
        Returns global pose of current frame (4x4) or None.
        """
        # Choose consistent representation for ORB
        if use_grayscale and rgb is not None and rgb.ndim == 3:
            cur_img = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        else:
            cur_img = rgb

        # Need at least one previous frame with depth
        if self.prev_rgb is None or self.prev_depth_m is None:
            self.prev_rgb = cur_img
            if depth is not None:
                self.prev_depth_m = depth.astype(np.float32) / float(self.depth_scale)
            # Seed frame 0 as the first keyframe so points enter the map immediately
            if not self.keyframes_poses:
                init_pose = self.poses[0].astype(np.float32)
                if self.dataset is not None:
                    self.dataset.rgb_keyframes.append(rgb)
                    if depth is not None:
                        self.dataset.depth_keyframes.append(depth)
                    self.keyframes_poses.append(init_pose)
                    self.last_kf_pose = init_pose
                    self.dataset.current_keyframe_index += 1
            return None

        prev_img = self.prev_rgb

        if depth is None:
            # No ICP without current depth
            self.prev_rgb = cur_img
            return None

        # depth in meters
        depth_m = depth.astype(np.float32) / float(self.depth_scale)
        h, w = depth_m.shape[:2]

        # --- ORB detect/compute ---
        kp1, des1 = self.orb.detectAndCompute(prev_img, None)
        kp2, des2 = self.orb.detectAndCompute(cur_img, None)

        if des1 is None or des2 is None or len(kp1) < 12 or len(kp2) < 12:
            self.prev_rgb = cur_img
            self.prev_depth_m = depth_m
            return None

        # --- KNN ratio matching ---
        knn = self._bf_knn.knnMatch(des1, des2, k=2)
        ratio = 0.75
        good = []
        for m_n in knn:
            if len(m_n) < 2:
                continue
            m, n = m_n
            if m.distance < ratio * n.distance:
                good.append(m)

        if len(good) < 12:
            self.prev_rgb = cur_img
            self.prev_depth_m = depth_m
            return None

        good.sort(key=lambda x: x.distance)
        max_corr = 1500
        good = good[:max_corr]

        # --- build arrays ---
        qidx = np.fromiter((m.queryIdx for m in good), dtype=np.int32, count=len(good))
        tidx = np.fromiter((m.trainIdx for m in good), dtype=np.int32, count=len(good))

        pts1 = np.array([kp1[i].pt for i in qidx], dtype=np.float32)  # prev image coords
        pts2 = np.array([kp2[i].pt for i in tidx], dtype=np.float32)  # current image coords

        ui = np.clip(pts1[:, 0].astype(np.int32), 0, w - 1)
        vi = np.clip(pts1[:, 1].astype(np.int32), 0, h - 1)

        z = self.prev_depth_m[vi, ui]
        valid = (z > 0.0) & np.isfinite(z)

        if valid.sum() < 12:
            self.prev_rgb = cur_img
            self.prev_depth_m = depth_m
            return None

        pts1v = pts1[valid]
        pts2v = pts2[valid]
        zv = z[valid]

        X = (pts1v[:, 0] - self.cx) * zv / self.fx
        Y = (pts1v[:, 1] - self.cy) * zv / self.fy
        pts3d = np.stack([X, Y, zv], axis=1).astype(np.float32)  # prev camera coords
        pts2d = pts2v.astype(np.float32)

        if pts3d.shape[0] < 12:
            self.prev_rgb = cur_img
            self.prev_depth_m = depth_m
            return None

        # --- PnP (prev -> current) ---
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d,
            pts2d,
            self.K,
            None,
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=0.1,
            confidence=0.999,
            iterationsCount=1000
        )
        if not ok or inliers is None or len(inliers) < 8:
            self.prev_rgb = cur_img
            self.prev_depth_m = depth_m
            return None

        inl = inliers.reshape(-1)
        ok2, rvec, tvec = cv2.solvePnP(
            pts3d[inl],
            pts2d[inl],
            self.K,
            None,
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok2:
            self.prev_rgb = cur_img
            self.prev_depth_m = depth_m
            return None

        R, _ = cv2.Rodrigues(rvec)
        T_pnp = np.eye(4, dtype=np.float32)
        T_pnp[:3, :3] = R.astype(np.float32)
        T_pnp[:3, 3] = tvec.reshape(3).astype(np.float32)
        # T_pnp maps previous camera coords -> current camera coords

        # --- PyTorch GPU ICP refinement (projective, point-to-point) ---
        T_icp = self._gpu_icp_refine_torch(self.prev_depth_m, depth_m, T_pnp)

        # If ICP failed, fallback to PnP
        T_final = T_icp if T_icp is not None else T_pnp

        # Compose global pose (preserving your original convention)
        pose = self.poses[-1] @ np.linalg.inv(T_final)
        self.poses.append(pose.astype(np.float32))

        # Keyframe logic
        is_keyframe = False
        if self.last_kf_pose is None:
            is_keyframe = True
        else:
            dt, dR = self._pose_distance(self.last_kf_pose, pose)
            if dt > self.alpha or dR > self.theta:
                is_keyframe = True

        if is_keyframe and self.dataset is not None:
            self.dataset.rgb_keyframes.append(rgb)
            if depth is not None:
                self.dataset.depth_keyframes.append(depth)
            self.keyframes_poses.append(pose.astype(np.float32))
            self.last_kf_pose = pose
            self.dataset.current_keyframe_index += 1

        # Visualization of matches
        if self.show_matching_window:
            draw_n = min(80, len(good))
            self.viz_img = cv2.drawMatches(
                prev_img, kp1, cur_img, kp2, good[:draw_n], None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

        # Update previous caches
        self.prev_rgb = cur_img
        self.prev_depth_m = depth_m
        return pose

    def _gpu_icp_refine_torch(self, prev_depth_m, cur_depth_m, T_init_np,
                              stride: int = 4,
                              depth_min: float = 0.05,
                              depth_max: float = 6.0,
                              max_iters: int = 10,
                              eps: float = 0.005,
                              min_pairs: int = 500):
        """
        Projective ICP in PyTorch (GPU-capable).
        - Build source points from previous depth.
        - Iteratively:
            - Transform to current coords with T_k
            - Project to current image, sample target depth
            - Backproject target points
            - Estimate rigid T_delta (Kabsch) aligning src->tgt
            - Update T_k = T_delta @ T_k
        Returns refined 4x4 transform (prev->current) or None.
        """
        h, w = prev_depth_m.shape[:2]

        # Torch tensors
        prev_d = torch.from_numpy(prev_depth_m).to(self.device)
        cur_d = torch.from_numpy(cur_depth_m).to(self.device)

        # Sample grid
        v = torch.arange(0, h, stride, device=self.device)
        u = torch.arange(0, w, stride, device=self.device)
        uu, vv = torch.meshgrid(u, v, indexing="xy")
        z = prev_d[vv, uu]
        valid = (z > depth_min) & (z < depth_max) & torch.isfinite(z)

        if valid.sum().item() < min_pairs:
            # If too few points, relax constraints slightly
            valid = (z > 0.0) & torch.isfinite(z)

        uu = uu[valid].float()
        vv = vv[valid].float()
        z = z[valid].float()
        if uu.numel() < 128:
            return None

        # Backproject source (prev camera coords)
        X = (uu - float(self.cx)) * z / float(self.fx)
        Y = (vv - float(self.cy)) * z / float(self.fy)
        source_prev = torch.stack([X, Y, z], dim=1)  # [N,3]

        # Initialize T_k
        T_k = torch.from_numpy(T_init_np).to(self.device).float()

        def transform_points(T, pts):
            R = T[:3, :3]
            t = T[:3, 3]
            return (pts @ R.T) + t

        def project_points(pts):
            # pts in current camera coords
            Xc, Yc, Zc = pts[:, 0], pts[:, 1], pts[:, 2]
            valid_z = Zc > 1e-6
            u_proj = float(self.fx) * (Xc / Zc) + float(self.cx)
            v_proj = float(self.fy) * (Yc / Zc) + float(self.cy)
            return u_proj, v_proj, valid_z

        residual_prev = None

        for _ in range(max_iters):
            # Transform source to current coords
            src_cur = transform_points(T_k, source_prev)  # [N,3]

            # Project to current image
            u_proj, v_proj, valid_z = project_points(src_cur)

            # Nearest-neighbor pixel sampling
            ui = torch.round(u_proj).long()
            vi = torch.round(v_proj).long()

            # In-bounds mask
            in_bounds = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
            mask = valid_z & in_bounds
            if mask.sum().item() < min_pairs:
                return None

            ui = ui[mask]
            vi = vi[mask]
            src_cur_v = src_cur[mask]

            # Sample target depth
            z2 = cur_d[vi, ui]
            valid2 = (z2 > depth_min) & (z2 < depth_max) & torch.isfinite(z2)
            if valid2.sum().item() < min_pairs:
                return None

            ui = ui[valid2].float()
            vi = vi[valid2].float()
            src_cur_v = src_cur_v[valid2]
            z2 = z2[valid2].float()

            # Backproject target points in current camera coords
            X2 = (ui - float(self.cx)) * z2 / float(self.fx)
            Y2 = (vi - float(self.cy)) * z2 / float(self.fy)
            tgt_cur = torch.stack([X2, Y2, z2], dim=1)  # [M,3]

            # Estimate rigid transform aligning src_cur_v -> tgt_cur
            T_delta = self._rigid_transform_kabsch_torch(src_cur_v, tgt_cur)
            if T_delta is None:
                return None

            # Update T_k
            T_k = T_delta @ T_k

            # Check residual for convergence
            src_cur_upd = transform_points(T_delta, src_cur_v)
            res = torch.linalg.norm(src_cur_upd - tgt_cur, dim=1).mean()
            if residual_prev is not None and torch.abs(residual_prev - res) < eps:
                break
            residual_prev = res

        # Return refined transform
        return T_k.detach().cpu().numpy().astype(np.float32)

    @staticmethod
    def _rigid_transform_kabsch_torch(A: torch.Tensor, B: torch.Tensor):
        """
        Compute rigid transform (SE3) aligning A -> B with Kabsch (point-to-point).
        A, B: [N,3] on same device.
        Returns 4x4 transform T or None.
        """
        if A.shape[0] < 3 or B.shape[0] < 3:
            return None

        CA = A.mean(dim=0)
        CB = B.mean(dim=0)
        AA = A - CA
        BB = B - CB

        H = AA.T @ BB
        U, S, Vh = torch.linalg.svd(H)
        R = Vh.T @ U.T

        # Handle reflection
        if torch.linalg.det(R) < 0:
            Vh[:, -1] *= -1
            R = Vh.T @ U.T

        t = CB - R @ CA

        T = torch.eye(4, device=A.device, dtype=A.dtype)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def _get_pred_xyz(self):
        if len(self.poses) == 0:
            return None
        traj = np.array([p[:3, 3] for p in self.poses], dtype=np.float32)
        return np.ascontiguousarray(traj, dtype=np.float32)

    def _get_gt_xyz(self):
        if self.dataset is None or self.dataset.gt_poses is None or len(self.dataset.gt_poses) == 0:
            return None
        traj = np.array([p[:3, 3] for p in self.dataset.gt_poses], dtype=np.float32)
        return np.ascontiguousarray(traj, dtype=np.float32)

    @staticmethod
    def _padded_limits_from_two(a, b, pad_ratio=0.05):
        amin = float(min(np.min(a), np.min(b)))
        amax = float(max(np.max(a), np.max(b)))
        pad = 1.0 if amin == amax else (amax - amin) * pad_ratio
        return amin - pad, amax + pad

    @staticmethod
    def _umeyama_align(src, dst, with_scale=True):
        src = np.asarray(src, dtype=np.float64)
        dst = np.asarray(dst, dtype=np.float64)
        n = src.shape[0]
        mu_src = src.mean(axis=0)
        mu_dst = dst.mean(axis=0)

        X = src - mu_src
        Y = dst - mu_dst

        cov = (Y.T @ X) / n
        U, D, Vt = np.linalg.svd(cov)

        S = np.eye(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S[2, 2] = -1.0

        R = U @ S @ Vt

        if with_scale:
            var_src = (X ** 2).sum() / n
            scale = (D * np.diag(S)).sum() / var_src
        else:
            scale = 1.0

        t = mu_dst - scale * (R @ mu_src)

        aligned = (scale * (R @ src.T)).T + t
        return aligned.astype(np.float32), (float(scale), R.astype(np.float32), t.astype(np.float32))

    def _pose_distance(self, T1, T2):
        # translation
        t1 = T1[:3, 3]
        t2 = T2[:3, 3]
        trans_dist = np.linalg.norm(t1 - t2)
        # rotation
        R1 = T1[:3, :3]
        R2 = T2[:3, :3]
        R = R1.T @ R2
        trace = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
        rot_angle = np.arccos(trace)
        return trans_dist, rot_angle

    # Visualization windows (same style as your class)

    def visualize_tracking(self):
        if self.viz_img is None:
            return

        traj = self._get_pred_xyz()
        if traj is None or traj.shape[0] < 2:
            imgui.separator()
            imgui.text("Trajectory: not enough poses yet.")
            imgui.end()
            return

        self.visualize_matching(traj)
        self.visualize_comparison(traj)

    def visualize_matching(self, traj):
        if not self.show_matching_window:
            return
        self.show_matching_window, _ = imgui.begin("ORB Tracking Matches", self.show_matching_window)

        if self.img_window is None:
            self.img_window = ImageWidget(self.viz_img)
        else:
            self.img_window.set_image_rgb(self.viz_img)
        self.img_window.draw()

        # contiguous arrays for implot
        x = np.ascontiguousarray(traj[:, 0], dtype=np.float32)
        y = np.ascontiguousarray(traj[:, 1], dtype=np.float32)
        z = np.ascontiguousarray(traj[:, 2], dtype=np.float32)
        t = np.ascontiguousarray(np.arange(traj.shape[0], dtype=np.float32))

        imgui.separator()
        imgui.text(f"Pred trajectory points: {traj.shape[0]}")
        if self.dataset is not None and getattr(self.dataset, "time_stamps", None) is not None and self.dataset.time_stamps.shape[0] > 1:
            imgui.text(f"Est streaming fps: {self.dataset.time_stamps.shape[0]/(self.dataset.time_stamps[-1]-self.dataset.time_stamps[0]):.2f}")

        def padded_limits(a, pad_ratio=0.05):
            amin = float(np.min(a))
            amax = float(np.max(a))
            pad = 1.0 if amin == amax else (amax - amin) * pad_ratio
            return amin - pad, amax + pad

        cond_always = imgui.Cond_.always

        # Plot Y(t)
        if implot.begin_plot("Pred: Y over time", (-1, 200)):
            implot.setup_axes("frame", "Y")
            x0, x1 = 0.0, float(t[-1])
            y0, y1 = padded_limits(y)
            implot.setup_axis_limits(implot.ImAxis_.x1, x0, x1, cond_always)
            implot.setup_axis_limits(implot.ImAxis_.y1, y0, y1, cond_always)
            implot.plot_line("pred y(t)", t, y)
            implot.end_plot()

        # Plot XZ
        if implot.begin_plot("Pred: XZ (top-down)", (-1, 240)):
            implot.setup_axes("X", "Z")
            x0, x1 = padded_limits(x)
            z0, z1 = padded_limits(z)
            implot.setup_axis_limits(implot.ImAxis_.x1, x0, x1, cond_always)
            implot.setup_axis_limits(implot.ImAxis_.y1, z0, z1, cond_always)
            implot.plot_line("pred xz", x, z)
            implot.end_plot()

        if imgui.begin_popup_context_window("window_ctx"):
            clicked, self.show_comparison_window = imgui.menu_item("Show Prediction/Ground truth comparision", "", self.show_comparison_window, True)
            imgui.end_popup()

        imgui.end()

    def visualize_comparison(self, traj):
        if not self.show_comparison_window:
            return
        self.show_comparison_window, _ = imgui.begin("Trajectory Comparison (Pred vs GT)", self.show_comparison_window)

        gt_traj_full = self._get_gt_xyz()
        if gt_traj_full is None or gt_traj_full.shape[0] < 2:
            imgui.text("Ground truth not available.")
            imgui.end()
            return

        n = min(traj.shape[0], gt_traj_full.shape[0])
        pred = traj[:n]
        gt = gt_traj_full[:n]

        valid = np.isfinite(gt).all(axis=1)
        pred = pred[valid]
        gt = gt[valid]

        if pred.shape[0] < 3:
            imgui.text("Not enough valid points to compare/align.")
            imgui.end()
            return

        pred_aligned, (s, R, tt) = self._umeyama_align(pred, gt, with_scale=True)
        imgui.text(f"Alignment: scale={s:.4f}")

        px = np.ascontiguousarray(pred_aligned[:, 0], dtype=np.float32)
        py = np.ascontiguousarray(pred_aligned[:, 1], dtype=np.float32)
        pz = np.ascontiguousarray(pred_aligned[:, 2], dtype=np.float32)
        gx = np.ascontiguousarray(gt[:, 0], dtype=np.float32)
        gy = np.ascontiguousarray(gt[:, 1], dtype=np.float32)
        gz = np.ascontiguousarray(gt[:, 2], dtype=np.float32)
        t2 = np.ascontiguousarray(np.arange(pred_aligned.shape[0], dtype=np.float32))

        cond_always = imgui.Cond_.always

        # Y(t)
        if implot.begin_plot("Y over time (Aligned Pred vs GT)", (-1, 240)):
            implot.setup_axes("frame", "Y")
            x0, x1 = 0.0, float(t2[-1])
            y0, y1 = self._padded_limits_from_two(py, gy)
            implot.setup_axis_limits(implot.ImAxis_.x1, x0, x1, cond_always)
            implot.setup_axis_limits(implot.ImAxis_.y1, y0, y1, cond_always)
            implot.plot_line("pred (aligned)", t2, py)
            implot.plot_line("gt", t2, gy)
            implot.end_plot()

        # XZ
        if implot.begin_plot("XZ top-down (Aligned Pred vs GT)", (-1, 280)):
            implot.setup_axes("X", "Z")
            x0, x1 = self._padded_limits_from_two(px, gx)
            z0, z1 = self._padded_limits_from_two(pz, gz)
            implot.setup_axis_limits(implot.ImAxis_.x1, x0, x1, cond_always)
            implot.setup_axis_limits(implot.ImAxis_.y1, z0, z1, cond_always)
            implot.plot_line("pred (aligned)", px, pz)
            implot.plot_line("gt", gx, gz)
            implot.end_plot()

        imgui.end()