from __future__ import annotations

import cv2
import numpy as np
import torch
from pytorch3d.ops import knn_points

from RTSGS.DataLoader.DataLoader import DataLoader
from RTSGS.GUI.ImageWidget import ImageWidget
from RTSGS.Tracker.Tracker import Tracker
from imgui_bundle import imgui, implot


class ProjectedPointToPlaneTracker(Tracker):
    """Fast RGB-D point-to-plane tracker using PyTorch3D KNN correspondences.

    The tracker aligns depth(t-1) to depth(t) and follows the existing RTSGS pose
    convention used by the current trackers:
      - T_rel maps previous camera coordinates -> current camera coordinates
      - global_pose_t = global_pose_{t-1} @ inv(T_rel)
    """

    def __init__(self, dataset: DataLoader, config):
        super().__init__(dataset, config)

        self.fx, self.fy = float(config.get("fx")), float(config.get("fy"))
        self.cx, self.cy = float(config.get("cx")), float(config.get("cy"))
        self.depth_scale = float(config.get("depth_scale"))

        self.alpha = float(config.get("kf_translation", 0.05))
        self.theta = float(config.get("kf_rotation", 5.0 * np.pi / 180.0))

        self.icp_stride = int(config.get("icp_stride", 4))
        self.icp_max_iters = int(config.get("icp_max_iters", 12))
        self.icp_corr_dist = float(config.get("icp_corr_dist", 0.12))
        self.icp_min_pairs = int(config.get("icp_min_pairs", 1200))
        self.icp_huber_delta = float(config.get("icp_huber_delta", 0.02))
        self.icp_damping = float(config.get("icp_damping", 1e-5))
        self.icp_plane_residual_max = float(config.get("icp_plane_residual_max", 0.04))
        self.icp_use_projective = bool(config.get("icp_use_projective", True))
        self.icp_proj_depth_max_diff = float(config.get("icp_proj_depth_max_diff", 0.08))
        self.depth_min = float(config.get("depth_min", 0.10))
        self.depth_max = float(config.get("depth_max", 8.0))

        self.depth_median_ksize = int(config.get("depth_median_ksize", 5))
        self.depth_bilateral_d = int(config.get("depth_bilateral_d", 5))
        self.depth_bilateral_sigma_color = float(config.get("depth_bilateral_sigma_color", 0.04))
        self.depth_bilateral_sigma_space = float(config.get("depth_bilateral_sigma_space", 2.0))

        cuda_index = int(config.get("cuda_device", 0))
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{cuda_index}")
        else:
            self.device = torch.device("cpu")

        self.poses = [self._initial_pose_from_dataset(dataset)]
        self.last_kf_pose = None

        self.prev_depth_m: np.ndarray | None = None
        self.prev_rgb: np.ndarray | None = None
        self.prev_rel_T = torch.eye(4, dtype=torch.float32, device=self.device)

        self.viz_img = None
        self.img_window = None
        self.show_matching_window = True
        self.show_comparison_window = False

        self.last_icp_pairs = 0
        self.last_icp_iterations = 0
        self.last_icp_rmse = 0.0

    def track_frame(self, rgb, depth=None):
        if depth is None:
            self.prev_rgb = rgb
            return None

        depth_m_raw = depth.astype(np.float32) / self.depth_scale
        depth_m = self._preprocess_depth(depth_m_raw)

        if self.prev_depth_m is None:
            self.prev_depth_m = depth_m
            self.prev_rgb = rgb
            self.viz_img = self._make_tracking_debug_image(rgb, depth_m, depth_m)
            init_pose = self.poses[0].astype(np.float32)
            if self.dataset is not None:
                self.dataset.rgb_keyframes.append(rgb)
                self.dataset.depth_keyframes.append(depth)
                self.keyframes_poses.append(init_pose)
                self.last_kf_pose = init_pose
                self.dataset.current_keyframe_index += 1
            return None

        T_rel = self._point_to_plane_icp(self.prev_depth_m, depth_m, self.prev_rel_T)
        self.viz_img = self._make_tracking_debug_image(rgb, self.prev_depth_m, depth_m)
        if T_rel is None:
            self.prev_depth_m = depth_m
            self.prev_rgb = rgb
            return None

        T_rel_np = T_rel.detach().cpu().numpy().astype(np.float32)
        pose = self.poses[-1] @ np.linalg.inv(T_rel_np)
        self.poses.append(pose.astype(np.float32))
        self.prev_rel_T = T_rel.detach()

        is_keyframe = False
        if self.last_kf_pose is None:
            is_keyframe = True
        else:
            dt, dR = self._pose_distance(self.last_kf_pose, pose)
            if dt > self.alpha or dR > self.theta:
                is_keyframe = True

        if is_keyframe and self.dataset is not None:
            self.dataset.rgb_keyframes.append(rgb)
            self.dataset.depth_keyframes.append(depth)
            self.keyframes_poses.append(pose.astype(np.float32))
            self.last_kf_pose = pose
            self.dataset.current_keyframe_index += 1

        self.prev_depth_m = depth_m
        self.prev_rgb = rgb
        return pose

    def visualize_tracking(self):
        if self.viz_img is None:
            return

        traj = self._get_pred_xyz()
        if traj is None or traj.shape[0] < 2:
            return

        self.visualize_matching(traj)
        self.visualize_comparison(traj)

    def visualize_matching(self, traj):
        if not self.show_matching_window:
            return

        self.show_matching_window, _ = imgui.begin("Point-to-Plane Tracking", self.show_matching_window)

        if self.img_window is None:
            self.img_window = ImageWidget(self.viz_img)
        else:
            self.img_window.set_image_rgb(self.viz_img)
        self.img_window.draw()

        x = np.ascontiguousarray(traj[:, 0], dtype=np.float32)
        y = np.ascontiguousarray(traj[:, 1], dtype=np.float32)
        z = np.ascontiguousarray(traj[:, 2], dtype=np.float32)
        t = np.ascontiguousarray(np.arange(traj.shape[0], dtype=np.float32))

        imgui.separator()
        imgui.text(f"Pred trajectory points: {traj.shape[0]}")
        imgui.text(f"Last ICP pairs: {self.last_icp_pairs}")
        imgui.text(f"Last ICP iterations: {self.last_icp_iterations}")
        imgui.text(f"Last ICP RMSE: {self.last_icp_rmse:.5f} m")

        if self.dataset is not None and getattr(self.dataset, "time_stamps", None) is not None and self.dataset.time_stamps.shape[0] > 1:
            fps = self.dataset.time_stamps.shape[0] / (self.dataset.time_stamps[-1] - self.dataset.time_stamps[0])
            imgui.text(f"Est streaming fps: {fps:.2f}")

        cond_always = imgui.Cond_.always

        if implot.begin_plot("Pred: Y over time", (-1, 200)):
            implot.setup_axes("frame", "Y")
            x0, x1 = 0.0, float(t[-1])
            y0, y1 = self._padded_limits(y)
            implot.setup_axis_limits(implot.ImAxis_.x1, x0, x1, cond_always)
            implot.setup_axis_limits(implot.ImAxis_.y1, y0, y1, cond_always)
            implot.plot_line("pred y(t)", t, y)
            implot.end_plot()

        if implot.begin_plot("Pred: XZ (top-down)", (-1, 240)):
            implot.setup_axes("X", "Z")
            x0, x1 = self._padded_limits(x)
            z0, z1 = self._padded_limits(z)
            implot.setup_axis_limits(implot.ImAxis_.x1, x0, x1, cond_always)
            implot.setup_axis_limits(implot.ImAxis_.y1, z0, z1, cond_always)
            implot.plot_line("pred xz", x, z)
            implot.end_plot()

        if imgui.begin_popup_context_window("window_ctx"):
            _, self.show_comparison_window = imgui.menu_item(
                "Show Prediction/Ground truth comparison", "", self.show_comparison_window, True
            )
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

        pred_aligned, (s, _, _) = self._umeyama_align(pred, gt, with_scale=True)
        imgui.text(f"Alignment: scale={s:.4f}")

        px = np.ascontiguousarray(pred_aligned[:, 0], dtype=np.float32)
        py = np.ascontiguousarray(pred_aligned[:, 1], dtype=np.float32)
        pz = np.ascontiguousarray(pred_aligned[:, 2], dtype=np.float32)
        gx = np.ascontiguousarray(gt[:, 0], dtype=np.float32)
        gy = np.ascontiguousarray(gt[:, 1], dtype=np.float32)
        gz = np.ascontiguousarray(gt[:, 2], dtype=np.float32)
        t2 = np.ascontiguousarray(np.arange(pred_aligned.shape[0], dtype=np.float32))

        cond_always = imgui.Cond_.always

        if implot.begin_plot("Y over time (Aligned Pred vs GT)", (-1, 240)):
            implot.setup_axes("frame", "Y")
            x0, x1 = 0.0, float(t2[-1])
            y0, y1 = self._padded_limits_from_two(py, gy)
            implot.setup_axis_limits(implot.ImAxis_.x1, x0, x1, cond_always)
            implot.setup_axis_limits(implot.ImAxis_.y1, y0, y1, cond_always)
            implot.plot_line("pred (aligned)", t2, py)
            implot.plot_line("gt", t2, gy)
            implot.end_plot()

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

    def _point_to_plane_icp(self, prev_depth_m: np.ndarray, cur_depth_m: np.ndarray, T_init: torch.Tensor):
        if self.icp_use_projective:
            return self._point_to_plane_icp_projective(prev_depth_m, cur_depth_m, T_init)

        # Fallback to KNN-style correspondence path (kept for compatibility).
        src = self._depth_to_points(prev_depth_m, self.icp_stride)
        tgt_pts, tgt_nrm = self._depth_to_points_normals(cur_depth_m, self.icp_stride)

        self.last_icp_pairs = 0
        self.last_icp_iterations = 0
        self.last_icp_rmse = 0.0

        if src is None or tgt_pts is None or tgt_nrm is None:
            return None
        if src.shape[0] < self.icp_min_pairs or tgt_pts.shape[0] < self.icp_min_pairs:
            return None

        T = T_init.clone()

        for iter_idx in range(self.icp_max_iters):
            src_t = self._transform_points(T, src)

            nn = knn_points(src_t[None, ...], tgt_pts[None, ...], K=1, return_nn=False)
            idx = nn.idx[0, :, 0]
            d2 = nn.dists[0, :, 0]

            tgt_corr = tgt_pts[idx]
            nrm_corr = tgt_nrm[idx]

            mask = d2 < (self.icp_corr_dist * self.icp_corr_dist)
            num_pairs = torch.count_nonzero(mask).item()
            if num_pairs < self.icp_min_pairs:
                return None

            p = src_t[mask]
            q = tgt_corr[mask]
            n = nrm_corr[mask]

            r = torch.sum(n * (p - q), dim=1)
            if self.icp_plane_residual_max > 0.0:
                residual_mask = torch.abs(r) < self.icp_plane_residual_max
                if torch.count_nonzero(residual_mask).item() < self.icp_min_pairs:
                    return None
                p = p[residual_mask]
                q = q[residual_mask]
                n = n[residual_mask]
                r = r[residual_mask]

            w = self._huber_weight(r, self.icp_huber_delta)

            self.last_icp_pairs = num_pairs
            self.last_icp_iterations = iter_idx + 1
            self.last_icp_rmse = float(torch.sqrt(torch.mean(r * r)).item())

            A_rot = torch.cross(p, n, dim=1)
            A = torch.cat([A_rot, n], dim=1)
            b = -r

            Aw = A * w[:, None]
            H = Aw.T @ A + self.icp_damping * torch.eye(6, device=self.device, dtype=torch.float32)
            g = Aw.T @ b

            try:
                xi = torch.linalg.solve(H, g)
            except RuntimeError:
                return None

            if not torch.isfinite(xi).all():
                return None

            dT = self._se3_exp(xi)
            T = dT @ T

            if torch.linalg.norm(xi).item() < 1e-5:
                break

        return T

    def _point_to_plane_icp_projective(self, prev_depth_m: np.ndarray, cur_depth_m: np.ndarray, T_init: torch.Tensor):
        prev_d = torch.from_numpy(prev_depth_m).to(self.device)
        cur_d = torch.from_numpy(cur_depth_m).to(self.device)

        h, w = prev_d.shape

        self.last_icp_pairs = 0
        self.last_icp_iterations = 0
        self.last_icp_rmse = 0.0

        # Build source sample from previous depth (subsampled grid).
        v = torch.arange(1, h - 1, self.icp_stride, device=self.device)
        u = torch.arange(1, w - 1, self.icp_stride, device=self.device)
        uu, vv = torch.meshgrid(u, v, indexing="xy")

        z = prev_d[vv, uu]
        valid = (z > self.depth_min) & (z < self.depth_max) & torch.isfinite(z)
        if torch.count_nonzero(valid).item() < self.icp_min_pairs:
            return None

        uu = uu[valid].float()
        vv = vv[valid].float()
        z = z[valid].float()

        x = (uu - self.cx) * z / self.fx
        y = (vv - self.cy) * z / self.fy
        src = torch.stack([x, y, z], dim=1)

        # Build current-frame vertex and normal maps once.
        vmap, nmap, valid_map = self._vertex_normal_maps(cur_d)

        T = T_init.clone()

        for iter_idx in range(self.icp_max_iters):
            p = self._transform_points(T, src)

            zc = p[:, 2]
            valid_z = zc > 1e-6
            u_proj = self.fx * (p[:, 0] / zc) + self.cx
            v_proj = self.fy * (p[:, 1] / zc) + self.cy

            ui = torch.round(u_proj).long()
            vi = torch.round(v_proj).long()

            in_bounds = (ui >= 1) & (ui < (w - 1)) & (vi >= 1) & (vi < (h - 1))
            mask = valid_z & in_bounds
            if torch.count_nonzero(mask).item() < self.icp_min_pairs:
                return None

            p = p[mask]
            ui = ui[mask]
            vi = vi[mask]

            q = vmap[vi, ui]
            n = nmap[vi, ui]
            vm = valid_map[vi, ui]
            if torch.count_nonzero(vm).item() < self.icp_min_pairs:
                return None

            p = p[vm]
            q = q[vm]
            n = n[vm]

            # Reject pairs with large depth disagreement (projective outlier gate).
            if self.icp_proj_depth_max_diff > 0.0:
                dz_mask = torch.abs(p[:, 2] - q[:, 2]) < self.icp_proj_depth_max_diff
                if torch.count_nonzero(dz_mask).item() < self.icp_min_pairs:
                    return None
                p = p[dz_mask]
                q = q[dz_mask]
                n = n[dz_mask]

            r = torch.sum(n * (p - q), dim=1)
            if self.icp_plane_residual_max > 0.0:
                residual_mask = torch.abs(r) < self.icp_plane_residual_max
                if torch.count_nonzero(residual_mask).item() < self.icp_min_pairs:
                    return None
                p = p[residual_mask]
                q = q[residual_mask]
                n = n[residual_mask]
                r = r[residual_mask]

            num_pairs = p.shape[0]
            self.last_icp_pairs = int(num_pairs)
            self.last_icp_iterations = iter_idx + 1
            self.last_icp_rmse = float(torch.sqrt(torch.mean(r * r)).item())

            w_huber = self._huber_weight(r, self.icp_huber_delta)

            A_rot = torch.cross(p, n, dim=1)
            A = torch.cat([A_rot, n], dim=1)
            b = -r

            Aw = A * w_huber[:, None]
            H = Aw.T @ A + self.icp_damping * torch.eye(6, device=self.device, dtype=torch.float32)
            g = Aw.T @ b

            try:
                xi = torch.linalg.solve(H, g)
            except RuntimeError:
                return None

            if not torch.isfinite(xi).all():
                return None

            dT = self._se3_exp(xi)
            T = dT @ T

            if torch.linalg.norm(xi).item() < 1e-5:
                break

        return T

    def _vertex_normal_maps(self, depth_t: torch.Tensor):
        h, w = depth_t.shape

        u = torch.arange(0, w, device=self.device, dtype=torch.float32)
        v = torch.arange(0, h, device=self.device, dtype=torch.float32)
        uu, vv = torch.meshgrid(u, v, indexing="xy")

        z = depth_t
        x = (uu - self.cx) * z / self.fx
        y = (vv - self.cy) * z / self.fy
        vmap = torch.stack([x, y, z], dim=2)

        # Neighbor differences (valid only for interior pixels)
        vx = vmap[1:-1, 2:, :] - vmap[1:-1, :-2, :]
        vy = vmap[2:, 1:-1, :] - vmap[:-2, 1:-1, :]

        n_inner = torch.cross(vx, vy, dim=2)
        n_norm = torch.linalg.norm(n_inner, dim=2, keepdim=True)
        n_inner = n_inner / torch.clamp(n_norm, min=1e-8)

        nmap = torch.zeros((h, w, 3), device=self.device, dtype=torch.float32)
        nmap[1:-1, 1:-1, :] = n_inner

        # Keep normals facing camera.
        toward_camera = nmap[..., 2] > 0.0
        nmap[toward_camera] = -nmap[toward_camera]

        z_valid = (z > self.depth_min) & (z < self.depth_max) & torch.isfinite(z)
        n_valid = torch.isfinite(nmap).all(dim=2) & (torch.linalg.norm(nmap, dim=2) > 1e-6)
        valid_map = z_valid & n_valid

        return vmap, nmap, valid_map

    def _depth_to_points(self, depth_m: np.ndarray, stride: int):
        d = torch.from_numpy(depth_m).to(self.device)
        h, w = d.shape

        v = torch.arange(0, h, stride, device=self.device)
        u = torch.arange(0, w, stride, device=self.device)
        uu, vv = torch.meshgrid(u, v, indexing="xy")

        z = d[vv, uu]
        valid = (z > self.depth_min) & (z < self.depth_max) & torch.isfinite(z)
        if torch.count_nonzero(valid).item() < 16:
            return None

        uu = uu[valid].float()
        vv = vv[valid].float()
        z = z[valid].float()

        x = (uu - self.cx) * z / self.fx
        y = (vv - self.cy) * z / self.fy
        return torch.stack([x, y, z], dim=1)

    def _depth_to_points_normals(self, depth_m: np.ndarray, stride: int):
        d = torch.from_numpy(depth_m).to(self.device)
        h, w = d.shape

        v = torch.arange(1, h - 1, stride, device=self.device)
        u = torch.arange(1, w - 1, stride, device=self.device)
        uu, vv = torch.meshgrid(u, v, indexing="xy")

        z = d[vv, uu]
        valid = (z > self.depth_min) & (z < self.depth_max) & torch.isfinite(z)
        if torch.count_nonzero(valid).item() < 16:
            return None, None

        uu_f = uu.float()
        vv_f = vv.float()

        x = (uu_f - self.cx) * z / self.fx
        y = (vv_f - self.cy) * z / self.fy
        p = torch.stack([x, y, z], dim=2)

        z_l = d[vv, uu - 1]
        z_r = d[vv, uu + 1]
        z_u = d[vv - 1, uu]
        z_d = d[vv + 1, uu]

        px_l = (uu_f - 1.0 - self.cx) * z_l / self.fx
        py_l = (vv_f - self.cy) * z_l / self.fy
        p_l = torch.stack([px_l, py_l, z_l], dim=2)

        px_r = (uu_f + 1.0 - self.cx) * z_r / self.fx
        py_r = (vv_f - self.cy) * z_r / self.fy
        p_r = torch.stack([px_r, py_r, z_r], dim=2)

        px_u = (uu_f - self.cx) * z_u / self.fx
        py_u = (vv_f - 1.0 - self.cy) * z_u / self.fy
        p_u = torch.stack([px_u, py_u, z_u], dim=2)

        px_d = (uu_f - self.cx) * z_d / self.fx
        py_d = (vv_f + 1.0 - self.cy) * z_d / self.fy
        p_d = torch.stack([px_d, py_d, z_d], dim=2)

        vx = p_r - p_l
        vy = p_d - p_u
        n = torch.cross(vx, vy, dim=2)
        n_norm = torch.linalg.norm(n, dim=2)

        valid_neighbors = (
            torch.isfinite(z_l)
            & torch.isfinite(z_r)
            & torch.isfinite(z_u)
            & torch.isfinite(z_d)
            & (z_l > self.depth_min)
            & (z_r > self.depth_min)
            & (z_u > self.depth_min)
            & (z_d > self.depth_min)
            & (z_l < self.depth_max)
            & (z_r < self.depth_max)
            & (z_u < self.depth_max)
            & (z_d < self.depth_max)
        )

        valid = valid & valid_neighbors & (n_norm > 1e-7) & torch.isfinite(n_norm)
        if torch.count_nonzero(valid).item() < 16:
            return None, None

        pts = p[valid]
        nrms = n[valid] / n_norm[valid][:, None]

        toward_camera = nrms[:, 2] > 0.0
        nrms[toward_camera] = -nrms[toward_camera]

        return pts, nrms

    @staticmethod
    def _huber_weight(r: torch.Tensor, delta: float):
        a = torch.abs(r)
        ones = torch.ones_like(a)
        return torch.where(a <= delta, ones, delta / (a + 1e-12))

    @staticmethod
    def _transform_points(T: torch.Tensor, pts: torch.Tensor):
        R = T[:3, :3]
        t = T[:3, 3]
        return pts @ R.T + t

    @staticmethod
    def _skew(w: torch.Tensor):
        wx, wy, wz = w[0], w[1], w[2]
        return torch.tensor(
            [[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]],
            device=w.device,
            dtype=w.dtype,
        )

    def _se3_exp(self, xi: torch.Tensor):
        w = xi[:3]
        v = xi[3:]
        theta = torch.linalg.norm(w)

        I = torch.eye(3, device=self.device, dtype=torch.float32)
        wx = self._skew(w)

        if theta.item() < 1e-8:
            R = I + wx
            V = I + 0.5 * wx
        else:
            theta2 = theta * theta
            A = torch.sin(theta) / theta
            B = (1.0 - torch.cos(theta)) / theta2
            C = (theta - torch.sin(theta)) / (theta2 * theta)
            R = I + A * wx + B * (wx @ wx)
            V = I + B * wx + C * (wx @ wx)

        t = V @ v

        T = torch.eye(4, device=self.device, dtype=torch.float32)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    @staticmethod
    def _pose_distance(T1, T2):
        t1 = T1[:3, 3]
        t2 = T2[:3, 3]
        trans_dist = np.linalg.norm(t1 - t2)

        R1 = T1[:3, :3]
        R2 = T2[:3, :3]
        R = R1.T @ R2
        trace = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
        rot_angle = np.arccos(trace)

        return trans_dist, rot_angle

    @staticmethod
    def _initial_pose_from_dataset(dataset: DataLoader):
        I = np.eye(4, dtype=np.float32)
        if dataset is None or getattr(dataset, "gt_poses", None) is None:
            return I
        if len(dataset.gt_poses) == 0:
            return I

        T = np.asarray(dataset.gt_poses[0], dtype=np.float32)
        if T.shape != (4, 4):
            return I
        if not np.isfinite(T).all():
            return I

        R = T[:3, :3]
        if abs(np.linalg.det(R)) < 1e-8:
            return I

        T = T.copy()
        T[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
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
    def _padded_limits(a, pad_ratio=0.05):
        amin = float(np.min(a))
        amax = float(np.max(a))
        if amin == amax:
            pad = 1.0
        else:
            pad = (amax - amin) * pad_ratio
        return amin - pad, amax + pad

    @staticmethod
    def _padded_limits_from_two(a, b, pad_ratio=0.05):
        amin = float(min(np.min(a), np.min(b)))
        amax = float(max(np.max(a), np.max(b)))
        if amin == amax:
            pad = 1.0
        else:
            pad = (amax - amin) * pad_ratio
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

    def _make_tracking_debug_image(self, rgb, prev_depth_m, cur_depth_m):
        if rgb is None:
            return None

        rgb_vis = rgb if rgb.ndim == 3 else cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
        h, w = rgb_vis.shape[:2]

        prev_color = self._depth_to_colormap(prev_depth_m, h, w)
        cur_color = self._depth_to_colormap(cur_depth_m, h, w)

        top = np.hstack([rgb_vis, prev_color])
        bottom = np.hstack([np.zeros_like(rgb_vis), cur_color])
        canvas = np.vstack([top, bottom])

        cv2.putText(canvas, "RGB", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Prev Depth", (w + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Cur Depth", (w + 10, h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        return canvas

    def _depth_to_colormap(self, depth_m, h, w):
        if depth_m is None:
            return np.zeros((h, w, 3), dtype=np.uint8)

        d = np.asarray(depth_m, dtype=np.float32)
        if d.shape[0] != h or d.shape[1] != w:
            d = cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST)

        valid = np.isfinite(d) & (d > self.depth_min) & (d < self.depth_max)
        img = np.zeros((h, w), dtype=np.uint8)

        if np.any(valid):
            dv = d[valid]
            dmin = float(np.percentile(dv, 5.0))
            dmax = float(np.percentile(dv, 95.0))
            if dmax <= dmin:
                dmax = dmin + 1e-6
            norm = np.clip((d - dmin) / (dmax - dmin), 0.0, 1.0)
            img = (norm * 255.0).astype(np.uint8)
            img[~valid] = 0

        return cv2.applyColorMap(img, cv2.COLORMAP_TURBO)

    def _preprocess_depth(self, depth_m: np.ndarray):
        d = np.asarray(depth_m, dtype=np.float32)

        valid = np.isfinite(d) & (d > self.depth_min) & (d < self.depth_max)
        if not np.any(valid):
            return np.zeros_like(d, dtype=np.float32)

        out = d.copy()
        out[~valid] = 0.0

        k = self.depth_median_ksize
        if k >= 3 and (k % 2) == 1:
            out = cv2.medianBlur(out, k)

        if self.depth_bilateral_d >= 3:
            out = cv2.bilateralFilter(
                out,
                d=self.depth_bilateral_d,
                sigmaColor=self.depth_bilateral_sigma_color,
                sigmaSpace=self.depth_bilateral_sigma_space,
            )

        out = np.asarray(out, dtype=np.float32)
        out[~np.isfinite(out)] = 0.0
        out[(out <= self.depth_min) | (out >= self.depth_max)] = 0.0
        return out
