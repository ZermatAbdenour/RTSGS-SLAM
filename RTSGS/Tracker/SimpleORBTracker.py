from RTSGS.Config.Config import Config
from RTSGS.DataLoader.DataLoader import DataLoader
from RTSGS.Tracker.Tracker import Tracker
import numpy as np
import cv2
from RTSGS.GUI.ImageWidget import ImageWidget
from imgui_bundle import imgui, implot


class SimpleORBTracker(Tracker):

    def __init__(self,dataset:DataLoader,config:Config, Orb_features=1000):
        super().__init__(dataset,config)
        self.fx, self.fy = config.get('fx'), config.get('fy')
        self.cx, self.cy = config.get('cx'), config.get('cy')
        self.K = config.get_camera_intrinsics()

        self.depth_scale = config.get('depth_scale')
        self.orb = cv2.ORB_create(nfeatures=Orb_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.poses = [dataset.gt_poses[0 ]]
        #keyframes
        # keyframe thresholds
        self.alpha = config.get("kf_translation", 0.05)
        self.theta = config.get("kf_rotation", 5.0 * np.pi / 180.0) 

        self.last_kf_pose = None


        self.prev_rgb = None

        self.viz_img = None
        self.img_window = None

        self.show_matching_window = True
        self.show_comparison_window = False
    def track_frame(self, rgb, depth=None, use_grayscale: bool = False):
        # Choose representation for ORB (and keep it consistent across frames)
        if use_grayscale and rgb is not None and rgb.ndim == 3:
            cur_img = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        else:
            cur_img = rgb

        if self.prev_rgb is None:
            self.prev_rgb = cur_img
            return None

        prev_img = self.prev_rgb

        # depth in meters
        depth_m = depth.astype(np.float32) / float(self.depth_scale)
        h, w = depth_m.shape[:2]

        # --- detect/compute ---
        kp1, des1 = self.orb.detectAndCompute(prev_img, None)
        kp2, des2 = self.orb.detectAndCompute(cur_img, None)

        if des1 is None or des2 is None or len(kp1) < 12 or len(kp2) < 12:
            self.prev_rgb = cur_img
            return None

        # --- matching (KNN + Lowe ratio) ---
        if not hasattr(self, "_bf_knn"):
            self._bf_knn = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

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
            return None

        good.sort(key=lambda x: x.distance)
        max_corr = 1500
        good = good[:max_corr]

        # --- build arrays (vectorized) ---
        qidx = np.fromiter((m.queryIdx for m in good), dtype=np.int32, count=len(good))
        tidx = np.fromiter((m.trainIdx for m in good), dtype=np.int32, count=len(good))

        pts1 = np.array([kp1[i].pt for i in qidx], dtype=np.float32)
        pts2 = np.array([kp2[i].pt for i in tidx], dtype=np.float32)

        ui = np.clip(pts1[:, 0].astype(np.int32), 0, w - 1)
        vi = np.clip(pts1[:, 1].astype(np.int32), 0, h - 1)

        z = depth_m[vi, ui]
        valid = (z > 0.0) & np.isfinite(z)

        if valid.sum() < 12:
            self.prev_rgb = cur_img
            return None

        pts1v = pts1[valid]
        pts2v = pts2[valid]
        zv = z[valid]

        X = (pts1v[:, 0] - self.cx) * zv / self.fx
        Y = (pts1v[:, 1] - self.cy) * zv / self.fy
        pts3d = np.stack([X, Y, zv], axis=1).astype(np.float32)
        pts2d = pts2v.astype(np.float32)

        if pts3d.shape[0] < 12:
            self.prev_rgb = cur_img
            return None

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d,
            pts2d,
            self.K,
            None,
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=3.0,
            confidence=0.999,
            iterationsCount=1000
        )
        if not ok or inliers is None or len(inliers) < 8:
            self.prev_rgb = cur_img
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
            return None

        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R.astype(np.float32)
        T[:3, 3] = tvec.reshape(3).astype(np.float32)

        pose = self.poses[-1] @ np.linalg.inv(T)   
        if(len(self.poses)+2 <len(self.dataset.gt_poses)):
            pose = self.dataset.gt_poses[len(self.poses )]
        else:
            return
        self.poses.append(pose)  
        #self.poses.append(pose.astype(np.float32))


        # add key frames
        is_keyframe = False

        if self.last_kf_pose is None:
            is_keyframe = True
        else:
            dt, dR = self._pose_distance(self.last_kf_pose, pose)
            if dt > self.alpha or dR > self.theta:
                is_keyframe = True

        if is_keyframe and self.dataset is not None:
            # store RGB keyframe
            self.dataset.rgb_keyframes.append(rgb)

            # store depth temporarily (will be cleared after updating pointcloud)
            if depth is not None:
                self.dataset.depth_keyframes.append(depth)
            self.keyframes_poses.append(pose.astype(np.float32))
            self.last_kf_pose = pose
            self.dataset.current_keyframe_index+=1

        # visualization
        if(self.show_matching_window):
            draw_n = min(80, len(good))
            self.viz_img = cv2.drawMatches(
                prev_img, kp1, cur_img, kp2, good[:draw_n], None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

        self.prev_rgb = cur_img
        return pose

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
        if amin == amax:
            pad = 1.0
        else:
            pad = (amax - amin) * pad_ratio
        return amin - pad, amax + pad

    @staticmethod
    def _umeyama_align(src, dst, with_scale=True):
        """
        Align src (Nx3) to dst (Nx3). Returns (aligned_src, (scale, R, t)).
        Similarity transform (Sim(3)) via Umeyama.
        """
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
        """
        Compute translation (L2) and rotation (angle) distance between two poses.
        """
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

    def visualize_matching(self,traj):

        if(not self.show_matching_window):
            return
        self.show_matching_window,_ = imgui.begin("ORB Tracking Matches",self.show_matching_window)

        if self.img_window is None:
            self.img_window = ImageWidget(self.viz_img)
        else:
            self.img_window.set_image_rgb(self.viz_img)
        self.img_window.draw()
        # contiguous 1D arrays for implot
        x = np.ascontiguousarray(traj[:, 0], dtype=np.float32)
        y = np.ascontiguousarray(traj[:, 1], dtype=np.float32)
        z = np.ascontiguousarray(traj[:, 2], dtype=np.float32)
        t = np.ascontiguousarray(np.arange(traj.shape[0], dtype=np.float32))

        imgui.separator()
        imgui.text(f"Pred trajectory points: {traj.shape[0]}")
        imgui.text(f"Est streaming fps: {self.dataset.time_stamps.shape[0]/(self.dataset.time_stamps[-1]-self.dataset.time_stamps[0]):.2f}")
        def padded_limits(a, pad_ratio=0.05):
            amin = float(np.min(a))
            amax = float(np.max(a))
            if amin == amax:
                pad = 1.0
            else:
                pad = (amax - amin) * pad_ratio
            return amin - pad, amax + pad

        cond_always = imgui.Cond_.always

        # ---- Plot 1: Pred Y(t) auto-fit each frame ----
        if implot.begin_plot("Pred: Y over time", (-1, 200)):
            implot.setup_axes("frame", "Y")

            x0, x1 = 0.0, float(t[-1])
            y0, y1 = padded_limits(y)

            implot.setup_axis_limits(implot.ImAxis_.x1, x0, x1, cond_always)
            implot.setup_axis_limits(implot.ImAxis_.y1, y0, y1, cond_always)

            implot.plot_line("pred y(t)", t, y)
            implot.end_plot()

        # ---- Plot 2: Pred XZ auto-fit each frame ----
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

    def visualize_comparison(self,traj):
        if(not self.show_comparison_window):
            return
        self.show_comparison_window,_ = imgui.begin("Trajectory Comparison (Pred vs GT)",self.show_comparison_window)


        gt_traj_full = self._get_gt_xyz()

        if gt_traj_full is None or gt_traj_full.shape[0] < 2:
            imgui.text("Ground truth not available.")
            imgui.end()
            return

        # Common prefix (assumes dataset.gt_poses is time-aligned to RGB frames!)
        n = min(traj.shape[0], gt_traj_full.shape[0])
        pred = traj[:n]
        gt = gt_traj_full[:n]

        # Drop NaN GT frames if your loader stores NaNs for unmatched timestamps
        valid = np.isfinite(gt).all(axis=1)
        pred = pred[valid]
        gt = gt[valid]

        if pred.shape[0] < 3:
            imgui.text("Not enough valid points to compare/align.")
            imgui.end()
            return

        # Align predicted -> GT so the overlay is meaningful
        pred_aligned, (s, R, tt) = self._umeyama_align(pred, gt, with_scale=True)
        
        imgui.text(f"Alignment: scale={s:.4f}")

        # contiguous arrays for plotting
        px = np.ascontiguousarray(pred_aligned[:, 0], dtype=np.float32)
        py = np.ascontiguousarray(pred_aligned[:, 1], dtype=np.float32)
        pz = np.ascontiguousarray(pred_aligned[:, 2], dtype=np.float32)

        gx = np.ascontiguousarray(gt[:, 0], dtype=np.float32)
        gy = np.ascontiguousarray(gt[:, 1], dtype=np.float32)
        gz = np.ascontiguousarray(gt[:, 2], dtype=np.float32)

        t2 = np.ascontiguousarray(np.arange(pred_aligned.shape[0], dtype=np.float32))

        cond_always = imgui.Cond_.always

        # ---- Plot: Y(t) (auto-fit each frame) ----
        if implot.begin_plot("Y over time (Aligned Pred vs GT)", (-1, 240)):
            implot.setup_axes("frame", "Y")

            x0, x1 = 0.0, float(t2[-1])
            y0, y1 = self._padded_limits_from_two(py, gy)

            implot.setup_axis_limits(implot.ImAxis_.x1, x0, x1, cond_always)
            implot.setup_axis_limits(implot.ImAxis_.y1, y0, y1, cond_always)

            implot.plot_line("pred (aligned)", t2, py)
            implot.plot_line("gt", t2, gy)
            implot.end_plot()

        # ---- Plot: XZ (auto-fit each frame) ----
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
        