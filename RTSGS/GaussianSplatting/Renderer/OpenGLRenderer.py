import torch
import numpy as np
from OpenGL.GL import *
from OpenGL import GL
from .FrameBuffer import FrameBuffer
import RTSGS.GaussianSplatting.Renderer.Resources as res
from .Camera import Camera
from .PoseWireframe import PoseWireframeBuilder
from imgui_bundle import imgui
import ctypes

class Renderer:
    def __init__(self, pcd, camera: Camera, tracker=None, dataset=None):
        # Initialize the resources
        res.init_resources()

        self.fb = FrameBuffer(width=800, height=600)
        self.pcd = pcd
        self.vbo_capacity_bytes = 0
        # Setup OpenGL buffers
        self.vbo = None
        self.vao = None
        self._initialized = False
        
        # camera setup
        self.camera = camera
        self.tracker = tracker
        self.dataset = dataset
        self.pose_wireframe_builder = PoseWireframeBuilder(camera_scale=0.04, aspect=0.75)

        self.pose_camera_vao = None
        self.pose_camera_vbo = None
        self.pose_camera_capacity_bytes = 0
        self.pose_camera_count = 0

        self.pose_traj_vao = None
        self.pose_traj_vbo = None
        self.pose_traj_capacity_bytes = 0
        self.pose_pred_traj_count = 0
        self.pose_gt_traj_count = 0

        self._pose_cache_sig = None
        self._pose_dirty = True

        self.use_segmentation_colors = False
        self.segmentation_class_filter = None
        self._last_color_mode = False
        self._last_segmentation_version = -1
        self._last_segmentation_filter = None
        self.show_instance_bboxes = True

        self.instance_bbox_vao = None
        self.instance_bbox_vbo = None
        self.instance_bbox_capacity_bytes = 0
        self.instance_bbox_vertex_count = 0
        self._last_instance_bbox_seg_version = -1

        # Opengl 
        # Enable depth testing for 3D points
        self.fb.bind()
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glEnable(GL_PROGRAM_POINT_SIZE) 
        res.simple_shader.use()
        self.fb.unbind()

        self.pcd_added_size = 0

        print("GL_VENDOR  :", glGetString(GL_VENDOR).decode())
        print("GL_RENDERER:", glGetString(GL_RENDERER).decode())
        print("GL_VERSION :", glGetString(GL_VERSION).decode())

    def _initialize_pose_rendering(self):
        if self.pose_camera_vao is not None:
            return

        self.pose_camera_vbo = glGenBuffers(1)
        self.pose_camera_vao = glGenVertexArrays(1)

        glBindVertexArray(self.pose_camera_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.pose_camera_vbo)
        glBufferData(GL_ARRAY_BUFFER, 0, None, GL_DYNAMIC_DRAW)
        cam_stride = 15 * 4
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, cam_stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, cam_stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, cam_stride, ctypes.c_void_p(24))
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, cam_stride, ctypes.c_void_p(36))
        glEnableVertexAttribArray(4)
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, cam_stride, ctypes.c_void_p(48))

        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # Trajectory lines (non-instanced): pos + color
        self.pose_traj_vbo = glGenBuffers(1)
        self.pose_traj_vao = glGenVertexArrays(1)

        glBindVertexArray(self.pose_traj_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.pose_traj_vbo)
        glBufferData(GL_ARRAY_BUFFER, 0, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(12))

        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _initialize_instance_bbox_rendering(self):
        if self.instance_bbox_vao is not None:
            return

        self.instance_bbox_vbo = glGenBuffers(1)
        self.instance_bbox_vao = glGenVertexArrays(1)

        glBindVertexArray(self.instance_bbox_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_bbox_vbo)
        glBufferData(GL_ARRAY_BUFFER, 0, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(12))
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _build_instance_bbox_vertices(self, instances, class_palette=None):
        if not isinstance(instances, list) or len(instances) == 0:
            return np.empty((0, 6), dtype=np.float32)

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]

        verts = []
        for inst in instances:
            try:
                bmin = np.asarray(inst.get("bbox_min", [0.0, 0.0, 0.0]), dtype=np.float32)
                bmax = np.asarray(inst.get("bbox_max", [0.0, 0.0, 0.0]), dtype=np.float32)
                if bmin.shape[0] != 3 or bmax.shape[0] != 3:
                    continue

                x0, y0, z0 = float(bmin[0]), float(bmin[1]), float(bmin[2])
                x1, y1, z1 = float(bmax[0]), float(bmax[1]), float(bmax[2])

                corners = np.asarray(
                    [
                        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
                    ],
                    dtype=np.float32,
                )

                cls_id = int(inst.get("class_id", -1))
                if class_palette is not None and 0 <= cls_id < class_palette.shape[0]:
                    col = np.asarray(class_palette[cls_id], dtype=np.float32)
                else:
                    col = np.asarray([1.0, 1.0, 0.2], dtype=np.float32)

                for a, b in edges:
                    verts.append(np.concatenate([corners[a], col], axis=0))
                    verts.append(np.concatenate([corners[b], col], axis=0))
            except Exception:
                continue

        if len(verts) == 0:
            return np.empty((0, 6), dtype=np.float32)
        return np.asarray(verts, dtype=np.float32)

    def _update_instance_bbox_vbo(self, vertices: np.ndarray):
        if self.instance_bbox_vao is None:
            self._initialize_instance_bbox_rendering()

        self.instance_bbox_vertex_count = int(vertices.shape[0])
        if self.instance_bbox_vertex_count <= 0:
            return

        bytes_required = int(vertices.nbytes)
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_bbox_vbo)
        if bytes_required > self.instance_bbox_capacity_bytes:
            self.instance_bbox_capacity_bytes = bytes_required
            glBufferData(GL_ARRAY_BUFFER, self.instance_bbox_capacity_bytes, vertices, GL_DYNAMIC_DRAW)
        else:
            glBufferSubData(GL_ARRAY_BUFFER, 0, bytes_required, vertices)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def render_instance_bboxes(self):
        if not bool(self.show_instance_bboxes):
            return

        with self.pcd.lock:
            seg_version = int(getattr(self.pcd, "segmentation_version", 0))
            instances = list(getattr(self.pcd, "segmentation_instances", []))
            md = getattr(self.pcd, "segmentation_metadata", {})
            class_palette = None
            if isinstance(md, dict) and "class_palette" in md:
                try:
                    class_palette = np.asarray(md["class_palette"], dtype=np.float32)
                except Exception:
                    class_palette = None

        if seg_version != self._last_instance_bbox_seg_version:
            vertices = self._build_instance_bbox_vertices(instances, class_palette=class_palette)
            self._update_instance_bbox_vbo(vertices)
            self._last_instance_bbox_seg_version = seg_version

        if self.instance_bbox_vertex_count <= 0 or self.instance_bbox_vao is None:
            return

        self.camera.update_view()
        res.line_shader.use()
        glUniformMatrix4fv(
            glGetUniformLocation(res.line_shader.program, 'u_view'),
            1,
            GL_FALSE,
            self.camera.view,
        )
        glUniformMatrix4fv(
            glGetUniformLocation(res.line_shader.program, 'u_projection'),
            1,
            GL_FALSE,
            self.camera.projection,
        )
        glBindVertexArray(self.instance_bbox_vao)
        glLineWidth(2.0)
        glDrawArrays(GL_LINES, 0, self.instance_bbox_vertex_count)
        glBindVertexArray(0)

    def _build_pose_overlay_data(self):
        if self.tracker is None:
            return np.empty((0, 15), dtype=np.float32), np.empty((0, 6), dtype=np.float32), np.empty((0, 6), dtype=np.float32)

        # Keep pose overlays in the same corrected world frame as map points.
        R_fix = None
        if hasattr(self.pcd, "R_fix") and self.pcd.R_fix is not None:
            try:
                R_fix = self.pcd.R_fix.detach().cpu().numpy().astype(np.float32)
            except Exception:
                R_fix = None

        def _correct_pose(T):
            if T is None:
                return None
            T = np.asarray(T, dtype=np.float32)
            if T.shape != (4, 4):
                return None
            if R_fix is None:
                return T
            T_corr = T.copy()
            T_corr[:3, :3] = R_fix @ T[:3, :3]
            T_corr[:3, 3] = R_fix @ T[:3, 3]
            return T_corr

        pred_poses = [_correct_pose(T) for T in list(getattr(self.tracker, "keyframes_poses", []))]

        gt_poses = []
        gt_poses_all = getattr(self.dataset, "gt_poses", None) if self.dataset is not None else None
        if gt_poses_all is not None:
            gt_poses_all = np.asarray(gt_poses_all)
            kf_frame_indices = list(getattr(self.tracker, "keyframe_frame_indices", []))
            if kf_frame_indices:
                for idx in kf_frame_indices:
                    i = int(idx)
                    if 0 <= i < gt_poses_all.shape[0]:
                        gt_poses.append(_correct_pose(gt_poses_all[i]))
            else:
                n = min(len(pred_poses), gt_poses_all.shape[0])
                gt_poses = [_correct_pose(gt_poses_all[i]) for i in range(n)]

        def _safe_normalize(v):
            n = float(np.linalg.norm(v))
            if n < 1e-8 or not np.isfinite(n):
                return None
            return (v / n).astype(np.float32)

        def _build_camera_points(poses, color):
            valid = [p for p in poses if p is not None and np.asarray(p).shape == (4, 4) and np.isfinite(p).all()]
            if not valid:
                return np.empty((0, 15), dtype=np.float32)

            rows = []
            col = np.asarray(color, dtype=np.float32)
            for p in valid:
                R = np.asarray(p[:3, :3], dtype=np.float32)
                c = np.asarray(p[:3, 3], dtype=np.float32)

                # OpenCV-style camera basis in world space:
                # x: right, y: down, z: forward.
                right = _safe_normalize(R[:, 0])
                up = _safe_normalize(-R[:, 1])
                forward = _safe_normalize(R[:, 2])

                if right is None or up is None or forward is None:
                    continue

                rows.append(np.concatenate([c, col, forward, up, right], axis=0))

            if not rows:
                return np.empty((0, 15), dtype=np.float32)
            return np.asarray(rows, dtype=np.float32)

        pred_cam_points = _build_camera_points(pred_poses, (1.0, 0.0, 0.0))
        gt_cam_points = _build_camera_points(gt_poses, (0.0, 1.0, 0.0))
        if pred_cam_points.size == 0:
            camera_points = gt_cam_points
        elif gt_cam_points.size == 0:
            camera_points = pred_cam_points
        else:
            camera_points = np.vstack([pred_cam_points, gt_cam_points]).astype(np.float32, copy=False)

        pred_traj = self.pose_wireframe_builder.build_trajectory_vertices(pred_poses, (1.0, 0.0, 0.0))
        gt_traj = self.pose_wireframe_builder.build_trajectory_vertices(gt_poses, (0.0, 1.0, 0.0))
        return camera_points, pred_traj, gt_traj

    def _update_pose_camera_vbo(self, camera_points: np.ndarray):
        if self.pose_camera_vao is None:
            self._initialize_pose_rendering()

        self.pose_camera_count = int(camera_points.shape[0])
        if self.pose_camera_count == 0:
            return

        bytes_required = int(camera_points.nbytes)
        glBindBuffer(GL_ARRAY_BUFFER, self.pose_camera_vbo)
        if bytes_required > self.pose_camera_capacity_bytes:
            self.pose_camera_capacity_bytes = bytes_required
            glBufferData(GL_ARRAY_BUFFER, self.pose_camera_capacity_bytes, camera_points, GL_DYNAMIC_DRAW)
        else:
            glBufferSubData(GL_ARRAY_BUFFER, 0, bytes_required, camera_points)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _update_pose_traj_vbo(self, pred_traj: np.ndarray, gt_traj: np.ndarray):
        if self.pose_traj_vao is None:
            self._initialize_pose_rendering()

        self.pose_pred_traj_count = int(pred_traj.shape[0])
        self.pose_gt_traj_count = int(gt_traj.shape[0])

        if self.pose_pred_traj_count == 0 and self.pose_gt_traj_count == 0:
            return

        merged = pred_traj
        if self.pose_pred_traj_count == 0:
            merged = gt_traj
        elif self.pose_gt_traj_count > 0:
            merged = np.vstack([pred_traj, gt_traj]).astype(np.float32, copy=False)

        bytes_required = int(merged.nbytes)
        glBindBuffer(GL_ARRAY_BUFFER, self.pose_traj_vbo)
        if bytes_required > self.pose_traj_capacity_bytes:
            self.pose_traj_capacity_bytes = bytes_required
            glBufferData(GL_ARRAY_BUFFER, self.pose_traj_capacity_bytes, merged, GL_DYNAMIC_DRAW)
        else:
            glBufferSubData(GL_ARRAY_BUFFER, 0, bytes_required, merged)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _maybe_refresh_pose_overlay(self):
        pred_count = len(getattr(self.tracker, "keyframes_poses", [])) if self.tracker is not None else 0
        gt_count = 0
        if self.dataset is not None and getattr(self.dataset, "gt_poses", None) is not None:
            gt_count = int(np.asarray(self.dataset.gt_poses).shape[0])
        kf_idx_count = len(getattr(self.tracker, "keyframe_frame_indices", [])) if self.tracker is not None else 0
        last_pred_t = None
        if pred_count > 0:
            lp = np.asarray(getattr(self.tracker, "keyframes_poses", [])[-1], dtype=np.float32)
            if lp.shape == (4, 4):
                last_pred_t = tuple(lp[:3, 3].tolist())
        sig = (pred_count, gt_count, kf_idx_count, last_pred_t)

        if (not self._pose_dirty) and (self._pose_cache_sig == sig):
            return

        camera_points, pred_traj, gt_traj = self._build_pose_overlay_data()
        self._update_pose_camera_vbo(camera_points)
        self._update_pose_traj_vbo(pred_traj, gt_traj)
        self._pose_cache_sig = sig
        self._pose_dirty = False

    def render_keyframe_poses(self):
        self._maybe_refresh_pose_overlay()
        if self.pose_camera_count == 0 and self.pose_pred_traj_count == 0 and self.pose_gt_traj_count == 0:
            return
        self.camera.update_view()

        # Draw camera wireframes expanded from world-space pose positions in geometry shader.
        if self.pose_camera_count > 0:
            res.camera_point_shader.use()
            glUniformMatrix4fv(
                glGetUniformLocation(res.camera_point_shader.program, 'u_view'),
                1,
                GL_FALSE,
                self.camera.view,
            )
            glUniformMatrix4fv(
                glGetUniformLocation(res.camera_point_shader.program, 'u_projection'),
                1,
                GL_FALSE,
                self.camera.projection,
            )
            glUniform1f(
                glGetUniformLocation(res.camera_point_shader.program, 'u_cam_scale'),
                float(self.pose_wireframe_builder.camera_scale),
            )
            glBindVertexArray(self.pose_camera_vao)
            glLineWidth(1.5)
            glDrawArrays(GL_POINTS, 0, self.pose_camera_count)
            glBindVertexArray(0)

        # Draw trajectory polylines between successive poses.
        if self.pose_pred_traj_count > 1 or self.pose_gt_traj_count > 1:
            res.line_shader.use()
            glUniformMatrix4fv(
                glGetUniformLocation(res.line_shader.program, 'u_view'),
                1,
                GL_FALSE,
                self.camera.view,
            )
            glUniformMatrix4fv(
                glGetUniformLocation(res.line_shader.program, 'u_projection'),
                1,
                GL_FALSE,
                self.camera.projection,
            )
            glBindVertexArray(self.pose_traj_vao)
            glLineWidth(2.0)
            if self.pose_pred_traj_count > 1:
                glDrawArrays(GL_LINE_STRIP, 0, self.pose_pred_traj_count)
            if self.pose_gt_traj_count > 1:
                glDrawArrays(GL_LINE_STRIP, self.pose_pred_traj_count, self.pose_gt_traj_count)
            glBindVertexArray(0)

    def _initialize_pcd_rendering(self):
        # UPDATED: Check for all_sh instead of all_colors
        if self._initialized or self.pcd.all_points is None or self.pcd.all_points.numel() == 0:
            return

        # Create VBO
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        # Allocate space for (Position: 3 floats + SH0: 3 floats) * num_points
        self.vbo_capacity_bytes = (
            self.pcd.all_points.shape[0] * 6 * 4
        )
        glBufferData(GL_ARRAY_BUFFER, self.vbo_capacity_bytes, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # Setup VAO
        self.vao = glGenVertexArrays(1)
        stride = 6 * 4  # 6 floats * 4 bytes
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        # position (location = 0)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))

        # color/sh0 (location = 1)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))

        glBindVertexArray(0)
        self._initialized = True

    def set_use_segmentation_colors(self, enabled: bool):
        self.use_segmentation_colors = bool(enabled)

    def set_show_instance_bboxes(self, enabled: bool):
        self.show_instance_bboxes = bool(enabled)

    def set_segmentation_class_filter(self, class_id):
        if class_id is None:
            self.segmentation_class_filter = None
            return

        if isinstance(class_id, (list, tuple, set)):
            filt = set()
            for v in class_id:
                try:
                    filt.add(int(v))
                except Exception:
                    continue
            self.segmentation_class_filter = filt if len(filt) > 0 else None
            return

        try:
            self.segmentation_class_filter = {int(class_id)}
        except Exception:
            self.segmentation_class_filter = None

    def update_vbo(self, positions, color_coeffs):
        # The arguments passed from render_pcd
        # are already detached/referenced under the lock there.
        if positions is None or positions.numel() == 0:
            return

        # Convert to numpy under the assumption these are consistent snapshots
        color_data = color_coeffs.detach().cpu().numpy().astype(np.float32)
        positions_data = positions.detach().cpu().numpy().astype(np.float32)
        
        # This will now succeed because they were grabbed under the lock together
        interleaved = np.hstack([positions_data, color_data])
        required_bytes = interleaved.nbytes

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        if required_bytes > self.vbo_capacity_bytes:
            self.vbo_capacity_bytes = required_bytes
            glBufferData(GL_ARRAY_BUFFER, self.vbo_capacity_bytes, interleaved, GL_DYNAMIC_DRAW)
        else:
            glBufferSubData(GL_ARRAY_BUFFER, 0, required_bytes, interleaved)
        
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        self.pcd_added_size = positions.shape[0]

    def Render(self):
        self.fb.bind()
        glViewport(0, 0, self.fb.width, self.fb.height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.08, 0.1, 0.13, 1.0)
        self.render_pcd()
        self.render_keyframe_poses()
        self.render_instance_bboxes()
        self.fb.unbind()

    def render_pcd(self):
        # 1. Thread-safe check and data extraction
        with self.pcd.lock:
            # Skip if data isn't ready
            if self.pcd.all_points is None or self.pcd.all_sh is None:
                return

            points_to_draw = self.pcd.all_points
            seg_version = int(getattr(self.pcd, "segmentation_version", 0))
            use_seg = bool(self.use_segmentation_colors)
            seg_filter = self.segmentation_class_filter
            seg_filter_key = tuple(sorted(seg_filter)) if seg_filter else None

            color_coeffs = self.pcd.all_sh[:, 0, :]
            if use_seg:
                seg_logits = getattr(self.pcd, "segmentation_color_logits", None)
                if seg_logits is not None and int(seg_logits.shape[0]) == int(points_to_draw.shape[0]):
                    color_coeffs = seg_logits

            if seg_filter:
                seg_labels = getattr(self.pcd, "segmentation_labels", None)
                if seg_labels is not None and int(seg_labels.shape[0]) == int(points_to_draw.shape[0]):
                    keep = torch.zeros_like(seg_labels, dtype=torch.bool)
                    for cls_id in seg_filter:
                        keep |= (seg_labels == int(cls_id))
                    points_to_draw = points_to_draw[keep]
                    color_coeffs = color_coeffs[keep]

            current_count = int(points_to_draw.shape[0])
            
            # 2. Check if we need to update the OpenGL buffers
            # Only run the heavy upload if the point count changed
            if not self._initialized:
                self._initialize_pcd_rendering()
                self.update_vbo(points_to_draw, color_coeffs)
            elif (
                self.pcd_added_size != current_count
                or self._last_color_mode != use_seg
                or (use_seg and self._last_segmentation_version != seg_version)
                or self._last_segmentation_filter != seg_filter_key
            ):
                # We pass the tensors directly while inside the lock
                self.update_vbo(points_to_draw, color_coeffs)

            self._last_color_mode = use_seg
            self._last_segmentation_version = seg_version
            self._last_segmentation_filter = seg_filter_key

        # 3. Standard OpenGL Drawing (Outside the lock to keep it fast)
        res.simple_shader.use()
        self.camera.update_view()
        
        glUniformMatrix4fv(
            glGetUniformLocation(res.simple_shader.program, 'u_view'),
            1, GL_FALSE, self.camera.view           
        )
        glUniformMatrix4fv(
            glGetUniformLocation(res.simple_shader.program, 'u_projection'),
            1, GL_FALSE, self.camera.projection           
        )

        glBindVertexArray(self.vao)
        glDrawArrays(GL_POINTS, 0, self.pcd_added_size)
        glBindVertexArray(0)

    def cleanup(self):
        """Release resources"""
        if self._initialized:
            glDeleteBuffers(1, [self.vbo])
            glDeleteVertexArrays(1, [self.vao])
        if self.pose_camera_vbo is not None:
            glDeleteBuffers(1, [self.pose_camera_vbo])
            self.pose_camera_vbo = None
        if self.pose_camera_vao is not None:
            glDeleteVertexArrays(1, [self.pose_camera_vao])
            self.pose_camera_vao = None
        if self.pose_traj_vbo is not None:
            glDeleteBuffers(1, [self.pose_traj_vbo])
            self.pose_traj_vbo = None
        if self.pose_traj_vao is not None:
            glDeleteVertexArrays(1, [self.pose_traj_vao])
            self.pose_traj_vao = None
        if self.instance_bbox_vbo is not None:
            glDeleteBuffers(1, [self.instance_bbox_vbo])
            self.instance_bbox_vbo = None
        if self.instance_bbox_vao is not None:
            glDeleteVertexArrays(1, [self.instance_bbox_vao])
            self.instance_bbox_vao = None

    def on_resize(self):
        self.camera.update_projection(self.fb)