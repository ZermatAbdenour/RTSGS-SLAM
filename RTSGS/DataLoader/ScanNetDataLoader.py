import os
import numpy as np

from RTSGS.DataLoader.DataLoader import DataLoader


class ScanNetDataLoader(DataLoader):
    """ScanNet loader with explicit RGB/depth calibration support."""

    def __init__(self, scene_extracted_path: str, config, trajectory_path: str = None, fps: float = 1.0):
        self._scene_extracted_path = scene_extracted_path
        self._intrinsic_dir = os.path.join(scene_extracted_path, "intrinsic")
        self._trajectory_path = trajectory_path
        self._pose_dir = trajectory_path if trajectory_path is not None else os.path.join(scene_extracted_path, "pose")

        rgb_path = os.path.join(scene_extracted_path, "color")
        depth_path = os.path.join(scene_extracted_path, "depth")
        super().__init__(rgb_path, depth_path)

        self.config = config
        self._fps = float(fps)

        self.gt_poses = None
        self.gt_timestamps = None

    @staticmethod
    def _load_mat4(path: str):
        if not os.path.isfile(path):
            return None
        vals = []
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                vals.extend(float(v) for v in parts)
        if len(vals) != 16:
            return None
        return np.asarray(vals, dtype=np.float32).reshape(4, 4)

    @staticmethod
    def _extract_number(filename: str) -> int:
        stem, _ = os.path.splitext(filename)
        return int(stem)

    def _read_calibration(self):
        k_color_path = os.path.join(self._intrinsic_dir, "intrinsic_color.txt")
        k_depth_path = os.path.join(self._intrinsic_dir, "intrinsic_depth.txt")
        ext_color_path = os.path.join(self._intrinsic_dir, "extrinsic_color.txt")
        ext_depth_path = os.path.join(self._intrinsic_dir, "extrinsic_depth.txt")

        K_color_4 = self._load_mat4(k_color_path)
        K_depth_4 = self._load_mat4(k_depth_path)
        E_color = self._load_mat4(ext_color_path)
        E_depth = self._load_mat4(ext_depth_path)

        if K_color_4 is None or K_depth_4 is None:
            raise RuntimeError("Missing ScanNet intrinsic files (intrinsic_color.txt / intrinsic_depth.txt)")

        K_color = K_color_4[:3, :3]
        K_depth = K_depth_4[:3, :3]

        T_depth_to_rgb = np.eye(4, dtype=np.float32)
        if E_color is not None and E_depth is not None:
            try:
                T_depth_to_rgb = np.linalg.inv(E_color).astype(np.float32) @ E_depth
            except np.linalg.LinAlgError:
                T_depth_to_rgb = np.eye(4, dtype=np.float32)

        return K_color, K_depth, T_depth_to_rgb

    def _configure_intrinsics(self):
        K_color, K_depth, T_depth_to_rgb = self._read_calibration()

        self.config.set("rgb_fx", float(K_color[0, 0]))
        self.config.set("rgb_fy", float(K_color[1, 1]))
        self.config.set("rgb_cx", float(K_color[0, 2]))
        self.config.set("rgb_cy", float(K_color[1, 2]))

        self.config.set("depth_fx", float(K_depth[0, 0]))
        self.config.set("depth_fy", float(K_depth[1, 1]))
        self.config.set("depth_cx", float(K_depth[0, 2]))
        self.config.set("depth_cy", float(K_depth[1, 2]))

        self.config.set("T_depth_to_rgb", T_depth_to_rgb.tolist())

        # ScanNet depth PNGs are in millimeters.
        self.config.set("depth_scale", 1000.0)

    def load_data(self, limit: int = -1):
        self._configure_intrinsics()

        if not os.path.isdir(self._pose_dir):
            raise RuntimeError(f"Missing ScanNet trajectory directory: {self._pose_dir}")

        rgb_files = [f for f in os.listdir(self._rgb_path) if f.lower().endswith(".jpg")]
        depth_files = [f for f in os.listdir(self._depth_path) if f.lower().endswith(".png")]
        pose_files = [f for f in os.listdir(self._pose_dir) if f.lower().endswith(".txt")]

        rgb_dict = {self._extract_number(f): f for f in rgb_files}
        depth_dict = {self._extract_number(f): f for f in depth_files}
        pose_dict = {self._extract_number(f): f for f in pose_files}

        keys = sorted(set(rgb_dict.keys()) & set(depth_dict.keys()) & set(pose_dict.keys()))
        if limit != -1:
            keys = keys[:limit]

        pairs = []
        poses = []

        for k in keys:
            pose_path = os.path.join(self._pose_dir, pose_dict[k])
            T = self._load_mat4(pose_path)
            if T is None or not np.isfinite(T).all():
                continue

            pairs.append(
                (
                    os.path.join(self._rgb_path, rgb_dict[k]),
                    os.path.join(self._depth_path, depth_dict[k]),
                )
            )
            poses.append(T)

        if not pairs:
            self.RGBD_pairs = []
            self.time_stamps = np.array([], dtype=np.float64)
            self.gt_poses = None
            self.gt_timestamps = None
            return

        dt = 1.0 / max(self._fps, 1e-6)
        ts = np.arange(len(pairs), dtype=np.float64) * dt

        self.RGBD_pairs = pairs
        self.time_stamps = ts
        self.gt_poses = np.stack(poses, axis=0).astype(np.float32)
        self.gt_timestamps = ts.copy()

        # Record image sizes from the first frame.
        import cv2

        rgb0 = cv2.imread(self.RGBD_pairs[0][0], cv2.IMREAD_COLOR)
        depth0 = cv2.imread(self.RGBD_pairs[0][1], cv2.IMREAD_UNCHANGED)
        if rgb0 is not None:
            h, w = rgb0.shape[:2]
            self.config.set("rgb_width", int(w))
            self.config.set("rgb_height", int(h))
        if depth0 is not None:
            h, w = depth0.shape[:2]
            self.config.set("depth_width", int(w))
            self.config.set("depth_height", int(h))

        print(f"Loaded {len(self.RGBD_pairs)} ScanNet RGB-D pairs.")
