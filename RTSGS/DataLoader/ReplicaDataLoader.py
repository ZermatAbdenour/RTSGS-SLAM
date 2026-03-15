import os
import numpy as np
from RTSGS.DataLoader.DataLoader import DataLoader

class ReplicaDataLoader(DataLoader):
    def __init__(self, data_path, trajectory_path=None, fps=10):
        super().__init__(data_path, data_path)
        self._trajectory_path = trajectory_path
        self._fps = fps

        self.gt_poses = None
        self.gt_timestamps = None
        self.time_stamps = None
        self.RGBD_pairs = None

    @staticmethod
    def _load_trajectory_file(traj_path):
        if traj_path is None:
            return None
        poses = []
        with open(traj_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = list(map(float, line.split()))
                if len(parts) != 16:
                    continue
                T = np.array(parts, dtype=np.float32).reshape((4,4))
                poses.append(T)
        if not poses:
            return None
        return np.stack(poses, axis=0)

    @staticmethod
    def _extract_number(filename):
        return int(''.join(filter(str.isdigit, filename)))

    def load_data(self, limit=-1):
        # List RGB and depth files
        rgb_files = [f for f in os.listdir(self._rgb_path) if f.startswith("frame") and f.endswith(".jpg")]
        depth_files = [f for f in os.listdir(self._depth_path) if f.startswith("depth") and f.endswith(".png")]

        if not rgb_files or not depth_files:
            print("Error: No RGB or Depth files found!")
            self.RGBD_pairs = []
            self.time_stamps = np.array([])
            self.gt_poses = None
            self.gt_timestamps = None
            return

        # Build a dict for matching by frame number
        depth_dict = {self._extract_number(f): f for f in depth_files}
        rgb_dict = {self._extract_number(f): f for f in rgb_files}

        common_keys = sorted(set(depth_dict.keys()) & set(rgb_dict.keys()))
        if limit != -1:
            common_keys = common_keys[:limit]

        pairs = []
        for k in common_keys:
            pairs.append((
                os.path.join(self._rgb_path, rgb_dict[k]),
                os.path.join(self._depth_path, depth_dict[k])
            ))

        # Synthetic timestamps
        dt = 1.0 / self._fps
        ts = np.arange(len(pairs), dtype=np.float64) * dt

        # Load trajectory
        gt_poses = self._load_trajectory_file(self._trajectory_path)
        if gt_poses is not None:
            if len(gt_poses) != len(pairs):
                print(f"Warning: trajectory length ({len(gt_poses)}) "
                      f"does not match number of frames ({len(pairs)}). Truncating to min length.")
                min_len = min(len(gt_poses), len(pairs))
                gt_poses = gt_poses[:min_len]
                pairs = pairs[:min_len]

        self.RGBD_pairs = pairs
        self.time_stamps = ts
        self.gt_poses = gt_poses
        self.gt_timestamps = ts if gt_poses is not None else None

        print(f"Loaded {len(pairs)} RGB-D pairs.")
        if gt_poses is not None:
            print(f"Loaded {len(gt_poses)} ground truth poses.")