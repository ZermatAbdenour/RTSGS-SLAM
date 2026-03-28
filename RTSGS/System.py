import threading
import os
from collections import deque
from RTSGS.GaussianSplatting.GaussianSplating import GaussianSplatting
from RTSGS.GUI.WindowManager import WindowManager
from RTSGS.GaussianSplatting.PointCloud import PointCloud
from RTSGS.DataLoader.DataLoader import DataLoader
from RTSGS.Tracker.Tracker import Tracker
from RTSGS.Segmentation.YOLOSegmenter import YOLOSemanticSegmenter

import cv2
import numpy as np

class RTSGSSystem:
    def __init__(self, dataset: DataLoader, tracker: Tracker, config):
        self.dataset = dataset
        self.tracker = tracker

        self._stop = False
        self._busy = False
        self._pending = None

        self._cv = threading.Condition()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._seg_cv = threading.Condition()
        self._seg_stop = False
        self._seg_pending = deque()
        self._seg_worker = threading.Thread(target=self._segmenter_loop, daemon=True)

        # Define the point cloud and GS engine
        self.pcd = PointCloud(config)
        self.gs = GaussianSplatting(self.pcd, self.dataset, self.tracker)
        if hasattr(self.tracker, "set_rendered_depth_provider"):
            self.tracker.set_rendered_depth_provider(self.gs.render_depth_at_pose)
        if hasattr(self.pcd, "set_rendered_depth_provider"):
            self.pcd.set_rendered_depth_provider(self.gs.render_depth_at_pose)
        self.window = WindowManager(
            self.pcd,
            self.gs,
            tracker=self.tracker,
            dataset=self.dataset,
            width=1280,
            height=720,
            title="RTSGS System",
        )

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.segmenter = YOLOSemanticSegmenter(self.pcd, config, project_root)

        # Track the last keyframe index added to the map
        self.last_added_keyframe_idx = -1


    def run(self):
        self._worker.start()
        self._seg_worker.start()
        self.segmenter.start()

        while not self.window.window_should_close():
            self.window.start_frame()

            # 1. Process streaming data and track
            frame = self.process_stream_frame()
            
            # 2. Run a training step (Optimization)
            # This is now throttled internally by your max_steps_per_sec logic
            self.gs.training_step()

            # 3. Asynchronous Map Update
            # Process the NEXT pending keyframe (one at a time, in order)
            next_kf = self.last_added_keyframe_idx + 1
            
            if next_kf < self.dataset.current_keyframe_index:
                success = self.pcd.update_async(
                    self.dataset.rgb_keyframes[next_kf],
                    self.dataset.depth_keyframes[next_kf],
                    self.tracker.keyframes_poses[next_kf],
                    None,
                )
                     
                if success:
                    # Queue semantic fusion asynchronously in the segmenter worker.
                    with self._seg_cv:
                        self._seg_pending.append(
                            (
                                self.dataset.rgb_keyframes[next_kf],
                                self.dataset.depth_keyframes[next_kf],
                                self.tracker.keyframes_poses[next_kf],
                            )
                        )
                        self._seg_cv.notify()
                    print(f"Update triggered for keyframe: {next_kf}")
                    self.last_added_keyframe_idx = next_kf

            # 4. Visualization and GUI
            self.tracker.visualize_tracking()
            self.window.render_frame()

        # Shutdown sequence
        with self._cv:
            self._stop = True
            self._cv.notify_all()

        with self._seg_cv:
            self._seg_stop = True
            self._seg_cv.notify_all()

        self._worker.join(timeout=1.0)
        self._seg_worker.join(timeout=1.0)
        self.segmenter.stop()
        self.window.shutdown()

    def process_stream_frame(self):
        # Check busy without grabbing frames/decoding when we will skip anyway
        with self._cv:
            if self._busy:
                return None

        frame_paths = self.dataset.get_next_frame()
        if frame_paths is None:
            return None
        
        # Read color and depth img
        rgb = cv2.imread(frame_paths[0], cv2.IMREAD_COLOR)
        depth = cv2.imread(frame_paths[1], cv2.IMREAD_UNCHANGED).astype(np.float32)
        
        with self._cv:
            self._pending = (rgb, depth)
            self._busy = True
            self._cv.notify()
        return (rgb, depth)

    def _worker_loop(self):
        while True:
            with self._cv:
                while not self._stop and self._pending is None:
                    self._cv.wait()

                if self._stop:
                    return

                img, depth = self._pending
                self._pending = None

            try:
                # The tracker updates dataset.current_keyframe_index internally 
                # when a new keyframe is detected.
                self.tracker.track_frame(img, depth)
            finally:
                with self._cv:
                    self._busy = False

    def _segmenter_loop(self):
        while True:
            with self._seg_cv:
                while not self._seg_stop and len(self._seg_pending) == 0:
                    self._seg_cv.wait()

                if self._seg_stop and len(self._seg_pending) == 0:
                    return

                rgb, depth, pose = self._seg_pending.popleft()

            try:
                self.segmenter.process_frame(rgb, depth, pose)
            except Exception as e:
                print(f"Segmenter worker error: {e}")