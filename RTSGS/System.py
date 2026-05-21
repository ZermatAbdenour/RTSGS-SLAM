import threading
import os
import time
from collections import deque
from RTSGS.GaussianSplatting.GaussianSplating import GaussianSplatting
from RTSGS.GUI.WindowManager import WindowManager
from RTSGS.GaussianSplatting.PointCloud import PointCloud
from RTSGS.DataLoader.DataLoader import DataLoader
from RTSGS.Tracker.Tracker import Tracker
from RTSGS.Segmentation.Segmenter import YOLOSemanticSegmenter
from RTSGS.SceneGraph import RealtimeSceneGraphRuntime

import cv2
import numpy as np
import traceback

from RTSGS.Benchmarking.BenchmarkCollector import BenchmarkCollector

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
        self._seg_busy = False
        self._seg_last_total_ms = 0.0
        self._seg_last_kf_index = -1
        self._seg_processed_count = 0
        self._seg_last_error = ""
        self._seg_worker = threading.Thread(target=self._segmenter_loop, daemon=True)

        # Define the point cloud and GS engine
        self.pcd = PointCloud(config)
        self.gs = GaussianSplatting(self.pcd, self.dataset, self.tracker)
        if hasattr(self.tracker, "set_rendered_depth_provider"):
            self.tracker.set_rendered_depth_provider(self.gs.render_depth_at_pose)
        if hasattr(self.tracker, "set_rendered_rgb_provider"):
            self.tracker.set_rendered_rgb_provider(self.gs.render_rgb_at_pose)
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
        self.scene_graph = RealtimeSceneGraphRuntime(self.pcd, config, project_root)

        # Segmenter runtime status shown in GUI.
        with self.pcd.lock:
            self.pcd.segmentation_worker_busy = False
            self.pcd.segmentation_queue_size = 0
            self.pcd.segmentation_processed_count = 0
            self.pcd.segmentation_last_kf_index = -1
            self.pcd.segmentation_last_total_ms = 0.0
            self.pcd.segmentation_last_error = ""

        # Track the last keyframe index added to the map
        self.last_added_keyframe_idx = -1

        # Throttle semantic segmentation: run once every N keyframes.
        self.semantic_update_stride = max(1, int(config.segmentation.update_stride))
        self._last_enqueued_semantic_kf = -1

        # Benchmarking (enabled only when run(benchmark=True))
        self.benchmark: BenchmarkCollector | None = None
        self._trajectory_done = False
        self.stop_training_on_trajectory_end = True


    def run(self, benchmark: bool = False):
        self._worker.start()
        self._seg_worker.start()
        self.segmenter.start()
        self.scene_graph.start()

        # Enable benchmark window if requested
        if benchmark:
            self.benchmark = BenchmarkCollector(enabled=True)
            self.benchmark.reset_total_peak()
            # Expose the collector to the modules that can self-instrument.
            try:
                self.pcd.benchmark = self.benchmark
            except Exception:
                pass
            try:
                self.segmenter.benchmark = self.benchmark
            except Exception:
                pass
            try:
                self.scene_graph.benchmark = self.benchmark
            except Exception:
                pass
            try:
                self.window.enable_benchmark_window(self.gs, self.dataset, tracker=self.tracker, bench=self.benchmark)
            except Exception:
                pass

        while not self.window.window_should_close():
            self.window.start_frame()

            # 1. Process streaming data and track
            frame = self.process_stream_frame()

            # Mark end-of-trajectory once the dataset is exhausted.
            if benchmark and not self._trajectory_done and self._dataset_exhausted():
                self._trajectory_done = True
                if self.benchmark is not None:
                    self.benchmark.mark_trajectory_ended()
            
            # 2. Run a training step (Optimization)
            if not (self.stop_training_on_trajectory_end and self._trajectory_done):
                self.gs.training_step()

            # 3. Asynchronous Map Update
            # Drain as many pending keyframes as possible without waiting for the next GUI frame.
            while True:
                next_kf = self.last_added_keyframe_idx + 1
                if next_kf >= self.dataset.current_keyframe_index:
                    break

                success = self.pcd.update_async(
                    self.dataset.rgb_keyframes[next_kf],
                    self.dataset.depth_keyframes[next_kf],
                    self.tracker.keyframes_poses[next_kf],
                    None,
                )

                if not success:
                    break

                # Queue semantic fusion asynchronously in the segmenter worker.
                should_enqueue_semantic = (
                    self._last_enqueued_semantic_kf < 0
                    or (int(next_kf) - int(self._last_enqueued_semantic_kf)) >= int(self.semantic_update_stride)
                )
                if should_enqueue_semantic:
                    with self._seg_cv:
                        self._seg_pending.append(
                            (
                                self.dataset.rgb_keyframes[next_kf],
                                self.dataset.depth_keyframes[next_kf],
                                self.tracker.keyframes_poses[next_kf],
                                int(next_kf),
                            )
                        )
                        q_size = int(len(self._seg_pending))
                        self._seg_cv.notify()
                    self._last_enqueued_semantic_kf = int(next_kf)
                else:
                    with self._seg_cv:
                        q_size = int(len(self._seg_pending))

                with self.pcd.lock:
                    self.pcd.segmentation_queue_size = q_size

                print(f"Update triggered for keyframe: {next_kf}")
                self.last_added_keyframe_idx = next_kf

            # 4. Visualization and GUI
            self.tracker.visualize_tracking()
            self.window.render_frame()

            # Finalize the benchmark once the stream is done and workers drained.
            if benchmark and self.benchmark is not None:
                self._maybe_finalize_benchmark()

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
        self.scene_graph.stop()
        self.window.shutdown()

    def process_stream_frame(self):
        # Check busy without grabbing frames/decoding when we will skip anyway
        with self._cv:
            if self._busy:
                return None

        frame_paths = self.dataset.get_next_frame()
        if frame_paths is None:
            return None
        
        # Read color and depth img (robust to missing/unreadable frames)
        rgb_path = frame_paths[0]
        depth_path = frame_paths[1] if len(frame_paths) > 1 else None

        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb is None:
            print(f"[System] Warning: failed to read RGB: {rgb_path}")
            return None

        depth = None
        if depth_path is not None:
            depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_raw is None:
                print(f"[System] Warning: failed to read depth: {depth_path}")
                return None
            if depth_raw.ndim == 3:
                depth_raw = depth_raw[..., 0]
            depth = depth_raw.astype(np.float32)
        
        with self._cv:
            self._pending = (rgb, depth)
            self._busy = True
            self._cv.notify()
        return (rgb, depth)

    def _dataset_exhausted(self) -> bool:
        """True when the streaming loader has served all frames."""
        try:
            pairs = getattr(self.dataset, "RGBD_pairs", None)
            cur = int(getattr(self.dataset, "current_frame_index", -1))
            if pairs is None or cur < 0:
                return False
            return cur >= int(len(pairs))
        except Exception:
            return False

    def _maybe_finalize_benchmark(self) -> None:
        """Finalize benchmark numbers once no background work remains."""
        b = self.benchmark
        if b is None or not b.enabled or b.finalized:
            return
        if not self._trajectory_done:
            return

        # Wait for tracker worker, mapping executor, and segmenter queue to drain.
        with self._cv:
            tracker_idle = (not self._busy) and (self._pending is None)
        with self._seg_cv:
            seg_queue_empty = len(self._seg_pending) == 0
            seg_idle = (not self._seg_busy)

        mapping_drained = True
        try:
            next_kf = int(self.last_added_keyframe_idx) + 1
            mapping_drained = next_kf >= int(getattr(self.dataset, "current_keyframe_index", 0))
        except Exception:
            mapping_drained = True

        pcd_idle = True
        try:
            pcd_idle = not bool(getattr(self.pcd, "is_processing", False))
        except Exception:
            pcd_idle = True

        if tracker_idle and seg_queue_empty and seg_idle and mapping_drained and pcd_idle:
            b.finalize(pcd=self.pcd, segmenter=self.segmenter, scene_graph=self.scene_graph)

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
                if self.benchmark is not None and self.benchmark.enabled:
                    t0 = time.perf_counter()
                    try:
                        self.tracker.track_frame(img, depth)
                    finally:
                        dt_ms = (time.perf_counter() - t0) * 1000.0
                        self.benchmark.record_ms(BenchmarkCollector.STAGE_TRACKING, dt_ms)
                        try:
                            self.benchmark.observe_total_peak()
                        except Exception:
                            pass
                else:
                    self.tracker.track_frame(img, depth)
            except Exception:
                # Never let the worker thread die silently; otherwise _busy can
                # get stuck true and the stream appears frozen.
                print("[System] Tracker worker exception:\n" + traceback.format_exc())
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

                rgb, depth, pose, kf_index = self._seg_pending.popleft()
                self._seg_busy = True
                q_size = int(len(self._seg_pending))

            with self.pcd.lock:
                self.pcd.segmentation_worker_busy = True
                self.pcd.segmentation_queue_size = q_size

            try:
                t0 = time.perf_counter()
                seg_result = self.segmenter.process_frame(rgb, depth, pose)
                try:
                    self.scene_graph.update_from_segmenter(seg_result, int(kf_index))
                except Exception as scene_e:
                    with self.pcd.lock:
                        self.pcd.scene_graph_last_error = str(scene_e)
                    print(f"Scene graph worker error: {scene_e}")
                self._seg_last_total_ms = float((time.perf_counter() - t0) * 1000.0)
                if self.benchmark is not None and self.benchmark.enabled:
                    self.benchmark.record_ms(BenchmarkCollector.STAGE_END_TO_END, float(self._seg_last_total_ms))
                self._seg_last_kf_index = int(kf_index)
                self._seg_processed_count += 1
                self._seg_last_error = ""
            except Exception as e:
                self._seg_last_error = str(e)
                print(f"Segmenter worker error: {e}")
            finally:
                with self._seg_cv:
                    self._seg_busy = False
                    q_size = int(len(self._seg_pending))

                with self.pcd.lock:
                    self.pcd.segmentation_worker_busy = bool(self._seg_busy)
                    self.pcd.segmentation_queue_size = q_size
                    self.pcd.segmentation_processed_count = int(self._seg_processed_count)
                    self.pcd.segmentation_last_kf_index = int(self._seg_last_kf_index)
                    self.pcd.segmentation_last_total_ms = float(self._seg_last_total_ms)
                    self.pcd.segmentation_last_error = str(self._seg_last_error)