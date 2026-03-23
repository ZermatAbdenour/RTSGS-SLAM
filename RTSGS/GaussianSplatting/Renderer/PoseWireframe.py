import numpy as np


class PoseWireframeBuilder:
    """Build line-list vertices for camera wireframe visualization."""

    def __init__(self, camera_scale: float = 0.08, aspect: float = 0.75):
        self.camera_scale = float(camera_scale)
        self.aspect = float(aspect)

    def camera_edges_local(self) -> np.ndarray:
        s = self.camera_scale
        w = s
        h = s * self.aspect
        z = s * 1.5

        o = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        p0 = np.array([-w, -h, z], dtype=np.float32)
        p1 = np.array([w, -h, z], dtype=np.float32)
        p2 = np.array([w, h, z], dtype=np.float32)
        p3 = np.array([-w, h, z], dtype=np.float32)

        # 8 edges => 16 vertices (GL_LINES)
        return np.stack(
            [
                o,
                p0,
                o,
                p1,
                o,
                p2,
                o,
                p3,
                p0,
                p1,
                p1,
                p2,
                p2,
                p3,
                p3,
                p0,
            ],
            axis=0,
        )

    @staticmethod
    def _transform_points(T: np.ndarray, points_local: np.ndarray) -> np.ndarray:
        R = T[:3, :3].astype(np.float32, copy=False)
        t = T[:3, 3].astype(np.float32, copy=False)
        return (R @ points_local.T).T + t

    def build_vertices(self, poses: list[np.ndarray], color_rgb: tuple[float, float, float]) -> np.ndarray:
        if not poses:
            return np.empty((0, 6), dtype=np.float32)

        local_edges = self.camera_edges_local()
        color = np.array(color_rgb, dtype=np.float32)[None, :]
        chunks = []
        for T in poses:
            if T is None:
                continue
            T = np.asarray(T, dtype=np.float32)
            if T.shape != (4, 4) or not np.isfinite(T).all():
                continue
            world = self._transform_points(T, local_edges)
            colors = np.repeat(color, world.shape[0], axis=0)
            chunks.append(np.hstack([world, colors]).astype(np.float32, copy=False))

        if not chunks:
            return np.empty((0, 6), dtype=np.float32)
        return np.vstack(chunks)

    @staticmethod
    def build_trajectory_vertices(poses: list[np.ndarray], color_rgb: tuple[float, float, float]) -> np.ndarray:
        if not poses:
            return np.empty((0, 6), dtype=np.float32)
        points = []
        for T in poses:
            if T is None:
                continue
            T = np.asarray(T, dtype=np.float32)
            if T.shape != (4, 4) or not np.isfinite(T).all():
                continue
            points.append(T[:3, 3])
        if len(points) == 0:
            return np.empty((0, 6), dtype=np.float32)
        pts = np.asarray(points, dtype=np.float32)
        col = np.array(color_rgb, dtype=np.float32)[None, :]
        cols = np.repeat(col, pts.shape[0], axis=0)
        return np.hstack([pts, cols]).astype(np.float32, copy=False)
