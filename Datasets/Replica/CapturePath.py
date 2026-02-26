import os
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import imageio.v2 as imageio

import habitat_sim
import magnum as mn
from habitat_sim.agent import AgentConfiguration
from habitat_sim.utils.common import quat_from_coeffs

from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


# =============================================================================
# USER CONFIG
# =============================================================================
SENSOR_HEIGHT_M = 1.6
SCENE_ROT_X_DEG = 90.0


def normalize_quat_xyzw(q: np.ndarray) -> np.ndarray:
    q = q.astype(np.float32)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([0, 0, 0, 1], dtype=np.float32)
    return (q / n).astype(np.float32)


def enforce_quat_continuity_xyzw(quats: np.ndarray) -> np.ndarray:
    q = quats.copy().astype(np.float32)
    q[0] = normalize_quat_xyzw(q[0])
    for i in range(1, len(q)):
        q[i] = normalize_quat_xyzw(q[i])
        if float(np.dot(q[i], q[i - 1])) < 0.0:
            q[i] *= -1.0
    return q


def habitat_wc_quat_to_opencv_wc_quat(q_wc_hab_xyzw: np.ndarray) -> np.ndarray:
    R_wc_hab = R.from_quat(q_wc_hab_xyzw).as_matrix().astype(np.float32)
    R_chab__cocv = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    R_wc_ocv = R_wc_hab @ R_chab__cocv
    return R.from_matrix(R_wc_ocv).as_quat().astype(np.float32)


def process_rgb(rgb_rgba_or_rgb: np.ndarray, flip_vertical: bool) -> np.ndarray:
    img = rgb_rgba_or_rgb
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    if flip_vertical:
        img = np.ascontiguousarray(img[::-1, :, :])
    return img


def estimate_bounds_from_navmesh(sim: habitat_sim.Simulator, num_samples: int = 25000):
    pts = []
    for _ in range(int(num_samples)):
        p = sim.pathfinder.get_random_navigable_point()
        pts.append([p[0], p[1], p[2]])
    pts = np.asarray(pts, dtype=np.float32)
    bmin = np.min(pts, axis=0).astype(np.float32)
    bmax = np.max(pts, axis=0).astype(np.float32)
    return bmin, bmax


def rotation_matrix_x(deg: float) -> np.ndarray:
    return R.from_euler("x", float(deg), degrees=True).as_matrix().astype(np.float32)


def rotate_points(points: np.ndarray, R3: np.ndarray) -> np.ndarray:
    if points.ndim == 1:
        return (R3 @ points.reshape(3, 1)).reshape(3).astype(np.float32)
    return (points @ R3.T).astype(np.float32)


def rotate_aabb(bounds_min: np.ndarray, bounds_max: np.ndarray, R3: np.ndarray):
    corners = np.array(
        [
            [bounds_min[0], bounds_min[1], bounds_min[2]],
            [bounds_min[0], bounds_min[1], bounds_max[2]],
            [bounds_min[0], bounds_max[1], bounds_min[2]],
            [bounds_min[0], bounds_max[1], bounds_max[2]],
            [bounds_max[0], bounds_min[1], bounds_min[2]],
            [bounds_max[0], bounds_min[1], bounds_max[2]],
            [bounds_max[0], bounds_max[1], bounds_min[2]],
            [bounds_max[0], bounds_max[1], bounds_max[2]],
        ],
        dtype=np.float32,
    )
    rc = rotate_points(corners, R3)
    return np.min(rc, axis=0).astype(np.float32), np.max(rc, axis=0).astype(np.float32)


def normalize(v: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)


def project_to_plane(v: np.ndarray, n_unit: np.ndarray) -> np.ndarray:
    return (v - np.dot(v, n_unit) * n_unit).astype(np.float32)


def make_sim_and_agent(dataset_config, scene_name, width, height, hfov_deg, min_depth, max_depth, sensor_height_m: float):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = dataset_config
    sim_cfg.scene_id = scene_name
    sim_cfg.enable_physics = False

    sensor_specs = []

    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [height, width]
    rgb.position = mn.Vector3(0.0, float(sensor_height_m), 0.0)
    rgb.hfov = float(hfov_deg)
    sensor_specs.append(rgb)

    depth = habitat_sim.CameraSensorSpec()
    depth.uuid = "depth"
    depth.sensor_type = habitat_sim.SensorType.DEPTH
    depth.resolution = [height, width]
    depth.position = mn.Vector3(0.0, float(sensor_height_m), 0.0)
    depth.hfov = float(hfov_deg)
    depth.min_depth = float(min_depth)
    depth.max_depth = float(max_depth)
    depth.normalize_depth = False
    sensor_specs.append(depth)

    agent_cfg = AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs

    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    agent = sim.get_agent(0)
    return sim, agent


class FullBoundsTourGenerator:
    def __init__(self, bounds_min, bounds_max, scene_rot_x_deg=0.0):
        self.scene_rot_x_deg = float(scene_rot_x_deg)
        self.R_scene = rotation_matrix_x(self.scene_rot_x_deg)
        self.bmin, self.bmax = rotate_aabb(bounds_min, bounds_max, self.R_scene)
        self.center = ((self.bmin + self.bmax) / 2.0).astype(np.float32)

        self.up = normalize(rotate_points(np.array([0.0, 1.0, 0.0], dtype=np.float32), self.R_scene))
        u0 = project_to_plane(np.array([1.0, 0.0, 0.0], dtype=np.float32), self.up)
        if np.linalg.norm(u0) < 1e-4:
            u0 = project_to_plane(np.array([0.0, 0.0, 1.0], dtype=np.float32), self.up)
        self.u = normalize(u0)
        self.v = normalize(np.cross(self.up, self.u).astype(np.float32))

    def _create_bspline_path(self, keypoints, num_frames):
        distances = np.cumsum(np.sqrt(np.sum(np.diff(keypoints, axis=0) ** 2, axis=1)))
        distances = np.insert(distances, 0, 0)
        t_norm = distances / distances[-1]
        t_new = np.linspace(0, 1, num_frames)

        positions = np.zeros((num_frames, 3), dtype=np.float32)
        for i in range(3):
            spline = CubicSpline(t_norm, keypoints[:, i], bc_type="natural")
            positions[:, i] = spline(t_new).astype(np.float32)
        return positions

    def _smooth_positions(self, positions):
        sm = positions.copy()
        for i in range(3):
            sm[:, i] = gaussian_filter1d(sm[:, i], sigma=1.0)
            window = min(25, max(7, (len(sm) // 20) | 1))
            if window >= 7 and window % 2 == 1:
                sm[:, i] = savgol_filter(sm[:, i], window_length=window, polyorder=3)
        return sm.astype(np.float32)

    def _smooth_quaternions(self, quaternions):
        num_frames = len(quaternions)
        if num_frames < 10:
            return quaternions
        rotations = R.from_quat(quaternions)
        rotvecs = rotations.as_rotvec()
        sm = np.zeros_like(rotvecs)
        for k in range(3):
            sm[:, k] = gaussian_filter1d(rotvecs[:, k], sigma=1.5)
        sm_rots = R.from_rotvec(sm)
        key_int = max(1, num_frames // 25)
        key_idx = list(range(0, num_frames, key_int))
        if key_idx[-1] != num_frames - 1:
            key_idx.append(num_frames - 1)
        slerp = Slerp(key_idx, sm_rots[key_idx])
        final_rots = slerp(np.arange(num_frames))
        return final_rots.as_quat().astype(np.float32)

    def compute_orientations_old(self, positions, look_at_points):
        quats = np.zeros((len(positions), 4), dtype=np.float32)
        for i in range(len(positions)):
            forward = (look_at_points[i] - positions[i]).astype(np.float32)
            n = float(np.linalg.norm(forward))
            forward = forward / n if n > 1e-6 else self.u.copy()

            up = self.up - np.dot(self.up, forward) * forward
            up_n = float(np.linalg.norm(up))
            up = up / up_n if up_n > 1e-6 else self.up.copy()

            right = np.cross(forward, up)
            r_n = float(np.linalg.norm(right))
            right = right / r_n if r_n > 1e-6 else self.v.copy()

            up = np.cross(right, forward)

            R_w_c = np.column_stack([right, up, -forward]).astype(np.float32)
            quats[i] = R.from_matrix(R_w_c).as_quat().astype(np.float32)

        quats = self._smooth_quaternions(quats)
        quats = enforce_quat_continuity_xyzw(quats)
        return quats

    def generate(self, num_frames=1800, fps=30, inset=0.03, look_center_bias=0.30, seed=2):
        rng = np.random.default_rng(int(seed))

        ext = (self.bmax - self.bmin).astype(np.float32)
        half_u = float(0.5 * max(ext[0], 1e-3))
        half_v = float(0.5 * max(ext[2], 1e-3))
        inset = float(np.clip(inset, 0.0, 0.25))
        half_u *= (1.0 - inset)
        half_v *= (1.0 - inset)

        origin = self.center.copy()

        corners = [
            origin + (+half_u) * self.u + (+half_v) * self.v,
            origin + (+half_u) * self.u + (-half_v) * self.v,
            origin + (-half_u) * self.u + (-half_v) * self.v,
            origin + (-half_u) * self.u + (+half_v) * self.v,
        ]
        mids = [
            origin + (+half_u) * self.u,
            origin + (-half_u) * self.u,
            origin + (+half_v) * self.v,
            origin + (-half_v) * self.v,
        ]

        keypoints = []
        pattern = [0, 4, 1, 6, 2, 5, 3, 7]
        seq = pattern * 5
        for idx in seq:
            p = (corners[idx] if idx < 4 else mids[idx - 4]).copy()
            p += rng.normal(scale=0.15) * self.u
            p += rng.normal(scale=0.15) * self.v
            keypoints.append(p)

        keypoints = np.asarray(keypoints, dtype=np.float32)
        keypoints = np.vstack([keypoints, keypoints[0]])

        positions = self._create_bspline_path(keypoints, num_frames)
        positions = self._smooth_positions(positions)

        look_ahead = max(2, int(num_frames * 0.03))
        look = np.zeros_like(positions)
        for i in range(num_frames):
            j = min(i + look_ahead, num_frames - 1)
            forward_target = positions[j]
            look[i] = (1.0 - look_center_bias) * forward_target + look_center_bias * origin
        for k in range(3):
            look[:, k] = gaussian_filter1d(look[:, k], sigma=2)

        quats = self.compute_orientations_old(positions, look)
        timestamps = np.arange(num_frames, dtype=np.float32) / float(fps)

        return {
            "positions": positions.astype(np.float32),
            "orientations_hab": quats.astype(np.float32),
            "timestamps": timestamps,
            "fps": int(fps),
            "num_frames": int(num_frames),
        }


def write_twc(output_path, traj, sensor_height_m: float):
    """
    Export Twc but translate by a CONSTANT WORLD-UP offset (0, h, 0).
    This matches the rendered camera height without introducing rotation-coupled offset.
    """
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    offset_world = np.array([0.0, float(sensor_height_m), 0.0], dtype=np.float32)

    with open(output_path, "w") as f:
        f.write("# format=T_wc camera=OpenCV quat=xyzw\n")
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for i in range(traj["num_frames"]):
            ts = float(traj["timestamps"][i])
            t_agent = traj["positions"][i].astype(np.float32)
            q_hab = normalize_quat_xyzw(traj["orientations_hab"][i])

            # FIX: camera center = agent + constant world-up offset
            t_cam = (t_agent + offset_world).astype(np.float32)

            q_ocv = habitat_wc_quat_to_opencv_wc_quat(q_hab)

            f.write(
                f"{ts:.6f} {t_cam[0]:.6f} {t_cam[1]:.6f} {t_cam[2]:.6f} "
                f"{q_ocv[0]:.6f} {q_ocv[1]:.6f} {q_ocv[2]:.6f} {q_ocv[3]:.6f}\n"
            )

    print(f"[OK] Wrote: {output_path}")


def main():
    DATASET_CONFIG = "ThirdParty/Replica-Dataset/data/data/replica.scene_dataset_config.json"
    SCENE_NAME = "room_0"

    OUTPUT_DIR = "habitat_capture"
    RGB_DIR = os.path.join(OUTPUT_DIR, "rgb")
    DEPTH_DIR = os.path.join(OUTPUT_DIR, "depth")
    os.makedirs(RGB_DIR, exist_ok=True)
    os.makedirs(DEPTH_DIR, exist_ok=True)

    W, H = 640, 480
    HFOV = 90.0
    MIN_DEPTH, MAX_DEPTH = 0.1, 10.0
    FLIP_RGB_VERTICAL = False

    NUM_FRAMES = 2000
    FPS = 30

    TWC_OUT = os.path.join(OUTPUT_DIR, "trajectory_twc_eye.txt")

    sim, agent = make_sim_and_agent(
        dataset_config=DATASET_CONFIG,
        scene_name=SCENE_NAME,
        width=W,
        height=H,
        hfov_deg=HFOV,
        min_depth=MIN_DEPTH,
        max_depth=MAX_DEPTH,
        sensor_height_m=SENSOR_HEIGHT_M,
    )
    print(f"[OK] Loaded scene: {SCENE_NAME}")
    print(f"[INFO] Sensor height set to: {SENSOR_HEIGHT_M}m")

    bmin, bmax = estimate_bounds_from_navmesh(sim, num_samples=25000)

    gen = FullBoundsTourGenerator(bounds_min=bmin, bounds_max=bmax, scene_rot_x_deg=SCENE_ROT_X_DEG)
    traj = gen.generate(num_frames=NUM_FRAMES, fps=FPS, inset=0.03, look_center_bias=0.30, seed=2)
    traj["orientations_hab"] = enforce_quat_continuity_xyzw(traj["orientations_hab"])

    # Export Twc with constant world-up offset
    write_twc(TWC_OUT, traj, sensor_height_m=SENSOR_HEIGHT_M)

    # Capture
    for i in range(traj["num_frames"]):
        ts = float(traj["timestamps"][i])
        t_agent = traj["positions"][i].astype(np.float32)
        q_hab = normalize_quat_xyzw(traj["orientations_hab"][i])

        state = habitat_sim.AgentState()
        state.position = t_agent
        state.rotation = quat_from_coeffs(q_hab)
        agent.set_state(state, reset_sensors=True)

        obs = sim.get_sensor_observations()
        rgb = process_rgb(obs["rgb"], flip_vertical=FLIP_RGB_VERTICAL)
        depth_m = obs["depth"].astype(np.float32)
        if FLIP_RGB_VERTICAL:
            depth_m = np.ascontiguousarray(depth_m[::-1, :])

        stamp = f"{ts:.6f}"
        imageio.imwrite(os.path.join(RGB_DIR, f"{stamp}.png"), rgb)
        imageio.imwrite(os.path.join(DEPTH_DIR, f"{stamp}.png"), (depth_m * 1000.0).astype(np.uint16))

        if i % 200 == 0:
            print(f"[INFO] Captured {i}/{traj['num_frames']}")

    sim.close()
    print("[DONE] Capture complete.")


if __name__ == "__main__":
    main()