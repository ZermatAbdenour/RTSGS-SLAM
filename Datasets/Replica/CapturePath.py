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
# FIX: Eye height low
# =============================================================================
# The reason eye height is still low is this logic:
#   h_eye is clamped into [eye_h_min, eye_h_max] where
#   eye_h_min/max are derived from floor (h_floor + 1.4/1.8).
#
# If you want "center height" in the room, you MUST NOT clamp it to floor+1.8.
# That clamp forces a low height.
#
# So we introduce TWO independent height modes:
#   - "human": clamp to floor+ [1.4, 1.8]  (old)
#   - "center": clamp to a band inside scene bounds (e.g. 30%..70% of height)
#
# This is the minimal correct fix.
# =============================================================================


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


def estimate_bounds_and_floor_from_navmesh(sim: habitat_sim.Simulator, num_samples: int = 10000):
    pts = []
    for _ in range(int(num_samples)):
        p = sim.pathfinder.get_random_navigable_point()
        pts.append([p[0], p[1], p[2]])
    pts = np.asarray(pts, dtype=np.float32)

    bmin = np.min(pts, axis=0).astype(np.float32)
    bmax = np.max(pts, axis=0).astype(np.float32)

    floor_y = float(np.percentile(pts[:, 1], 5))
    ceil_y = float(np.percentile(pts[:, 1], 95))
    return bmin, bmax, floor_y, ceil_y


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


class VelocityControlledTrajectoryGenerator:
    def __init__(self, bounds_min, bounds_max, floor_y, scene_rot_x_deg=0.0, margin=0.2):
        self.scene_rot_x_deg = float(scene_rot_x_deg)
        self.R_scene = rotation_matrix_x(self.scene_rot_x_deg)

        self.bounds_min, self.bounds_max = rotate_aabb(bounds_min, bounds_max, self.R_scene)
        self.center = ((self.bounds_min + self.bounds_max) / 2.0).astype(np.float32)

        self.world_up = normalize(rotate_points(np.array([0.0, 1.0, 0.0], dtype=np.float32), self.R_scene))

        floor_point = np.array([0.0, float(floor_y), 0.0], dtype=np.float32)
        floor_point_rot = rotate_points(floor_point, self.R_scene)
        self.h_floor = float(np.dot(floor_point_rot, self.world_up))

        # "Human" eye height band (used only in height_mode="human")
        self.eye_h_min = self.h_floor + 1.4
        self.eye_h_max = self.h_floor + 1.8

        self.margin = float(margin)

        u0 = project_to_plane(np.array([1.0, 0.0, 0.0], dtype=np.float32), self.world_up)
        if np.linalg.norm(u0) < 1e-4:
            u0 = project_to_plane(np.array([0.0, 0.0, 1.0], dtype=np.float32), self.world_up)
        self.u = normalize(u0)
        self.v = normalize(np.cross(self.world_up, self.u).astype(np.float32))

    def generate_smooth_velocity_profile(self, num_frames, max_speed=0.5, acceleration_time=0.2):
        accel_frames = int(num_frames * acceleration_time)
        cruise_frames = num_frames - 2 * accel_frames
        if cruise_frames < 0:
            accel_frames = num_frames // 3
            cruise_frames = num_frames - 2 * accel_frames

        t_accel = np.linspace(0, 1, accel_frames)
        accel_profile = 0.5 - 0.5 * np.cos(t_accel * np.pi)
        cruise_profile = np.ones(cruise_frames, dtype=np.float32)
        t_decel = np.linspace(0, 1, accel_frames)
        decel_profile = 0.5 + 0.5 * np.cos(t_decel * np.pi)

        v = np.concatenate([accel_profile, cruise_profile, decel_profile]).astype(np.float32)
        v *= float(max_speed)

        if len(v) > num_frames:
            v = v[:num_frames]
        elif len(v) < num_frames:
            v = np.pad(v, (0, num_frames - len(v)), mode="edge")
        return v

    def _create_bspline_path(self, keypoints, num_frames):
        distances = np.cumsum(np.sqrt(np.sum(np.diff(keypoints, axis=0) ** 2, axis=1)))
        distances = np.insert(distances, 0, 0)
        t_norm = distances / distances[-1]
        t_new = np.linspace(0, 1, num_frames)

        positions = np.zeros((num_frames, 3), dtype=np.float32)
        for i in range(3):
            spline = CubicSpline(t_norm, keypoints[:, i], bc_type="periodic")
            positions[:, i] = spline(t_new).astype(np.float32)
        return positions

    def _adjust_path_for_velocity(self, positions, velocity_profile):
        num_frames = len(positions)
        diffs = np.diff(positions, axis=0)
        seg_lens = np.sqrt(np.sum(diffs**2, axis=1))
        cumdist = np.insert(np.cumsum(seg_lens), 0, 0.0)
        total_len = float(cumdist[-1])

        dt = 1.0 / (num_frames - 1)
        desired = np.zeros(num_frames, dtype=np.float32)
        for i in range(1, num_frames):
            avg_v = (velocity_profile[i - 1] + velocity_profile[i]) / 2.0
            desired[i] = desired[i - 1] + avg_v * dt

        if desired[-1] > 0:
            desired *= (total_len / float(desired[-1]))

        new_pos = np.zeros_like(positions)
        new_pos[0] = positions[0]

        for i in range(1, num_frames):
            target = float(desired[i])
            idx = int(np.searchsorted(cumdist, target) - 1)
            idx = np.clip(idx, 0, len(seg_lens) - 1)
            dist_in_seg = target - float(cumdist[idx])
            if seg_lens[idx] > 1e-12:
                alpha = np.clip(dist_in_seg / float(seg_lens[idx]), 0.0, 1.0)
                new_pos[i] = positions[idx] + alpha * (positions[idx + 1] - positions[idx])
            else:
                new_pos[i] = positions[idx]
        return new_pos

    def _apply_gentle_smoothing(self, positions):
        sm = positions.copy()
        for i in range(3):
            sm[:, i] = gaussian_filter1d(sm[:, i], sigma=1.5)
            window = min(15, len(sm) // 10)
            if window >= 5 and window % 2 == 1:
                sm[:, i] = savgol_filter(sm[:, i], window_length=window, polyorder=3)
        return sm.astype(np.float32)

    def _weak_aabb_clamp(self, positions):
        p = positions.copy()
        p[:, 0] = np.clip(p[:, 0], self.bounds_min[0] + self.margin, self.bounds_max[0] - self.margin)
        p[:, 1] = np.clip(p[:, 1], self.bounds_min[1] + self.margin, self.bounds_max[1] - self.margin)
        p[:, 2] = np.clip(p[:, 2], self.bounds_min[2] + self.margin, self.bounds_max[2] - self.margin)
        return p.astype(np.float32)

    def _smooth_quaternions(self, quaternions):
        num_frames = len(quaternions)
        if num_frames < 10:
            return quaternions
        rotations = R.from_quat(quaternions)
        rotvecs = rotations.as_rotvec()
        sm = np.zeros_like(rotvecs)
        for k in range(3):
            sm[:, k] = gaussian_filter1d(rotvecs[:, k], sigma=2)
        sm_rots = R.from_rotvec(sm)

        key_int = max(1, num_frames // 20)
        key_idx = list(range(0, num_frames, key_int))
        if key_idx[-1] != num_frames - 1:
            key_idx.append(num_frames - 1)

        slerp = Slerp(key_idx, sm_rots[key_idx])
        final_rots = slerp(np.arange(num_frames))
        return final_rots.as_quat().astype(np.float32)

    def compute_orientations_old(self, positions, look_ahead_frames=10):
        num_frames = len(positions)
        look_at_points = np.zeros_like(positions)

        for i in range(num_frames):
            look_ahead_idx = min(i + look_ahead_frames, num_frames - 1)
            blend = 0.8
            look_at_points[i] = blend * positions[look_ahead_idx] + (1 - blend) * self.center

        for k in range(3):
            look_at_points[:, k] = gaussian_filter1d(look_at_points[:, k], sigma=3)

        quats = np.zeros((num_frames, 4), dtype=np.float32)

        for i in range(num_frames):
            forward = (look_at_points[i] - positions[i]).astype(np.float32)
            n = float(np.linalg.norm(forward))
            forward = forward / n if n > 1e-6 else self.u.copy()

            up = self.world_up - np.dot(self.world_up, forward) * forward
            up_n = float(np.linalg.norm(up))
            up = up / up_n if up_n > 1e-6 else self.world_up.copy()

            right = np.cross(forward, up)
            r_n = float(np.linalg.norm(right))
            right = right / r_n if r_n > 1e-6 else self.v.copy()

            up = np.cross(right, forward)

            R_w_c = np.column_stack([right, up, -forward]).astype(np.float32)
            quats[i] = R.from_matrix(R_w_c).as_quat().astype(np.float32)

        return self._smooth_quaternions(quats)

    def generate(
        self,
        num_frames=300,
        fps=30,
        max_speed=0.4,
        path_scale=0.80,
        height_mode: str = "center",          # "center" | "human"
        center_height_fraction: float = 0.55, # 0..1 within bounds height
        center_height_band: tuple[float, float] = (0.35, 0.75),  # keep away from extremes
        fixed_height_offset_m: float = 0.0,   # extra meters along up
    ):
        vprof = self.generate_smooth_velocity_profile(num_frames, max_speed=max_speed, acceleration_time=0.2)

        extent = (self.bounds_max - self.bounds_min).astype(np.float32)
        base = float(min(extent[0], extent[2]))
        path_scale = float(np.clip(path_scale, 0.1, 0.95))
        rx = max(0.8, base * path_scale)
        rz = max(0.8, base * path_scale)

        # Determine height along UP
        h_min = float(np.dot(self.bounds_min, self.world_up))
        h_max = float(np.dot(self.bounds_max, self.world_up))

        if height_mode == "center":
            band_lo, band_hi = center_height_band
            band_lo = float(np.clip(band_lo, 0.0, 1.0))
            band_hi = float(np.clip(band_hi, 0.0, 1.0))
            if band_hi <= band_lo:
                band_lo, band_hi = 0.35, 0.75

            # Compute a bounded fraction inside the band
            frac = float(np.clip(center_height_fraction, 0.0, 1.0))
            frac = band_lo + frac * (band_hi - band_lo)

            h_eye = h_min + frac * (h_max - h_min)
            h_eye = float(h_eye + fixed_height_offset_m)  # allow pushing up more
            # Clamp inside band in meters (not human eye band)
            h_eye = float(np.clip(h_eye, h_min + band_lo * (h_max - h_min), h_min + band_hi * (h_max - h_min)))
        else:
            # Human eye band relative to floor
            h_eye = float(np.clip(self.h_floor + 1.6, self.eye_h_min, self.eye_h_max))

        # Center point with chosen height
        origin = self.center.copy()
        cur_h = float(np.dot(origin, self.world_up))
        origin = origin + (h_eye - cur_h) * self.world_up

        # Figure-8 in horizontal plane (u,v)
        t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        a = rx * np.sin(t)
        b = rz * np.sin(2 * t) * 0.6
        keypoints = (origin[None, :]
                     + a[:, None] * self.u[None, :]
                     + b[:, None] * self.v[None, :]).astype(np.float32)
        keypoints = np.vstack([keypoints, keypoints[0]])

        positions = self._create_bspline_path(keypoints, num_frames)
        positions = self._adjust_path_for_velocity(positions, vprof)

        # Keep inside AABB, but do NOT force back to floor band
        positions = self._weak_aabb_clamp(positions)
        positions = self._apply_gentle_smoothing(positions)

        # Re-enforce exact height along up after smoothing (hold constant height)
        # This avoids drift downwards.
        cur_h = positions @ self.world_up
        positions = positions + (h_eye - cur_h)[:, None] * self.world_up[None, :]

        quats = self.compute_orientations_old(positions, look_ahead_frames=max(1, int(num_frames * 0.05)))
        timestamps = np.arange(num_frames, dtype=np.float32) / float(fps)

        return {
            "positions": positions.astype(np.float32),
            "orientations_hab": quats.astype(np.float32),
            "timestamps": timestamps,
            "fps": int(fps),
            "num_frames": int(num_frames),
            "max_speed": float(max_speed),
            "scene_rot_x_deg": float(self.scene_rot_x_deg),
            "bounds_min": self.bounds_min,
            "bounds_max": self.bounds_max,
            "world_up": self.world_up,
            "rx": float(rx),
            "rz": float(rz),
            "h_eye": float(h_eye),
            "h_min": float(h_min),
            "h_max": float(h_max),
        }


def write_tum_t_wc_opencv(output_path, traj):
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw   (T_wc, OpenCV camera convention)\n")
        for i in range(traj["num_frames"]):
            ts = float(traj["timestamps"][i])
            t = traj["positions"][i]
            q_hab = traj["orientations_hab"][i]
            q_ocv = habitat_wc_quat_to_opencv_wc_quat(q_hab)
            f.write(
                f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                f"{q_ocv[0]:.6f} {q_ocv[1]:.6f} {q_ocv[2]:.6f} {q_ocv[3]:.6f}\n"
            )
    print(f"[OK] Wrote: {output_path}")


def make_sim_and_agent(dataset_config, scene_name, width, height, hfov_deg, min_depth, max_depth):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = dataset_config
    sim_cfg.scene_id = scene_name
    sim_cfg.enable_physics = False

    sensor_specs = []

    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [height, width]
    rgb.position = mn.Vector3(0.0, 0.0, 0.0)
    rgb.hfov = float(hfov_deg)
    sensor_specs.append(rgb)

    depth = habitat_sim.CameraSensorSpec()
    depth.uuid = "depth"
    depth.sensor_type = habitat_sim.SensorType.DEPTH
    depth.resolution = [height, width]
    depth.position = mn.Vector3(0.0, 0.0, 0.0)
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


def main():
    DATASET_CONFIG = "ThirdParty/Replica-Dataset/data/data/replica.scene_dataset_config.json"
    SCENE_NAME = "room_0"

    OUTPUT_DIR = "habitat_capture"
    RGB_DIR = os.path.join(OUTPUT_DIR, "rgb")
    DEPTH_DIR = os.path.join(OUTPUT_DIR, "depth")
    os.makedirs(RGB_DIR, exist_ok=True)
    os.makedirs(DEPTH_DIR, exist_ok=True)

    # Capture
    W, H = 640, 480
    HFOV = 90.0
    MIN_DEPTH, MAX_DEPTH = 0.1, 10.0
    FLIP_RGB_VERTICAL = False

    # Trajectory
    NUM_FRAMES = 200
    FPS = 30
    MAX_SPEED = 0.35

    # Scene rotation correction (set 0 if not needed)
    SCENE_ROT_X_DEG = 90.0  # try -90.0 or 0.0

    # More motion in horizontal plane
    PATH_SCALE = 0.85

    # Height control: use center band, not floor eye band
    HEIGHT_MODE = "center"
    CENTER_HEIGHT_FRACTION = 0.60
    CENTER_HEIGHT_BAND = (0.40, 0.85)  # keep camera higher; adjust if too close to ceiling
    FIXED_HEIGHT_OFFSET_M = 0.3        # push up by +0.3m along up-axis

    # Outputs
    TUM_OUT = os.path.join(OUTPUT_DIR, "trajectory_twc_opencv.txt")
    POSES_USED_HAB = os.path.join(OUTPUT_DIR, "poses_used_habitat.txt")

    # Create simulator first (navmesh)
    sim, agent = make_sim_and_agent(
        dataset_config=DATASET_CONFIG,
        scene_name=SCENE_NAME,
        width=W,
        height=H,
        hfov_deg=HFOV,
        min_depth=MIN_DEPTH,
        max_depth=MAX_DEPTH,
    )
    print(f"[OK] Loaded scene: {SCENE_NAME}")

    bmin, bmax, floor_y, _ = estimate_bounds_and_floor_from_navmesh(sim, num_samples=12000)
    print("[INFO] Navmesh bounds raw min:", bmin)
    print("[INFO] Navmesh bounds raw max:", bmax)
    print("[INFO] Navmesh floor_y raw:", floor_y)

    gen = VelocityControlledTrajectoryGenerator(
        bounds_min=bmin,
        bounds_max=bmax,
        floor_y=floor_y,
        scene_rot_x_deg=SCENE_ROT_X_DEG,
        margin=0.25,
    )

    traj = gen.generate(
        num_frames=NUM_FRAMES,
        fps=FPS,
        max_speed=MAX_SPEED,
        path_scale=PATH_SCALE,
        height_mode=HEIGHT_MODE,
        center_height_fraction=CENTER_HEIGHT_FRACTION,
        center_height_band=CENTER_HEIGHT_BAND,
        fixed_height_offset_m=FIXED_HEIGHT_OFFSET_M,
    )

    print("[INFO] Rotated bounds min:", traj["bounds_min"])
    print("[INFO] Rotated bounds max:", traj["bounds_max"])
    print("[INFO] Rotated world_up:", traj["world_up"])
    print("[INFO] Trajectory rx/rz:", traj["rx"], traj["rz"])
    print("[INFO] Height range along up: h_min/h_max:", traj["h_min"], traj["h_max"])
    print("[INFO] Chosen h_eye:", traj["h_eye"])

    write_tum_t_wc_opencv(TUM_OUT, traj)

    # Capture
    with open(POSES_USED_HAB, "w") as fposes:
        fposes.write("# timestamp tx ty tz qx qy qz qw  (Habitat R_w_c actually used)\n")

        for i in range(traj["num_frames"]):
            ts = float(traj["timestamps"][i])
            t_w_c = traj["positions"][i]
            q_w_c_hab = traj["orientations_hab"][i]

            state = habitat_sim.AgentState()
            state.position = t_w_c
            state.rotation = quat_from_coeffs(q_w_c_hab)
            agent.set_state(state, reset_sensors=True)

            obs = sim.get_sensor_observations()

            rgb = process_rgb(obs["rgb"], flip_vertical=FLIP_RGB_VERTICAL)
            depth_m = obs["depth"].astype(np.float32)
            if FLIP_RGB_VERTICAL:
                depth_m = np.ascontiguousarray(depth_m[::-1, :])

            stamp = f"{ts:.6f}"
            imageio.imwrite(os.path.join(RGB_DIR, f"{stamp}.png"), rgb)
            imageio.imwrite(os.path.join(DEPTH_DIR, f"{stamp}.png"), (depth_m * 1000.0).astype(np.uint16))

            fposes.write(
                f"{stamp} {t_w_c[0]} {t_w_c[1]} {t_w_c[2]} "
                f"{q_w_c_hab[0]} {q_w_c_hab[1]} {q_w_c_hab[2]} {q_w_c_hab[3]}\n"
            )

            if i % 50 == 0:
                print(f"[INFO] Captured {i}/{traj['num_frames']}")

    sim.close()
    print("[DONE] Capture complete.")
    print(f"  RGB:   {RGB_DIR}")
    print(f"  Depth: {DEPTH_DIR}")
    print(f"  TUM (OpenCV): {TUM_OUT}")
    print(f"  Poses used (Habitat): {POSES_USED_HAB}")


if __name__ == "__main__":
    main()