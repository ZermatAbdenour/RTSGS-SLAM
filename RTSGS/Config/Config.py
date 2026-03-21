import numpy as np

class Config:
    def __init__(self, config_dict=None):
        if( config_dict is not None):
            self.config_dict = config_dict
        else:
            self.config_dict = {}
            # Default to TUM-like intrinsics for both rgb and depth cameras.
            self.config_dict.setdefault('rgb_fx', 525.0)
            self.config_dict.setdefault('rgb_fy', 525.0)
            self.config_dict.setdefault('rgb_cx', 319.5)
            self.config_dict.setdefault('rgb_cy', 239.5)
            self.config_dict.setdefault('rgb_width', 640)
            self.config_dict.setdefault('rgb_height', 480)

            self.config_dict.setdefault('depth_fx', 525.0)
            self.config_dict.setdefault('depth_fy', 525.0)
            self.config_dict.setdefault('depth_cx', 319.5)
            self.config_dict.setdefault('depth_cy', 239.5)
            self.config_dict.setdefault('depth_width', 640)
            self.config_dict.setdefault('depth_height', 480)

            self.config_dict.setdefault('T_depth_to_rgb', np.eye(4, dtype=np.float32).tolist())
            self.config_dict.setdefault('depth_scale', 5000.0)
            self.config_dict.setdefault('voxel_size', 0.05)
            self.config_dict.setdefault('kf_translation',0.02)
            self.config_dict.setdefault('kf_rotation',5.0 * np.pi / 180.0)
            
            self.config_dict.setdefault("sigma_px", 4.0)
            self.config_dict.setdefault("sigma_z0", 0.005)
            self.config_dict.setdefault("sigma_z1", 0.0)
            self.config_dict.setdefault("alpha_init", 1.0)
            self.config_dict.setdefault("alpha_min", 0.01)
            self.config_dict.setdefault("alpha_max", 1.0)
            self.config_dict.setdefault("alpha_depth_scale", 0.0)
            self.config_dict.setdefault("gs_points_lr_mult", 0.3)
            self.config_dict.setdefault("gs_depth_loss_weight", 0.1)
            self.config_dict.setdefault("gs_depth_huber_delta", 0.05)
            self.config_dict.setdefault("use_rendered_depth_icp", True)

    def get_rgb_intrinsics(self):
        return np.array(
            [[self.config_dict['rgb_fx'], 0, self.config_dict['rgb_cx']],
             [0, self.config_dict['rgb_fy'], self.config_dict['rgb_cy']],
             [0, 0, 1]],
            dtype=np.float32,
        )

    def get_depth_intrinsics(self):
        return np.array(
            [[self.config_dict['depth_fx'], 0, self.config_dict['depth_cx']],
             [0, self.config_dict['depth_fy'], self.config_dict['depth_cy']],
             [0, 0, 1]],
            dtype=np.float32,
        )

    def get_rgb_size(self):
        return int(self.config_dict['rgb_width']), int(self.config_dict['rgb_height'])

    def get_depth_size(self):
        return int(self.config_dict['depth_width']), int(self.config_dict['depth_height'])

    def get_T_depth_to_rgb(self):
        T = np.asarray(self.config_dict.get('T_depth_to_rgb', np.eye(4, dtype=np.float32)), dtype=np.float32)
        if T.shape != (4, 4):
            return np.eye(4, dtype=np.float32)
        return T

    def get_T_rgb_to_depth(self):
        T_d2r = self.get_T_depth_to_rgb()
        try:
            return np.linalg.inv(T_d2r).astype(np.float32)
        except np.linalg.LinAlgError:
            return np.eye(4, dtype=np.float32)

    def get(self, key, default=None):
        return self.config_dict.get(key, default)

    def set(self, key, value):
        self.config_dict[key] = value

    def to_dict(self):
        return self.config_dict