import numpy as np

class Config:
    def __init__(self, config_dict=None):
        if( config_dict is not None):
            self.config_dict = config_dict
        else:
            self.config_dict = {}
            # set the default value to tum dataset intrinsics
            self.config_dict.setdefault('fx', 525.0)
            self.config_dict.setdefault('fy', 525.0)
            self.config_dict.setdefault('cx', 319.5)
            self.config_dict.setdefault('cy', 239.5)
            self.config_dict.setdefault('width', 640)
            self.config_dict.setdefault('height', 480)
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

    def get_camera_intrinsics(self):
        intrinsics = np.array([[self.config_dict['fx'], 0, self.config_dict['cx']],
                           [0, self.config_dict['fy'], self.config_dict['cy']],
                           [0, 0, 1]], dtype=np.float32)
        return intrinsics
    def get(self, key, default=None):
        return self.config_dict.get(key, default)

    def set(self, key, value):
        self.config_dict[key] = value

    def to_dict(self):
        return self.config_dict