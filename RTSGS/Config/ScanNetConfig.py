from RTSGS.Config.Config import Config
import numpy as np

class ScanNetConfig(Config):
    def __init__(self, config_dict=None):
        super().__init__(config_dict)

        # ScanNet-specific defaults.
        # These values match the provided ScanNet intrinsics and are used as defaults.
        # The data loader may still overwrite them from intrinsic_color/depth.txt per scene.
        self.set("rgb_fx", 1169.621094)
        self.set("rgb_fy", 1167.105103)
        self.set("rgb_cx", 646.295044)
        self.set("rgb_cy", 489.927032)
        self.set("rgb_width", 1296)
        self.set("rgb_height", 968)

        self.set("depth_fx", 577.590698)
        self.set("depth_fy", 578.729797)
        self.set("depth_cx", 318.905426)
        self.set("depth_cy", 242.683609)
        self.set("depth_width", 640)
        self.set("depth_height", 480)

        self.set("depth_scale", 1000.0)
        self.set("voxel_size", 0.03)

        self.set("sigma_px", 4.0)
        self.set("sigma_z0", 0.003)
        self.set("sigma_z1", 0.0)
        self.set("alpha_init", 1.0)
        self.set("alpha_min", 0.01)
        self.set("alpha_max", 1.0)
        self.set("alpha_depth_scale", 0.0)

        self.set("gs_points_lr_mult", 0.3)
        self.set("gs_depth_loss_weight", 0.1)
        self.set("gs_depth_huber_delta", 0.05)
        self.set("use_rendered_depth_icp", True)
        self.set('kf_translation',0.1)
        self.set('kf_rotation',35.0 * np.pi / 180.0)