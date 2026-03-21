from RTSGS.Config.Config import Config

class ReplicaConfig(Config):
    def __init__(self, config_dict=None):
        super().__init__(config_dict)

        self.set('rgb_fx', 600.0)
        self.set('rgb_fy', 600.0)
        self.set('rgb_cx', 599.5)
        self.set('rgb_cy', 339.5)

        self.set('depth_fx', 600.0)
        self.set('depth_fy', 600.0)
        self.set('depth_cx', 599.5)
        self.set('depth_cy', 339.5)

        self.set('rgb_width', 1200)
        self.set('rgb_height', 680)
        self.set('depth_width', 1200)
        self.set('depth_height', 680)
        self.set('depth_scale', 6553.5) 

        self.set('voxel_size', 0.03)
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