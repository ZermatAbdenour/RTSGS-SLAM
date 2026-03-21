from RTSGS.Config.Config import Config


class ReplicaConfig(Config):
    def __init__(self, config_dict=None):
        super().__init__(config_dict)

        self.set('rgb_fx', 320.0)
        self.set('rgb_fy', 320.0)
        self.set('rgb_cx', 319.5)
        self.set('rgb_cy', 239.5)

        self.set('depth_fx', 320.0)
        self.set('depth_fy', 320.0)
        self.set('depth_cx', 319.5)
        self.set('depth_cy', 239.5)

        self.set('rgb_width', 640)
        self.set('rgb_height', 480)
        self.set('depth_width', 640)
        self.set('depth_height', 480)
        self.set('depth_scale', 1000.0)
        self.set('voxel_size', 0.02)

        self.set("sigma_px", 2.0)
        self.set("sigma_z0", 0.001)
        self.set("sigma_z1", 0.0)
        self.set("alpha_init", 1.0)
        self.set("alpha_min", 0.01)
        self.set("alpha_max", 1.0)
        self.set("alpha_depth_scale", 0.)