from RTSGS.Config.Config import Config

class ReplicaConfig(Config):
    def __init__(self, config_dict=None):
        super().__init__(config_dict)

        self.set('fx', 600.0)
        self.set('fy', 600.0)
        self.set('cx', 599.5)
        self.set('cy', 339.5)

        self.set('width', 1200)
        self.set('height', 680)
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