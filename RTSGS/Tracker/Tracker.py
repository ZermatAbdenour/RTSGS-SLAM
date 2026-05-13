from abc import ABC, abstractmethod

from RTSGS.Config.Config import Config
from RTSGS.DataLoader.DataLoader import DataLoader

class Tracker(ABC):
    @abstractmethod
    def __init__(self,dataset:DataLoader,config:Config):
        self.poses = []
        self.keyframes_poses=[]
        self.keyframe_frame_indices = []
        self.keyframes_covis_masks = []
        self.config = config
        self.dataset = dataset

    @abstractmethod
    def track_frame(self, rgb,depth = None):
        pass
    
    @abstractmethod
    def visualize_tracking(self):
        pass