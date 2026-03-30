import os
from RTSGS.Config.ReplicaConfig import ReplicaConfig
from RTSGS.Config.ScanNetConfig import ScanNetConfig
from RTSGS.DataLoader.ReplicaDataLoader import ReplicaDataLoader
from RTSGS.DataLoader.ScanNetDataLoader import ScanNetDataLoader
from RTSGS.DataLoader.TUMDataLoader import TUMDataLoader
from RTSGS.System import RTSGSSystem
from RTSGS.Tracker.ProjectedPointToPlaneTracker import ProjectedPointToPlaneTracker
from RTSGS.Config.Config import Config
#from RTSGS.Tracker.SimpleOpen3DVO import SimpleOpen3DVO 
import torch
from torch.utils.cpp_extension import load
import compile
if __name__ == "__main__":

    # Load Data
    print("Starting RTSGS System...")

    print(torch.__version__)
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.zeros(1, device="cuda")

    # TUM
    #data_path = "./Datasets/rgbd_dataset_freiburg3_long_office_household"
    #trajectory_path = "./Datasets/rgbd_dataset_freiburg3_long_office_household/groundtruth.txt"
    #data = TUMDataLoader(os.path.join(data_path, "rgb"), os.path.join(data_path, "depth"),trajectory_path)

    # Replica
    data_path = "./Datasets/Replica/ThirdParty/Replica/room0/results"
    trajectory_path = "./Datasets/Replica/ThirdParty/Replica/room0/traj.txt"
    data = ReplicaDataLoader(data_path=data_path,trajectory_path=trajectory_path)

    # ScanNet
    #scene_extracted_path = "./Datasets/ScanNet/data/scans/scene0000_00/extracted"
    #trajectory_path = "./Datasets/ScanNet/data/scans/scene0000_00/extracted/pose"
    #config = ScanNetConfig()
    #data = ScanNetDataLoader(scene_extracted_path, config, trajectory_path=trajectory_path)

    #config = Config()
    #config = ReplicaConfig()
    config = ReplicaConfig()
    print("Loading Data...")
    
    data.load_data(1000)
    print("Data Loaded.")

    # Initialize Tracker
    tracker = ProjectedPointToPlaneTracker(dataset=data, config=config)

    # Initialize System
    system = RTSGSSystem(data,tracker,config)

    system.run()
