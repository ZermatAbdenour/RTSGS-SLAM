import os
from RTSGS.Config.ReplicaConfig import ReplicaConfig
from RTSGS.DataLoader.ReplicaDataLoader import ReplicaDataLoader
from RTSGS.DataLoader.TUMDataLoader import TUMDataLoader
from RTSGS.System import RTSGSSystem
from RTSGS.Tracker.SimpleORBTracker import SimpleORBTracker
from RTSGS.Config.Config import Config
#from RTSGS.Tracker.SimpleOpen3DVO import SimpleOpen3DVO 
import torch
from torch.utils.cpp_extension import load
import compile
if __name__ == "__main__":

    # Load Data
    print("Starting RTSGS System...")

    # TUM

    #data_path = "./Datasets/rgbd_dataset_freiburg3_long_office_household"
    #trajectory_path = "./Datasets/rgbd_dataset_freiburg3_long_office_household/groundtruth.txt"
    
    # Old Replica
    #data_path = "./Datasets/Replica/habitat_capture"
    #trajectory_path = "./Datasets/Replica/habitat_capture/trajectory_twc_eye.txt"
    #data = TUMDataLoader(os.path.join(data_path, "rgb"), os.path.join(data_path, "depth"),trajectory_path)
    print(torch.__version__) 
    torch.cuda.init()
    torch.zeros(1, device="cuda")
    # Replica
    data_path = "./Datasets/Replica/ThirdParty/Replica/room1/results"
    trajectory_path = "./Datasets/Replica/ThirdParty/Replica/room1/traj.txt"

    data = ReplicaDataLoader(data_path=data_path,trajectory_path=trajectory_path)

    #config = Config()
    config = ReplicaConfig()
    print("Loading Data...")
    data.load_data(2000)
    print("Data Loaded.")

    # Initialize Tracker
    tracker = SimpleORBTracker(dataset=data,config=config)

    # Initialize System
    system = RTSGSSystem(data,tracker,config)

    system.run()
