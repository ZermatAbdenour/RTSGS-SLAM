import os
from RTSGS.Config.ReplicaConfig import ReplicaConfig
from RTSGS.DataLoader.TUMDataLoader import TUMDataLoader
from RTSGS.System import RTSGSSystem
from RTSGS.Tracker.SimpleORBTracker import SimpleORBTracker
from RTSGS.Config.Config import Config
#from RTSGS.Tracker.SimpleOpen3DVO import SimpleOpen3DVO 

if __name__ == "__main__":

    # Load Data
    print("Starting RTSGS System...")
    #data_path = "./Datasets/rgbd_dataset_freiburg3_long_office_household"
    #trajectory_path = "./Datasets/rgbd_dataset_freiburg3_long_office_household/groundtruth.txt"
    
    data_path = "./Datasets/Replica/habitat_capture"
    trajectory_path = "./Datasets/Replica/habitat_capture/trajectory_twc_eye.txt"
    
    data = TUMDataLoader(os.path.join(data_path, "rgb"), os.path.join(data_path, "depth"),trajectory_path, stream=True)
    #config = Config()
    config = ReplicaConfig()
    print("Loading Data...")
    data.load_data(1000)
    print("Data Loaded.")

    # Initialize Tracker
    tracker = SimpleORBTracker(dataset=data,config=config)

    # Initialize System
    system = RTSGSSystem(data,tracker,config, stream=True)

    system.run()
