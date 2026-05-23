from RTSGS.Config.Config import load_config
from RTSGS.DataLoader.ReplicaDataLoader import ReplicaDataLoader
from RTSGS.DataLoader.TUMDataLoader import TUMDataLoader
from RTSGS.System import RTSGSSystem
from RTSGS.Tracker.ProjectedPointToPlaneTracker import ProjectedPointToPlaneTracker
import torch
import compile

if __name__ == "__main__":
    print("Starting RTSGS System...")
    print(torch.__version__)
    if torch.cuda.is_available():
        torch.cuda.init()

    data_path = "./Datasets/Replica/ThirdParty/Replica/room1/results"
    trajectory_path = "./Datasets/Replica/ThirdParty/Replica/room1/traj.txt"
    #rgb_path = "./Datasets/rgbd_dataset_freiburg3_long_office_household/rgb"
    #depth_path = "./Datasets/rgbd_dataset_freiburg3_long_office_household/depth"
    #trajectory_path = "./Datasets/rgbd_dataset_freiburg3_long_office_household/groundtruth.txt"

    #data = TUMDataLoader(rgb_path=rgb_path, depth_path=depth_path, gt_path=trajectory_path, fps=2)
    data = ReplicaDataLoader(data_path=data_path, trajectory_path=trajectory_path, fps=10)

    config = load_config("configs/replica.yaml")
    print("Loading Data...")

    data.load_data(1000)
    print("Data Loaded.")

    tracker = ProjectedPointToPlaneTracker(dataset=data, config=config)
    system = RTSGSSystem(data, tracker, config)
    system.run(benchmark=True)
