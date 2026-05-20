from RTSGS.Config.Config import load_config
from RTSGS.DataLoader.ReplicaDataLoader import ReplicaDataLoader
from RTSGS.System import RTSGSSystem
from RTSGS.Tracker.ProjectedPointToPlaneTracker import ProjectedPointToPlaneTracker
import torch
import compile

if __name__ == "__main__":
    print("Starting RTSGS System...")
    print(torch.__version__)
    if torch.cuda.is_available():
        torch.cuda.init()

    data_path = "./Datasets/Replica/ThirdParty/Replica/office0/results"
    trajectory_path = "./Datasets/Replica/ThirdParty/Replica/office0/traj.txt"
    data = ReplicaDataLoader(data_path=data_path, trajectory_path=trajectory_path)

    config = load_config("configs/replica.yaml")
    print("Loading Data...")

    data.load_data(2000)
    print("Data Loaded.")

    tracker = ProjectedPointToPlaneTracker(dataset=data, config=config)
    system = RTSGSSystem(data, tracker, config)
    system.run(benchmark=True)
