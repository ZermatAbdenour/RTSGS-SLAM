import argparse
import math
import os
import sys
import time

import numpy as np
import open3d as o3d
import torch
import yaml
from munch import Munch


def add_softgroup_to_path(project_root: str) -> None:
    softgroup_root = os.path.join(project_root, "ThirdParty", "SoftGroup")
    if softgroup_root not in sys.path:
        sys.path.insert(0, softgroup_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SoftGroup inference on a single PLY scan")
    parser.add_argument(
        "--ply",
        type=str,
        default="Datasets/ScanNet/data/scans/scene0000_00/scene0000_00_vh_clean.ply",
        help="Path to input PLY point cloud",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="ThirdParty/SoftGroup/configs/softgroup/softgroup_scannet.yaml",
        help="SoftGroup config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="Prototype/softgroup_scannet_spconv2.pth",
        help="Pretrained SoftGroup checkpoint",
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Disable Open3D visualization",
    )
    parser.add_argument(
        "--save-semantic",
        type=str,
        default="",
        help="Optional path to save semantic predictions (.npy)",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=3,
        help="Number of warmup forward passes before benchmark timing",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=20,
        help="Number of timed forward passes to average",
    )
    parser.add_argument(
        "--no-fast-mode",
        action="store_true",
        help="Disable fast inference settings (lvl_fusion, cuDNN benchmark, TF32)",
    )
    parser.add_argument(
        "--no-auto-vh-clean-2",
        action="store_true",
        help="Disable auto-switch from *_vh_clean.ply to *_vh_clean_2.ply when available",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=2.5,
        help="Point size used in visualization",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.1,
        help="Minimum instance confidence for filtering and labeling",
    )
    return parser.parse_args()


def to_abs(project_root: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(project_root, path)


def maybe_use_decimated_scannet_mesh(ply_path: str, allow_auto_switch: bool) -> str:
    if not allow_auto_switch:
        return ply_path
    if not ply_path.endswith("_vh_clean.ply"):
        return ply_path
    candidate = ply_path.replace("_vh_clean.ply", "_vh_clean_2.ply")
    if os.path.exists(candidate):
        print(f"Using decimated mesh for speed: {candidate}")
        return candidate
    return ply_path


def load_config(config_path: str) -> Munch:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return Munch.fromDict(cfg)


def load_scan_ply(ply_path: str) -> tuple[np.ndarray, np.ndarray]:
    pcd = o3d.io.read_point_cloud(ply_path)
    xyz = np.asarray(pcd.points, dtype=np.float32)
    if xyz.size == 0:
        raise ValueError(f"No points loaded from: {ply_path}")

    if pcd.has_colors():
        rgb = np.asarray(pcd.colors, dtype=np.float32)
    else:
        rgb = np.zeros_like(xyz, dtype=np.float32)

    # SoftGroup ScanNet preprocessing uses colors in [-1, 1].
    if rgb.max(initial=0.0) <= 1.0:
        rgb = rgb * 2.0 - 1.0
    else:
        rgb = rgb / 127.5 - 1.0

    return xyz, rgb


def scannet_test_transform(xyz: np.ndarray, scale: int) -> tuple[np.ndarray, np.ndarray]:
    # Match CustomDataset.transform_test deterministic rotation.
    theta = 0.35 * math.pi
    rot = np.array(
        [
            [math.cos(theta), math.sin(theta), 0.0],
            [-math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    xyz_middle = xyz @ rot
    xyz_scaled = xyz_middle * float(scale)
    xyz_scaled = xyz_scaled - xyz_scaled.min(axis=0, keepdims=True)
    return xyz_scaled, xyz_middle


def build_single_scene_batch(
    xyz: np.ndarray,
    rgb: np.ndarray,
    scan_id: str,
    cfg: Munch,
    device: torch.device,
):
    from softgroup.ops import voxelization_idx

    voxel_scale = int(cfg.data.test.voxel_cfg.scale)
    min_spatial_shape = int(cfg.data.test.voxel_cfg.spatial_shape[0])

    # Run test-time transform on GPU to minimize CPU<->GPU transfers.
    xyz_t = torch.from_numpy(xyz).to(device=device, dtype=torch.float32)
    theta = 0.35 * math.pi
    rot = torch.tensor(
        [
            [math.cos(theta), math.sin(theta), 0.0],
            [-math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    )
    coord_float = xyz_t @ rot
    xyz_scaled = coord_float * float(voxel_scale)
    xyz_scaled = xyz_scaled - xyz_scaled.min(dim=0, keepdim=True).values

    feat = torch.from_numpy(rgb).to(device=device, dtype=torch.float32)
    n = xyz_scaled.shape[0]

    # voxelization_idx expects CPU coordinates in this SoftGroup implementation.
    coord_cpu = xyz_scaled.to("cpu").long()
    coords_cpu = torch.cat([torch.zeros((n, 1), dtype=torch.long), coord_cpu], dim=1)
    batch_idxs = coords_cpu[:, 0].int().to(device)

    # Dummy labels keep forward_test signature satisfied for unlabeled inference.
    semantic_labels = torch.zeros((n,), dtype=torch.long, device=device)
    instance_labels = torch.full((n,), -100, dtype=torch.long, device=device)
    pt_offset_labels = torch.zeros((n, 3), dtype=torch.float32, device=device)

    spatial_shape = np.clip(coords_cpu[:, 1:].max(0)[0].numpy() + 1, min_spatial_shape, None)
    voxel_coords, v2p_map, p2v_map = voxelization_idx(coords_cpu, 1)

    batch = {
        "scan_ids": [scan_id],
        "batch_idxs": batch_idxs,
        "voxel_coords": voxel_coords.to(device),
        "p2v_map": p2v_map.to(device),
        "v2p_map": v2p_map.to(device),
        "coords_float": coord_float,
        "feats": feat,
        "semantic_labels": semantic_labels,
        "instance_labels": instance_labels,
        "pt_offset_labels": pt_offset_labels,
        "spatial_shape": spatial_shape,
        "batch_size": 1,
    }
    return batch


def colorize_semantic(semantic_preds: np.ndarray, num_classes: int) -> np.ndarray:
    rng = np.random.default_rng(seed=123)
    palette = rng.uniform(0.1, 0.95, size=(max(num_classes, 1), 3)).astype(np.float32)
    safe_labels = np.clip(semantic_preds.astype(np.int64), 0, palette.shape[0] - 1)
    return palette[safe_labels]


def get_instance_class_names(cfg: Munch) -> list[str]:
    data_type = str(cfg.data.test.type)
    if data_type == "scannetv2":
        from softgroup.data.scannetv2 import ScanNetDataset

        return list(ScanNetDataset.CLASSES)
    if data_type == "s3dis":
        from softgroup.data.s3dis import S3DISDataset

        return list(S3DISDataset.CLASSES)
    if data_type == "stpls3d":
        from softgroup.data.stpls3d import STPLS3DDataset

        return list(STPLS3DDataset.CLASSES)
    if data_type == "kitti":
        from softgroup.data.kitti import KITTIDataset

        return list(KITTIDataset.CLASSES)
    return []


def visualize_with_instance_labels(
    points: np.ndarray,
    colors: np.ndarray,
    pred_instances: list[dict],
    class_names: list[str],
    rle_decode,
    point_size: float,
) -> None:
    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(points)
    pcd_pred.colors = o3d.utility.Vector3dVector(colors)

    try:
        app = o3d.visualization.gui.Application.instance
        app.initialize()
        vis = o3d.visualization.O3DVisualizer("SoftGroup Inference", 1400, 900)
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = max(1.0, float(point_size))
        vis.show_skybox(False)
        vis.add_geometry("scene", pcd_pred, material)
        vis.reset_camera_to_default()

        for inst in pred_instances:
            label_id = int(inst.get("label_id", -1))
            conf = float(inst.get("conf", 0.0))
            mask = rle_decode(inst["pred_mask"]).astype(bool)
            if mask.shape[0] != points.shape[0] or not np.any(mask):
                continue
            center = points[mask].mean(axis=0)
            if 1 <= label_id <= len(class_names):
                class_name = class_names[label_id - 1]
            else:
                class_name = f"class_{label_id}"
            vis.add_3d_label(center, f"{class_name} ({conf:.2f})")

        app.add_window(vis)
        app.run()
    except Exception as e:
        print(f"Open3D 3D labels unavailable ({e}). Falling back to basic visualizer.")
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="SoftGroup Inference", width=1400, height=900)
        vis.add_geometry(pcd_pred)
        render_opt = vis.get_render_option()
        render_opt.background_color = np.array([1.0, 1.0, 1.0])
        render_opt.point_size = max(1.0, float(point_size))
        view_ctl = vis.get_view_control()
        view_ctl.set_zoom(0.7)
        vis.run()
        vis.destroy_window()


def filter_instances_by_confidence(pred_instances: list[dict], min_conf: float) -> list[dict]:
    return [inst for inst in pred_instances if float(inst.get("conf", 0.0)) > min_conf]


def benchmark_forward(model, batch: dict, warmup_runs: int, benchmark_runs: int):
    if warmup_runs < 0:
        warmup_runs = 0
    if benchmark_runs <= 0:
        benchmark_runs = 1

    with torch.inference_mode():
        for _ in range(warmup_runs):
            _ = model(batch)

    torch.cuda.reset_peak_memory_stats()
    durations_ms = []
    result = None
    with torch.inference_mode():
        for _ in range(benchmark_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = model(batch)
            torch.cuda.synchronize()
            durations_ms.append((time.perf_counter() - t0) * 1000.0)

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
    mean_ms = float(np.mean(durations_ms))
    std_ms = float(np.std(durations_ms))
    return result, mean_ms, std_ms, peak_vram_mb


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("SoftGroup inference requires CUDA and compiled CUDA ops.")

    project_root = os.path.dirname(os.path.abspath(__file__))
    add_softgroup_to_path(project_root)

    from softgroup.model import SoftGroup
    from softgroup.util import get_root_logger, load_checkpoint, rle_decode

    args = parse_args()
    ply_path = to_abs(project_root, args.ply)
    ply_path = maybe_use_decimated_scannet_mesh(ply_path, allow_auto_switch=not args.no_auto_vh_clean_2)
    config_path = to_abs(project_root, args.config)
    checkpoint_path = to_abs(project_root, args.checkpoint)

    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY not found: {ply_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    cfg = load_config(config_path)

    if not args.no_fast_mode:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        cfg.model.test_cfg.lvl_fusion = True

    preprocess_t0 = time.perf_counter()
    xyz, rgb = load_scan_ply(ply_path)
    scan_id = os.path.splitext(os.path.basename(ply_path))[0]
    device = torch.device("cuda")
    batch = build_single_scene_batch(xyz, rgb, scan_id, cfg, device)
    preprocess_ms = (time.perf_counter() - preprocess_t0) * 1000.0

    model = SoftGroup(**cfg.model).cuda()
    model.eval()
    logger = get_root_logger()
    load_checkpoint(checkpoint_path, logger, model)

    result, forward_mean_ms, forward_std_ms, peak_vram_mb = benchmark_forward(
        model,
        batch,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
    )

    post_t0 = time.perf_counter()

    semantic_preds = result["semantic_preds"]
    pred_instances = result.get("pred_instances", [])
    pred_instances = filter_instances_by_confidence(pred_instances, args.min_confidence)

    postprocess_ms = (time.perf_counter() - post_t0) * 1000.0
    end_to_end_ms = preprocess_ms + forward_mean_ms + postprocess_ms

    print(f"Scan ID: {result['scan_id']}")
    print(
        f"Forward time (avg over {args.benchmark_runs} runs, warmup {args.warmup_runs}): "
        f"{forward_mean_ms:.2f} +- {forward_std_ms:.2f} ms"
    )
    print(f"Preprocess time: {preprocess_ms:.2f} ms")
    print(f"Postprocess time: {postprocess_ms:.2f} ms")
    print(f"End-to-end time (no visualization): {end_to_end_ms:.2f} ms")
    print(f"Peak VRAM: {peak_vram_mb:.2f} MB")
    print(f"Fast mode: {not args.no_fast_mode}")
    print(f"Points in scan: {xyz.shape[0]}")
    print(f"Points: {semantic_preds.shape[0]}")
    print(f"Unique semantic classes: {np.unique(semantic_preds).tolist()}")

    print(f"Predicted instances (raw): {len(result.get('pred_instances', []))}")
    print(f"Predicted instances (conf > {args.min_confidence:.2f}): {len(pred_instances)}")

    if args.save_semantic:
        save_path = to_abs(project_root, args.save_semantic)
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        np.save(save_path, semantic_preds)
        print(f"Saved semantic predictions to: {save_path}")

    if not args.no_vis:
        vis_points = result.get("coords_float", xyz)
        vis_colors = colorize_semantic(semantic_preds, int(cfg.model.semantic_classes))
        class_names = get_instance_class_names(cfg)
        visualize_with_instance_labels(
            vis_points,
            vis_colors,
            pred_instances,
            class_names,
            rle_decode,
            point_size=args.point_size,
        )


if __name__ == "__main__":
    main()