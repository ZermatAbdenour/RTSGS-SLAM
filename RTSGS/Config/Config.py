from __future__ import annotations

from pathlib import Path

import numpy as np
from omegaconf import OmegaConf, DictConfig


_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


def load_config(path: str, overrides: dict | None = None) -> "Config":
    base_cfg = OmegaConf.load(_CONFIGS_DIR / "base.yaml")

    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path

    overlay_cfg = OmegaConf.load(config_path)
    overlay_cfg.pop("defaults", None)
    cfg = OmegaConf.merge(base_cfg, overlay_cfg)

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))

    OmegaConf.resolve(cfg)
    return Config(cfg)


class Config:
    def __init__(self, cfg: DictConfig):
        self._cfg = cfg

    # ── Structured access ─────────────────────────────────────────────────
    @property
    def camera(self) -> DictConfig:
        return self._cfg.camera

    @property
    def gaussian_splatting(self) -> DictConfig:
        return self._cfg.gaussian_splatting

    @property
    def tracker(self) -> DictConfig:
        return self._cfg.tracker

    @property
    def segmentation(self) -> DictConfig:
        return self._cfg.segmentation

    @property
    def instance(self) -> DictConfig:
        return self._cfg.instance

    @property
    def scenegraph(self) -> DictConfig:
        return self._cfg.scenegraph

    # ── Intrinsics helpers ────────────────────────────────────────────────
    def get_rgb_intrinsics(self) -> np.ndarray:
        c = self._cfg.camera.rgb
        return np.array(
            [[c.fx, 0, c.cx],
             [0, c.fy, c.cy],
             [0, 0, 1]],
            dtype=np.float32,
        )

    def get_depth_intrinsics(self) -> np.ndarray:
        c = self._cfg.camera.depth
        return np.array(
            [[c.fx, 0, c.cx],
             [0, c.fy, c.cy],
             [0, 0, 1]],
            dtype=np.float32,
        )

    def get_rgb_size(self) -> tuple[int, int]:
        c = self._cfg.camera.rgb
        return int(c.width), int(c.height)

    def get_depth_size(self) -> tuple[int, int]:
        c = self._cfg.camera.depth
        return int(c.width), int(c.height)

    def get_T_depth_to_rgb(self) -> np.ndarray:
        raw = self._cfg.camera.T_depth_to_rgb
        if raw is None:
            return np.eye(4, dtype=np.float32)
        T = np.asarray(raw, dtype=np.float32)
        if T.shape != (4, 4):
            return np.eye(4, dtype=np.float32)
        return T

    def get_T_rgb_to_depth(self) -> np.ndarray:
        T_d2r = self.get_T_depth_to_rgb()
        try:
            return np.linalg.inv(T_d2r).astype(np.float32)
        except np.linalg.LinAlgError:
            return np.eye(4, dtype=np.float32)

    # ── Runtime mutation (used by ScanNet loader to override per-scene intrinsics) ─
    _FLAT_KEY_MAP = {
        "rgb_fx": "camera.rgb.fx",
        "rgb_fy": "camera.rgb.fy",
        "rgb_cx": "camera.rgb.cx",
        "rgb_cy": "camera.rgb.cy",
        "rgb_width": "camera.rgb.width",
        "rgb_height": "camera.rgb.height",
        "depth_fx": "camera.depth.fx",
        "depth_fy": "camera.depth.fy",
        "depth_cx": "camera.depth.cx",
        "depth_cy": "camera.depth.cy",
        "depth_width": "camera.depth.width",
        "depth_height": "camera.depth.height",
        "depth_scale": "camera.depth_scale",
        "T_depth_to_rgb": "camera.T_depth_to_rgb",
    }

    def set(self, key: str, value) -> None:
        path = self._FLAT_KEY_MAP.get(key, key)
        OmegaConf.update(self._cfg, path, value)

    def to_dict(self) -> dict:
        return OmegaConf.to_container(self._cfg, resolve=True)
