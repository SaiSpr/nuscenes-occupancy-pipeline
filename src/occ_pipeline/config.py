from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "project": {"seed": 42},
    "paths": {"nuscenes_root": "/data/nuscenes", "output_dir": "artifacts/occupancy_gt"},
    "nuscenes": {
        "version": "v1.0-mini",
        "split": "mini_train",
        "max_samples": 5,
        "synthetic_if_missing": False,
        "num_sweeps": 5,
    },
    "grid": {
        "x_min": -40.0,
        "x_max": 40.0,
        "y_min": -40.0,
        "y_max": 40.0,
        "z_min": -3.0,
        "z_max": 3.0,
        "voxel_size": 1.0,
        "min_points_per_voxel": 1,
    },
    "gt": {"label_policy": "occupied_unknown", "include_point_counts": True, "ray_casting": False},
    "model": {"name": "tiny_3d_cnn", "in_channels": 1, "hidden_channels": 4, "out_channels": 1},
    "pipeline": {"batch_size": 1, "device": "cpu", "threshold": 0.5},
    "training": {
        "epochs": 10,
        "lr": 1e-3,
        "val_split": 0.2,
        "num_workers": 0,
        "batch_size": 1,
        "checkpoint_dir": "artifacts/checkpoints",
        "log_interval": 1,
    },
}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None) -> dict[str, Any]:
    """Load a YAML config and merge it over safe defaults."""
    if path is None:
        return deepcopy(DEFAULT_CONFIG)
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    config = _deep_update(DEFAULT_CONFIG, loaded)
    validate_config(config)
    return config


def validate_config(config):
    import math
    from occ_pipeline.gt_generation.voxel_grid import VoxelGridSpec
    from occ_pipeline.gt_generation.evidence.config import Config, GridSpec
    if set(config) != set(DEFAULT_CONFIG):raise ValueError('Unexpected configuration section')
    spec=VoxelGridSpec.from_config(config['grid'])
    if type(config['grid']['min_points_per_voxel']) is not int or config['grid']['min_points_per_voxel']<1:
        raise ValueError('min_points_per_voxel must be a positive integer')
    gt={k:v for k,v in config['gt'].items() if k not in ('ray_casting','label_policy','include_point_counts')}
    Config(GridSpec((spec.x_min,spec.y_min,spec.z_min),(spec.x_max,spec.y_max,spec.z_max),spec.voxel_size),
           num_sweeps=config['nuscenes']['num_sweeps'],**gt)
    model=config['model'];training=config['training']
    if model['name'] not in ('tiny_3d_cnn','unet_3d') or model['in_channels']!=1 or model['out_channels'] not in (1,3):
        raise ValueError('Unsupported model or input/output channels')
    if any(type(model.get(k,4)) is not int or model.get(k,4)<1 for k in ('hidden_channels','base_channels')):
        raise ValueError('Model channel dimensions must be positive integers')
    if training['epochs']<1 or not math.isfinite(training['lr']) or training['lr']<=0:
        raise ValueError('Invalid epochs/lr')
    if not 0<=training['val_split']<1 or training['batch_size']<1 or training['num_workers']<0:
        raise ValueError('Invalid training split or loader dimensions')
    if training.get('lr_scheduler','none') not in ('none','cosine','step') or training.get('loss','bce') not in ('bce','focal','ce'):
        raise ValueError('Unknown scheduler/loss')
    if type(config['gt']['ray_casting']) is not bool:raise ValueError('ray_casting must be boolean')

