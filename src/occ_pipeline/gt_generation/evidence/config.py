"""Validated configuration and a single, explicit grid convention."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class GridSpec:
    minimum: tuple[float, float, float] = (-20., -20., -2.)
    maximum: tuple[float, float, float] = (20., 20., 4.)
    voxel_size: float = 0.5

    def __post_init__(self):
        lo, hi = np.asarray(self.minimum), np.asarray(self.maximum)
        if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([*lo, *hi, self.voxel_size]).all():
            raise ValueError("Grid requires three finite coordinates per bound and a finite voxel size")
        if self.voxel_size <= 0 or np.any(hi <= lo):
            raise ValueError("Grid maximum must exceed minimum and voxel size must be positive")
        cells = (hi - lo) / self.voxel_size
        if not np.allclose(cells, np.round(cells), rtol=0, atol=1e-8):
            raise ValueError("Grid extent must be an integer multiple of voxel_size")
        if np.prod(cells) > 2_000_000:
            raise ValueError("This dense CPU baseline limits grids to 2 million voxels")

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(round((b - a) / self.voxel_size)) for a, b in zip(self.minimum, self.maximum))

    def contains(self, points: np.ndarray) -> np.ndarray:
        return ((points >= self.minimum) & (points < self.maximum)).all(axis=-1)

    def indices(self, points: np.ndarray) -> np.ndarray:
        return np.floor((points - self.minimum) / self.voxel_size).astype(np.int64)

    def centers(self, indices: np.ndarray) -> np.ndarray:
        return np.asarray(self.minimum) + (indices + 0.5) * self.voxel_size

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self)))


@dataclass(frozen=True)
class Config:
    grid: GridSpec
    min_range_m: float = 1.0
    max_range_m: float = 50.0
    image_size: tuple[int, int] = (96, 160)
    feature_channels: int = 24
    latent_shape: tuple[int, int, int] = (10, 10, 3)
    positive_weight: float = 3.0
    seed: int = 17
    num_sweeps: int = 1
    motion_compensation: bool = False
    box_margin_m: float = 0.2
    box_fill: bool = False
    ego_box_min: tuple | None = None
    ego_box_max: tuple | None = None
    poisson_enabled: bool = False
    poisson_depth: int = 7
    poisson_trim_quantile: float = 0.1
    poisson_support_m: float = 1.0
    poisson_max_points: int = 100_000
    min_hits_per_voxel: int = 1

    def __post_init__(self):
        if not np.isfinite([self.min_range_m, self.max_range_m, self.positive_weight]).all():
            raise ValueError("Range limits and positive_weight must be finite")
        if not 0 <= self.min_range_m < self.max_range_m or self.positive_weight <= 0:
            raise ValueError("Invalid range limits or positive_weight")
        if len(self.image_size) != 2 or len(self.latent_shape) != 3:
            raise ValueError("image_size must have two axes and latent_shape three")
        if any(not isinstance(n, int) or n <= 0 for n in (*self.image_size, *self.latent_shape, self.feature_channels)):
            raise ValueError("Model dimensions must be positive integers")
        if min(self.image_size) < 4:
            raise ValueError("Images must be at least 4 pixels per axis for the encoder")
        if type(self.num_sweeps) is not int or not 1 <= self.num_sweeps <= 100:
            raise ValueError("num_sweeps must be an integer from 1 to 100")
        if type(self.min_hits_per_voxel) is not int or self.min_hits_per_voxel < 1:
            raise ValueError("min_hits_per_voxel must be a positive integer")
        for name in ("motion_compensation", "box_fill", "poisson_enabled"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if not np.isfinite([self.box_margin_m, self.poisson_trim_quantile, self.poisson_support_m]).all():
            raise ValueError("GT reconstruction settings must be finite")
        if self.box_margin_m < 0 or not 0 <= self.poisson_trim_quantile < 1 or self.poisson_support_m <= 0:
            raise ValueError("Invalid box margin, Poisson trim quantile or support distance")
        if type(self.poisson_depth) is not int or not 4 <= self.poisson_depth <= 10:
            raise ValueError("poisson_depth must be an integer from 4 to 10")
        if type(self.poisson_max_points) is not int or self.poisson_max_points < 100:
            raise ValueError("poisson_max_points must be an integer >= 100")
        if (self.ego_box_min is None) != (self.ego_box_max is None):
            raise ValueError("Set both ego_box_min and ego_box_max, or neither")
        if self.ego_box_min is not None:
            lo, hi = np.asarray(self.ego_box_min), np.asarray(self.ego_box_max)
            if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([*lo, *hi]).all() or np.any(hi <= lo):
                raise ValueError("Invalid ego body exclusion box")

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self)))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or set(raw) - {"grid", "gt", "model", "seed"}:
        raise ValueError("Configuration must contain only grid, gt, model, and seed sections")
    allowed = {"grid": {"minimum", "maximum", "voxel_size"},
               "gt": {"min_range_m", "max_range_m", "num_sweeps", "motion_compensation", "box_margin_m",
                      "box_fill", "ego_box_min", "ego_box_max", "poisson_enabled", "poisson_depth",
                      "poisson_trim_quantile", "poisson_support_m", "poisson_max_points", "min_hits_per_voxel"},
               "model": {"image_size", "feature_channels", "latent_shape", "positive_weight"}}
    for section, keys in allowed.items():
        value = raw.get(section, {})
        if not isinstance(value, dict) or set(value) - keys:
            raise ValueError(f"Invalid {section} section; allowed keys: {sorted(keys)}")
    if not isinstance(raw.get("seed", 17), int) or not 0 <= raw.get("seed", 17) < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    try:
        grid = GridSpec(**raw.get("grid", {}))
        return Config(grid=grid, **raw.get("gt", {}), **raw.get("model", {}), seed=raw.get("seed", 17))
    except (TypeError, OverflowError) as exc:
        raise ValueError(f"Invalid configuration value: {exc}") from exc

