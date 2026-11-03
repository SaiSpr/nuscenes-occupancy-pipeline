from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from occ_pipeline.geometry.transforms import filter_finite_points


@dataclass(frozen=True)
class VoxelGridSpec:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    voxel_size: float

    def __post_init__(self):
        if not np.isfinite([self.x_min,self.x_max,self.y_min,self.y_max,self.z_min,self.z_max,self.voxel_size]).all() or self.voxel_size<=0:
            raise ValueError('Grid coordinates must be finite and voxel size positive')
        _=self.shape_zyx

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "VoxelGridSpec":
        return cls(
            x_min=float(config["x_min"]),
            x_max=float(config["x_max"]),
            y_min=float(config["y_min"]),
            y_max=float(config["y_max"]),
            z_min=float(config["z_min"]),
            z_max=float(config["z_max"]),
            voxel_size=float(config["voxel_size"]),
        )

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        x_bins = self._bins(self.x_min, self.x_max)
        y_bins = self._bins(self.y_min, self.y_max)
        z_bins = self._bins(self.z_min, self.z_max)
        return (z_bins, y_bins, x_bins)

    def metadata(self) -> dict[str, Any]:
        return {
            "axis_order": "zyx",
            "x_range_m": [self.x_min, self.x_max],
            "y_range_m": [self.y_min, self.y_max],
            "z_range_m": [self.z_min, self.z_max],
            "voxel_size_m": self.voxel_size,
            "grid_shape_zyx": list(self.shape_zyx),
        }

    def points_to_indices(self, points_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map ego-frame xyz points to integer zyx voxel indices.

        Points on the max boundary are excluded to keep bins half-open:
        [min, max). This prevents an index equal to the grid size.
        """
        points_xyz = filter_finite_points(points_xyz)
        if points_xyz.shape[0] == 0:
            return np.empty((0, 3), dtype=np.int64), np.zeros((0,), dtype=bool)

        x = np.floor((points_xyz[:, 0] - self.x_min) / self.voxel_size).astype(np.int64)
        y = np.floor((points_xyz[:, 1] - self.y_min) / self.voxel_size).astype(np.int64)
        z = np.floor((points_xyz[:, 2] - self.z_min) / self.voxel_size).astype(np.int64)

        z_bins, y_bins, x_bins = self.shape_zyx
        valid = (
            (x >= 0)
            & (x < x_bins)
            & (y >= 0)
            & (y < y_bins)
            & (z >= 0)
            & (z < z_bins)
        )
        return np.stack([z, y, x], axis=1), valid

    def _bins(self, min_value: float, max_value: float) -> int:
        span = max_value - min_value
        if span <= 0:
            raise ValueError("Voxel range max must be greater than min.")
        bins = span / self.voxel_size
        rounded = int(round(bins))
        if not np.isclose(bins, rounded):
            raise ValueError("Voxel range must be divisible by voxel_size for deterministic shape.")
        return rounded


def voxelize_points(
    points_ego_xyz: np.ndarray,
    spec: VoxelGridSpec,
    min_points_per_voxel: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    indices_zyx, valid_mask = spec.points_to_indices(points_ego_xyz)
    valid_indices = indices_zyx[valid_mask]

    point_counts = np.zeros(spec.shape_zyx, dtype=np.uint32)
    if valid_indices.size:
        z, y, x = valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]
        np.add.at(point_counts, (z, y, x), 1)

    point_counts=np.minimum(point_counts,65535).astype(np.uint16)

    occupancy = (point_counts >= int(min_points_per_voxel)).astype(np.uint8)
    metadata = {
        **spec.metadata(),
        "input_point_count": int(len(points_ego_xyz)),
        "finite_point_count": int(len(filter_finite_points(points_ego_xyz))),
        "in_range_point_count": int(valid_mask.sum()),
        "out_of_range_point_count": int((~valid_mask).sum()),
        "occupied_voxel_count": int(occupancy.sum()),
        "occupancy_ratio": float(occupancy.mean()),
        "min_points_per_voxel": int(min_points_per_voxel),
        "label_policy": "1=occupied_lidar_observation,0=unknown_or_not_observed",
    }
    return occupancy, point_counts, metadata

