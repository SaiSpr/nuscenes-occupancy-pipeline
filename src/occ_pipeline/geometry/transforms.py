from __future__ import annotations

import numpy as np


def transform_points(points_xyz: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Apply p_out = R * p_in + t for row-major point arrays of shape [N, 3]."""
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    rotation = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    translation = np.asarray(translation, dtype=np.float32).reshape(1, 3)
    return points_xyz @ rotation.T + translation


def invert_rigid_transform(points_xyz: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Exact inverse of transform_points: R.T @ (p - t) for row-major [N, 3] arrays."""
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    rotation = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    translation = np.asarray(translation, dtype=np.float32).reshape(1, 3)
    return (points_xyz - translation) @ rotation


def filter_finite_points(points_xyz: np.ndarray) -> np.ndarray:
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    if points_xyz.size == 0:
        return points_xyz.reshape(0, 3)
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"Expected points shape [N, 3], got {points_xyz.shape}")
    return points_xyz[np.isfinite(points_xyz).all(axis=1)]

