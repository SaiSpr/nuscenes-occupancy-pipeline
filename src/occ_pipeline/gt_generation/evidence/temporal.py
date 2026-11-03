"""Tracked oriented boxes and time interpolation, without devkit dependencies."""

from dataclasses import dataclass
import numpy as np

from occ_pipeline.gt_generation.evidence.geometry import rigid_transform


def slerp(a, b, alpha):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    dot = float(a @ b)
    if dot < 0:
        b, dot = -b, -dot
    if dot > .9995:
        q = a + alpha * (b - a)
        return q / np.linalg.norm(q)
    theta = np.arccos(np.clip(dot, -1, 1))
    return (np.sin((1-alpha)*theta)*a + np.sin(alpha*theta)*b) / np.sin(theta)


@dataclass
class Box:
    instance: str
    center: np.ndarray
    rotation: np.ndarray
    half_size: np.ndarray  # local X=length, Y=width, Z=height

    @classmethod
    def from_annotation(cls, record, reference_from_global):
        pose = reference_from_global @ rigid_transform(record['translation'], record['rotation'])
        width, length, height = record['size']
        return cls(record['instance_token'], pose[:3, 3], pose[:3, :3], np.array([length, width, height])/2)

    def contains(self, points, margin=0.):
        return (np.abs((points-self.center) @ self.rotation) <= self.half_size+margin).all(axis=1)

    def reposed(self, points, destination):
        return ((points-self.center) @ self.rotation) @ destination.rotation.T + destination.center


def interpolate_annotation(previous, current, timestamp, previous_time, current_time):
    """Interpolate only bracketed timestamps. Never extrapolate an object track."""
    if not previous_time <= timestamp <= current_time or current_time <= previous_time:
        return None
    alpha = (timestamp-previous_time)/(current_time-previous_time)
    return {**current,
            'translation': ((1-alpha)*np.array(previous['translation'])+alpha*np.array(current['translation'])).tolist(),
            'rotation': slerp(previous['rotation'], current['rotation'], alpha).tolist(),
            'size': ((1-alpha)*np.array(previous['size'])+alpha*np.array(current['size'])).tolist()}


def box_center_mask(boxes, grid):
    """Annotation prior: cells whose centers lie inside boxes (not exact solids)."""
    mask = np.zeros(grid.shape, dtype=bool)
    for box in boxes:
        radius = np.abs(box.rotation) @ box.half_size
        lo = np.maximum(0, np.floor((box.center-radius-np.array(grid.minimum))/grid.voxel_size).astype(int))
        hi = np.minimum(grid.shape, np.ceil((box.center+radius-np.array(grid.minimum))/grid.voxel_size).astype(int))
        if np.any(hi <= lo):
            continue
        indices = np.stack(np.meshgrid(*(np.arange(a,b) for a,b in zip(lo,hi)), indexing='ij'), -1).reshape(-1,3)
        selected = indices[box.contains(grid.centers(indices))]
        if len(selected):
            mask[tuple(selected.T)] = True
    return mask
