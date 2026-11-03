"""Deterministic single-scan occupancy with half-open voxels and clipped rays."""

from dataclasses import dataclass
import math

import numpy as np

from occ_pipeline.gt_generation.evidence.config import GridSpec

UNKNOWN, FREE, OCCUPIED = -1, 0, 1


def ray_voxels(origin, endpoint, grid: GridSpec):
    """Yield cells crossed for positive ray length using 3D DDA.

    Clip the segment to the grid. At an edge/corner crossing, advance all tied
    axes: cells touched only at a point are not evidence of free volume.
    Occupancy of the measured endpoint is handled separately by the caller.
    """
    delta = [float(endpoint[i] - origin[i]) for i in range(3)]
    if not any(delta):
        return
    enter, leave = 0.0, 1.0
    for a in range(3):
        if abs(delta[a]) < 1e-15:
            if not grid.minimum[a] <= origin[a] < grid.maximum[a]:
                return
            continue
        t0 = (grid.minimum[a] - origin[a]) / delta[a]
        t1 = (grid.maximum[a] - origin[a]) / delta[a]
        enter, leave = max(enter, min(t0, t1)), min(leave, max(t0, t1))
    if leave - enter <= 1e-12:
        return
    # Choose the cell just inside the segment, including at negative boundaries.
    probe = enter + min(1e-9, (leave - enter) / 2)
    shape = grid.shape
    cell = [min(shape[a] - 1, max(0, math.floor(
        (origin[a] + probe * delta[a] - grid.minimum[a]) / grid.voxel_size))) for a in range(3)]
    step, t_next, t_step = [], [], []
    for a in range(3):
        sign = 1 if delta[a] > 0 else -1 if delta[a] < 0 else 0
        step.append(sign)
        if sign:
            boundary = grid.minimum[a] + (cell[a] + (sign > 0)) * grid.voxel_size
            t_next.append((boundary - origin[a]) / delta[a])
            t_step.append(grid.voxel_size / abs(delta[a]))
        else:
            t_next.append(math.inf)
            t_step.append(math.inf)
    while all(0 <= cell[a] < shape[a] for a in range(3)):
        yield tuple(cell)
        crossing = min(t_next)
        if crossing >= leave - 1e-12:
            break
        for a in range(3):
            if abs(t_next[a] - crossing) < 1e-12:
                cell[a] += step[a]
                t_next[a] += t_step[a]


@dataclass
class OccupancyResult:
    labels: np.ndarray
    observed: np.ndarray
    stats: dict


def generate_occupancy(points: np.ndarray, origin: np.ndarray, grid: GridSpec,
                       min_range_m: float = 1., max_range_m: float = 50.) -> OccupancyResult:
    points = np.asarray(points, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("Expected points [N,3] and finite origin [3]")
    if not 0 <= min_range_m < max_range_m or not np.isfinite(max_range_m):
        raise ValueError("Invalid ray range limits")
    ranges = np.linalg.norm(points - origin, axis=1)
    valid = np.isfinite(points).all(axis=1) & (ranges >= min_range_m) & (ranges > 1e-12)
    endpoints, ranges = points[valid], ranges[valid]
    labels = np.full(grid.shape, UNKNOWN, dtype=np.int8)
    hits = endpoints[(ranges <= max_range_m) & grid.contains(endpoints)]
    for endpoint, distance in zip(endpoints, ranges):
        stop = origin + (endpoint - origin) * min(1., max_range_m / distance)
        for cell in ray_voxels(origin, stop, grid):
            labels[cell] = FREE
    # Apply all hits last so ray order never erases a measured surface.
    if len(hits):
        indices = grid.indices(hits)
        labels[tuple(indices.T)] = OCCUPIED
    observed = labels != UNKNOWN
    return OccupancyResult(labels, observed, {
        "input_points": len(points), "valid_rays": int(valid.sum()),
        "occupied_voxels": int((labels == OCCUPIED).sum()),
        "free_voxels": int((labels == FREE).sum()),
        "unknown_voxels": int((labels == UNKNOWN).sum()),
        "observed_fraction": float(observed.mean()),
    })
