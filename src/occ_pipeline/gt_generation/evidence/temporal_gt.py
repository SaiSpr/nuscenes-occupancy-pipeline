"""Multi-sweep ray evidence and separately identified completion proposals."""

import numpy as np
from occ_pipeline.gt_generation.evidence.ground_truth import UNKNOWN, FREE, OCCUPIED, OccupancyResult
from occ_pipeline.gt_generation.evidence.temporal import box_center_mask


def carve_free(points, origin, grid, max_range):
    """Batched clipped DDA; step all tied axes, ignoring zero-volume contacts."""
    free = np.zeros(grid.shape, dtype=bool)
    dims = np.array(grid.shape)
    origin_grid = (origin-np.array(grid.minimum))/grid.voxel_size
    for offset in range(0, len(points), 20_000):
        ends = points[offset:offset+20_000]
        delta = ends-origin
        ranges = np.linalg.norm(delta, axis=1)
        ends = origin + delta*np.minimum(1., max_range/np.maximum(ranges, 1e-12))[:,None]
        delta = (ends-origin)/grid.voxel_size
        parallel = np.abs(delta) < 1e-15
        safe = np.where(parallel, 1., delta)
        a, b = -origin_grid/safe, (dims-origin_grid)/safe
        low = np.where(parallel, np.where((origin_grid >= 0) & (origin_grid < dims), -np.inf, np.inf), np.minimum(a,b))
        high = np.where(parallel, np.where((origin_grid >= 0) & (origin_grid < dims), np.inf, -np.inf), np.maximum(a,b))
        enter, leave = np.maximum(0., low.max(1)), np.minimum(1., high.min(1))
        active = (leave-enter > 1e-12) & (ranges > 1e-12)
        # Avoid casting infinities from rejected rays to integers.
        probe = np.zeros_like(enter)
        probe[active] = enter[active]+np.minimum(1e-9, (leave[active]-enter[active])/2)
        cell = np.clip(np.floor(origin_grid+probe[:,None]*delta), 0, dims-1).astype(int)
        step = np.sign(delta).astype(int)
        boundary = cell+(step > 0)
        next_t = np.where(parallel, np.inf, (boundary-origin_grid)/safe)
        t_step = np.where(parallel, np.inf, np.abs(1/safe))
        for _ in range(int(dims.sum())+3):
            rows = np.flatnonzero(active)
            if not len(rows):
                break
            free[tuple(cell[rows].T)] = True
            crossing = next_t.min(1)
            advance = active & (crossing < leave-1e-12)
            ties = np.isclose(next_t, crossing[:,None], rtol=0, atol=1e-12) & advance[:,None]
            cell += step*ties
            next_t = np.where(ties, next_t+t_step, next_t)
            active = advance & ((cell >= 0) & (cell < dims)).all(1)
    return free


def generate_temporal_occupancy(sweeps, boxes, config):
    grid = config.grid
    labels = np.full(grid.shape, UNKNOWN, dtype=np.int8)
    provenance = np.zeros(grid.shape, dtype=np.uint8)
    hits, moved_hits = np.zeros(grid.shape, np.int32), np.zeros(grid.shape, np.int32)
    for sweep in sweeps:
        points, moved, origin = sweep['points'], sweep['moved'], sweep['origin']
        distances = sweep.get('ranges', np.linalg.norm(points-origin, axis=1))
        valid = np.isfinite(points).all(1) & (distances >= config.min_range_m) & (distances > 1e-12)
        free = carve_free(points[valid & ~moved], origin, grid, config.max_range_m)
        labels[free], provenance[free] = FREE, 1
        hit_mask = valid & (distances <= config.max_range_m) & grid.contains(points)
        for mask, target in ((hit_mask & ~moved, hits), (hit_mask & moved, moved_hits)):
            indices = grid.indices(points[mask])
            if len(indices):
                np.add.at(target, tuple(indices.T), 1)
    accepted = hits+moved_hits >= config.min_hits_per_voxel
    moved_only = accepted & (hits == 0)
    direct = accepted & (hits > 0)
    labels[moved_only], provenance[moved_only] = OCCUPIED, 3
    labels[direct], provenance[direct] = OCCUPIED, 2
    observed = labels != UNKNOWN
    # Completion proposals can only add unknown cells, never overwrite ray evidence.
    proposal_kind = np.zeros(grid.shape, dtype=np.uint8)
    if config.box_fill:
        proposal_kind[box_center_mask(boxes, grid) & ~observed] = 4
    poisson_stats = {'status': 'disabled'}
    if config.poisson_enabled:
        from occ_pipeline.gt_generation.evidence.reconstruction import poisson_candidates
        shell, poisson_stats = poisson_candidates(sweeps, grid, config)
        proposal_kind[shell & ~observed & (proposal_kind == 0)] = 5
    proposed_labels = labels.copy()
    proposed_labels[proposal_kind != 0] = OCCUPIED
    stats = {'input_points': sum(len(s['points']) for s in sweeps),
             'sweeps_used': len(sweeps), 'ego_points_removed': sum(s.get('ego_removed',0) for s in sweeps),
             'motion_compensated_voxels': int((provenance == 3).sum()),
             'occupied_voxels': int((labels == OCCUPIED).sum()), 'free_voxels': int((labels == FREE).sum()),
             'unknown_voxels': int((labels == UNKNOWN).sum()), 'observed_fraction': float(observed.mean()),
             'box_proposal_voxels': int((proposal_kind == 4).sum()),
             'poisson_proposal_voxels': int((proposal_kind == 5).sum()), 'poisson': poisson_stats}
    return OccupancyResult(labels, observed, stats), {
        'provenance': provenance, 'proposed_labels': proposed_labels, 'proposal_kind': proposal_kind}
