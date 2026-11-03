import numpy as np

from occ_pipeline.gt_generation.voxel_grid import VoxelGridSpec, voxelize_points


def test_voxel_indices_and_half_open_boundary():
    spec = VoxelGridSpec(-1, 1, -1, 1, -1, 1, 1.0)
    points = np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, 0.5, 0.5],
            [1.0, 0.0, 0.0],
            [np.nan, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    occupancy, counts, metadata = voxelize_points(points, spec)

    assert occupancy.shape == (2, 2, 2)
    assert int(occupancy.sum()) == 2
    assert int(counts.sum()) == 2
    assert metadata["in_range_point_count"] == 2
    assert metadata["out_of_range_point_count"] == 1
    assert metadata["finite_point_count"] == 3


def test_min_points_per_voxel():
    spec = VoxelGridSpec(0, 2, 0, 2, 0, 2, 1.0)
    points = np.array([[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [1.2, 1.2, 1.2]], dtype=np.float32)

    occupancy, _, _ = voxelize_points(points, spec, min_points_per_voxel=2)

    assert int(occupancy.sum()) == 1

