import numpy as np
import pytest

from occ_pipeline.data.nuscenes_adapter import SyntheticSampleProvider
from occ_pipeline.data.occupancy_dataset import OccupancyDataset
from occ_pipeline.gt_generation.generate import generate_ground_truth
from occ_pipeline.gt_generation.ray_casting import cast_free_space
from occ_pipeline.gt_generation.voxel_grid import VoxelGridSpec
from occ_pipeline.utils.io import load_occupancy_npz

# 10m x 10m x 4m grid with 1m voxels → shape [4, 10, 10], origin at (0,0,0)
SPEC = VoxelGridSpec(0, 10, 0, 10, 0, 4, 1.0)


def test_free_space_marked_along_ray():
    # Hit at x=3.5, y=0.5, z=0.5 → voxel index x=3, y=0, z=0
    # Ray from [0,0,0]: voxels x=0,1,2 should be free; x=3 occupied
    hit = np.array([[3.5, 0.5, 0.5]], dtype=np.float32)
    grid = cast_free_space(hit, SPEC, origin_xyz=np.zeros(3))

    # x=3 voxel (z=0, y=0, x=3) must be occupied
    assert grid[0, 0, 3] == 1
    # voxels x=0,1,2 along the ray must be free
    assert grid[0, 0, 0] == 2
    assert grid[0, 0, 1] == 2
    assert grid[0, 0, 2] == 2


def test_occupied_overrides_free():
    # Two hits on the same x-axis ray: near hit at x=1.5 and far at x=3.5
    # Near hit voxel (x=1) must stay occupied (1), not be overwritten as free
    hits = np.array([[1.5, 0.5, 0.5], [3.5, 0.5, 0.5]], dtype=np.float32)
    grid = cast_free_space(hits, SPEC, origin_xyz=np.zeros(3))

    assert grid[0, 0, 1] == 1  # near hit: occupied, not free
    assert grid[0, 0, 3] == 1  # far hit: occupied


def test_cast_empty_points_returns_unknown():
    empty = np.empty((0, 3), dtype=np.float32)
    grid = cast_free_space(empty, SPEC)
    assert grid.shape == SPEC.shape_zyx
    assert grid.sum() == 0


def test_generate_gt_with_ray_casting_end_to_end(tmp_path):
    spec = VoxelGridSpec(-20, 20, -20, 20, -3, 3, 2.0)
    provider = SyntheticSampleProvider(max_samples=2, seed=7)
    written = generate_ground_truth(
        provider.iter_samples(), spec, tmp_path, enable_ray_casting=True
    )
    assert len(written) == 2

    data = load_occupancy_npz(written[0])
    unique_vals = set(np.unique(data["occupancy"]).tolist())

    # Must have occupied and free labels
    assert 1 in unique_vals, "no occupied voxels found"
    assert 2 in unique_vals, "no free voxels found — ray casting may not be working"

    # label_policy metadata updated
    assert "2=free" in data["metadata"]["label_policy"]
    assert data["metadata"]["ray_casting"] is True

    # OccupancyDataset target must still be binary {0.0, 1.0}
    dataset = OccupancyDataset(tmp_path)
    sample = dataset[0]
    target_vals = set(sample["target"].unique().tolist())
    assert target_vals <= {0.0, 1.0}, f"target not binary: {target_vals}"

