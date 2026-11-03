from __future__ import annotations

import numpy as np
import pytest

from occ_pipeline.data.nuscenes_adapter import SyntheticSampleProvider
from occ_pipeline.geometry.transforms import invert_rigid_transform, transform_points
from occ_pipeline.gt_generation.generate import generate_ground_truth
from occ_pipeline.gt_generation.voxel_grid import VoxelGridSpec
from occ_pipeline.utils.io import load_occupancy_npz


def _small_spec() -> VoxelGridSpec:
    return VoxelGridSpec(
        x_min=-10.0, x_max=10.0,
        y_min=-10.0, y_max=10.0,
        z_min=-3.0, z_max=3.0,
        voxel_size=1.0,
    )


def test_synthetic_sensor_origin_is_zeros():
    provider = SyntheticSampleProvider(max_samples=2, seed=0)
    for sample in provider.iter_samples():
        assert sample.sensor_origin_ego_xyz.shape == (3,)
        np.testing.assert_array_equal(sample.sensor_origin_ego_xyz, np.zeros(3))


def test_invert_rigid_transform_roundtrip():
    rng = np.random.default_rng(42)
    points = rng.standard_normal((100, 3)).astype(np.float32)
    # Build a random valid rotation via QR decomposition
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    R = Q.astype(np.float32)
    t = rng.standard_normal(3).astype(np.float32)

    transformed = transform_points(points, R, t)
    recovered = invert_rigid_transform(transformed, R, t)
    np.testing.assert_allclose(recovered, points, atol=1e-5)


def test_sensor_origin_propagated_to_ray_casting(tmp_path):
    # Use a non-zero sensor origin to verify ray casting uses it
    provider = SyntheticSampleProvider(max_samples=2, seed=7)
    spec = _small_spec()

    written = generate_ground_truth(
        provider.iter_samples(),
        spec=spec,
        output_dir=tmp_path,
        enable_ray_casting=True,
    )

    assert len(written) == 2
    for path in written:
        d = load_occupancy_npz(path)
        occ = d["occupancy"]
        # With ray casting enabled, there must be free-space voxels (label 2)
        assert (occ == 2).sum() > 0, "Expected free-space voxels with ray casting enabled"
        # All values must be in {0, 1, 2}
        assert set(np.unique(occ)).issubset({0, 1, 2})

