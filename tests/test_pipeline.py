from occ_pipeline.data.nuscenes_adapter import SyntheticSampleProvider
from occ_pipeline.gt_generation.generate import generate_ground_truth
from occ_pipeline.gt_generation.voxel_grid import VoxelGridSpec
from occ_pipeline.training.pipeline import run_forward_loss_demo


def test_synthetic_gt_and_pipeline(tmp_path):
    spec = VoxelGridSpec(-20, 20, -20, 20, -3, 3, 2.0)
    provider = SyntheticSampleProvider(max_samples=2, seed=7)
    written = generate_ground_truth(provider.iter_samples(), spec, tmp_path)

    assert len(written) == 2

    result = run_forward_loss_demo(
        tmp_path,
        model_config={"name": "tiny_3d_cnn", "in_channels": 1, "hidden_channels": 2, "out_channels": 1},
        batch_size=1,
        device="cpu",
    )

    assert result["model_name"] == "tiny_3d_cnn"
    assert result["model_parameter_count"] > 0
    assert result["input_shape"] == [1, 1, 3, 20, 20]
    assert result["target_shape"] == [1, 1, 3, 20, 20]
    assert result["loss"] >= 0.0
    assert "iou" in result["metrics"]

