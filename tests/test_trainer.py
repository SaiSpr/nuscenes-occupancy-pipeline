from pathlib import Path

from occ_pipeline.data.nuscenes_adapter import SyntheticSampleProvider
from occ_pipeline.gt_generation.generate import generate_ground_truth
from occ_pipeline.gt_generation.voxel_grid import VoxelGridSpec
from occ_pipeline.training.trainer import train

TINY_SPEC = VoxelGridSpec(-20, 20, -20, 20, -3, 3, 2.0)
TINY_MODEL = {"name": "tiny_3d_cnn", "in_channels": 1, "hidden_channels": 2, "out_channels": 1}
TINY_TRAINING = {
    "epochs": 2,
    "lr": 1e-3,
    "val_split": 0.34,
    "num_workers": 0,
    "batch_size": 1,
    "log_interval": 1,
}


def _gen_gt(tmp_path: Path, n: int) -> Path:
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    provider = SyntheticSampleProvider(max_samples=n, seed=7)
    generate_ground_truth(provider.iter_samples(), TINY_SPEC, gt_dir)
    return gt_dir


def test_train_runs_and_loss_is_finite(tmp_path):
    gt_dir = _gen_gt(tmp_path, 3)
    ckpt_dir = tmp_path / "checkpoints"

    result = train(
        gt_dir=gt_dir,
        model_config=TINY_MODEL,
        training_config={**TINY_TRAINING, "checkpoint_dir": str(ckpt_dir)},
        device="cpu",
    )

    assert result["epochs"] == 2
    assert len(result["history"]["train_loss"]) == 2
    assert all(isinstance(v, float) and v >= 0 for v in result["history"]["train_loss"])
    assert (ckpt_dir / "checkpoint_latest.pt").exists()
    assert (ckpt_dir / "checkpoint_best.pt").exists()


def test_train_single_sample_no_val_crash(tmp_path):
    gt_dir = _gen_gt(tmp_path, 1)
    ckpt_dir = tmp_path / "checkpoints"

    result = train(
        gt_dir=gt_dir,
        model_config=TINY_MODEL,
        training_config={**TINY_TRAINING, "checkpoint_dir": str(ckpt_dir)},
        device="cpu",
    )

    assert result["val_samples"] == 0
    assert all(v is None for v in result["history"]["val_loss"])
    assert (ckpt_dir / "checkpoint_latest.pt").exists()

