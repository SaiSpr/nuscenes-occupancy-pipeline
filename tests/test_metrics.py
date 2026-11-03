import torch

from occ_pipeline.training.metrics import binary_occupancy_metrics


def test_binary_metrics_known_counts():
    logits = torch.tensor([[[[[8.0, 8.0], [-8.0, -8.0]]]]])
    target = torch.tensor([[[[[1.0, 0.0], [1.0, 0.0]]]]])

    metrics = binary_occupancy_metrics(logits, target, threshold=0.5)

    assert metrics["tp"] == 1.0
    assert metrics["fp"] == 1.0
    assert metrics["fn"] == 1.0
    assert metrics["tn"] == 1.0
    assert metrics["iou"] == 1.0 / 3.0
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5

