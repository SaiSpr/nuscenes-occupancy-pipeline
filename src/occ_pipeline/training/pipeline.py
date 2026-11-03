from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from occ_pipeline.data.occupancy_dataset import OccupancyDataset
from occ_pipeline.models.occupancy_cnn import TinyOccupancyCNN
from occ_pipeline.models.occupancy_unet import OccupancyUNet3D
from occ_pipeline.training.losses import occupancy_bce_loss, occupancy_ce_loss
from occ_pipeline.training.metrics import binary_occupancy_metrics, multiclass_occupancy_metrics


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    return torch.device(requested)


def build_model(model_config: dict[str, Any]) -> tuple[str, torch.nn.Module]:
    config = dict(model_config)
    model_name = config.pop("name", "tiny_3d_cnn")
    if config.get('in_channels',1) != 1 or config.get('out_channels',1) not in (1,3):
        raise ValueError('Completion dataset requires in_channels=1 and out_channels=1 or 3')
    if model_name == 'unet_3d':
        config.pop('hidden_channels',None)
        return model_name, OccupancyUNet3D(**config)
    if model_name == 'tiny_3d_cnn':
        config.pop('base_channels',None);config.pop('multi_level',None)
        return model_name, TinyOccupancyCNN(**config)
    raise ValueError(f'Unsupported model.name={model_name!r}')


def run_forward_loss_demo(
    gt_dir: str | Path,
    model_config: dict[str, Any],
    batch_size: int = 1,
    device: str = "cpu",
    threshold: float = 0.5,
) -> dict[str, Any]:
    torch_device = resolve_device(device)
    num_classes=int(model_config.get('out_channels',1))
    dataset = OccupancyDataset(gt_dir,num_classes=num_classes)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model_name, model = build_model(model_config)
    model = model.to(torch_device)
    model.eval()

    batch = next(iter(loader))
    inputs = batch["input"].to(torch_device)
    targets = batch["target"].to(torch_device)

    with torch.no_grad():
        logits = model(inputs)
        if isinstance(logits,tuple):logits=logits[0]
        observed=batch['observed'].to(torch_device)
        loss = occupancy_ce_loss(logits, targets) if num_classes==3 else occupancy_bce_loss(logits, targets,observed)
        metrics = multiclass_occupancy_metrics(logits, targets) if num_classes==3 else binary_occupancy_metrics(logits, targets, threshold=threshold,observed=observed)

    return {
        "device": str(torch_device),
        "model_name": model_name,
        "model_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "batch_size": int(inputs.shape[0]),
        "input_shape": list(inputs.shape),
        "target_shape": list(targets.shape),
        "logit_shape": list(logits.shape),
        "loss": float(loss.item()),
        "metrics": metrics,
    }

