"""CLI: render GT vs predicted occupancy grids for inspection.

Works headless (no display required) — outputs ASCII art to stdout and
optionally saves PNG slices if matplotlib is available.

Usage:
    python scripts/visualize.py --checkpoint artifacts/checkpoints/checkpoint_best.pt \\
        --gt-dir artifacts/occupancy_gt_val --n-samples 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from occ_pipeline.config import load_config
from occ_pipeline.data.occupancy_dataset import OccupancyDataset
from occ_pipeline.training.pipeline import build_model


# ── ASCII rendering ────────────────────────────────────────────────────────────

_BINARY_CHARS = {0: ".", 1: "#"}
_THREE_CLASS_CHARS = {0: " ", 1: "#", 2: "-"}  # unknown=space, occupied=#, free=-


def _topdown_ascii(grid_zyx: np.ndarray, chars: dict) -> list[str]:
    """Max-project along Z, return list of row strings."""
    topdown = np.where((grid_zyx==1).any(0),1,np.where((grid_zyx==2).any(0),2,0))
    return ["".join(chars.get(int(v), "?") for v in row) for row in topdown[::-1]]


def _zslice_ascii(grid_zyx: np.ndarray, z: int, chars: dict) -> list[str]:
    """Single Z-slice as ASCII."""
    return ["".join(chars.get(int(v), "?") for v in row) for row in grid_zyx[z][::-1]]


def _side_by_side(left_rows: list[str], right_rows: list[str], label_l: str, label_r: str) -> str:
    width_l = max(len(r) for r in left_rows) if left_rows else 0
    header = f"{label_l:<{width_l}}   {label_r}"
    sep = "-" * len(header)
    lines = [header, sep]
    for l, r in zip(left_rows, right_rows):
        lines.append(f"{l:<{width_l}}   {r}")
    return "\n".join(lines)


# ── Inference helpers ──────────────────────────────────────────────────────────

def _predict(model: torch.nn.Module, inp: torch.Tensor, num_classes: int, decision=None) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(inp.unsqueeze(0))  # (1, C, Z, Y, X)
    if isinstance(logits,tuple):logits=logits[0]
    if decision:
        from occ_pipeline.training.inference import apply_inference_decision
        logits=apply_inference_decision(logits,inp.unsqueeze(0),decision)
    if num_classes == 1:
        pred = (torch.sigmoid(logits[0, 0]) >= 0.5).numpy().astype(np.uint8)
    else:
        pred = logits[0].argmax(dim=0).numpy().astype(np.uint8)  # (Z, Y, X)
    return pred


def _gt_array(target: torch.Tensor, num_classes: int) -> np.ndarray:
    if num_classes == 1:
        return (target[0] >= 0.5).numpy().astype(np.uint8)  # (Z, Y, X)
    return target.numpy().astype(np.uint8)  # (Z, Y, X)


# ── PNG export (optional) ──────────────────────────────────────────────────────

def _try_save_png(gt: np.ndarray, pred: np.ndarray, sample_idx: int, out_dir: Path, num_classes: int) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    cmap = "Blues" if num_classes == 1 else ListedColormap(['#1f2b3e','#f1a33c','#bad7e1'])
    vmax = 1 if num_classes == 1 else 2

    gt_td = np.where((gt==1).any(0),1,np.where((gt==2).any(0),2,0))
    pred_td = np.where((pred==1).any(0),1,np.where((pred==2).any(0),2,0))

    _fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(gt_td, origin="lower", cmap=cmap, vmin=0, vmax=vmax)
    axes[0].set_title("GT (top-down)")
    axes[1].imshow(pred_td, origin="lower", cmap=cmap, vmin=0, vmax=vmax)
    axes[1].set_title("Predicted (top-down)")
    plt.suptitle(f"Sample {sample_idx} | GT mask completion")
    if num_classes==3:
        _fig.legend(handles=[Patch(color=c,label=t) for c,t in zip(
            ['#1f2b3e','#f1a33c','#bad7e1'],['Unknown','Occupied','Free'])],loc='lower center',ncol=3)
        _fig.subplots_adjust(bottom=.13)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"sample_{sample_idx:03d}.png"
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  PNG saved → {path}")
    return True


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise GT vs predicted occupancy")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--z-slice", type=int, default=None, help="Which Z to render (default: middle)")
    parser.add_argument("--png-dir", default=None, help="Directory to save PNG slices (optional)")
    args = parser.parse_args()

    config = load_config(args.config)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model_config=ckpt.get('config',{}).get('model',config['model'])
    num_classes=int(model_config.get('out_channels',1))
    chars = _THREE_CLASS_CHARS if num_classes > 1 else _BINARY_CHARS
    _, model = build_model(dict(model_config))
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Checkpoint: epoch {ckpt['epoch']}, val_loss {ckpt.get('val_loss')}")

    protocol=ckpt.get('protocol',{})
    ds = OccupancyDataset(args.gt_dir, num_classes=num_classes,seed=protocol.get('validation_seed',protocol.get('seed',42)))
    n = min(args.n_samples, len(ds))
    print(f"Rendering {n} samples from {args.gt_dir}  (num_classes={num_classes})\n")

    legend = "Legend: # occupied" + (" - free   blank unknown" if num_classes > 1 else "  . other")
    print(legend)

    for i in range(n):
        sample = ds[i]
        inp = sample["input"]  # (1, Z, Y, X)
        target = sample["target"]

        gt = _gt_array(target, num_classes)
        pred = _predict(model, inp, num_classes,ckpt.get('inference_decision'))
        z_idx = args.z_slice if args.z_slice is not None else gt.shape[0] // 2

        print(f"\n{'='*60}")
        print(f"Sample {i}  |  {Path(sample['path']).stem[:40]}")
        print(f"  Grid shape (Z,Y,X): {gt.shape}")

        print("\n[Top-down view]")
        gt_td = _topdown_ascii(gt, chars)
        pred_td = _topdown_ascii(pred, chars)
        print(_side_by_side(gt_td, pred_td, "GT", "Predicted"))

        print(f"\n[Z-slice z={z_idx}]")
        gt_sl = _zslice_ascii(gt, z_idx, chars)
        pred_sl = _zslice_ascii(pred, z_idx, chars)
        print(_side_by_side(gt_sl, pred_sl, "GT", "Predicted"))

        if args.png_dir:
            _try_save_png(gt, pred, i, Path(args.png_dir), num_classes)


if __name__ == "__main__":
    main()

