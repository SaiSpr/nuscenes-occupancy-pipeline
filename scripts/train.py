"""CLI: train the occupancy CNN on pre-generated GT .npz files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from occ_pipeline.config import load_config
from occ_pipeline.training.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train occupancy CNN")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--gt-dir", default=None, help="Directory of .npz GT files")
    parser.add_argument('--val-dir',default=None,help='Separate held-out-scene GT directory')
    parser.add_argument('--epochs',type=int,default=None)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"])
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs is not None:config['training']['epochs']=args.epochs
    gt_dir = args.gt_dir or config["paths"]["output_dir"]
    device = args.device or config["pipeline"]["device"]

    result = train(
        gt_dir=gt_dir,
        model_config=config["model"],
        training_config=config["training"],
        device=device,
        seed=config["project"]["seed"],
        val_dir=args.val_dir,
    )

    print("\n--- Training Summary ---")
    summary = {k: v for k, v in result.items() if k != "history"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

