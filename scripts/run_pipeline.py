#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from occ_pipeline.config import load_config
from occ_pipeline.training.pipeline import run_forward_loss_demo
from occ_pipeline.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an occupancy model forward pass and compute its loss.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--gt-dir", default=None)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["project"]["seed"]))

    result = run_forward_loss_demo(
        gt_dir=args.gt_dir or config["paths"]["output_dir"],
        model_config=config["model"],
        batch_size=int(config["pipeline"]["batch_size"]),
        device=args.device or config["pipeline"]["device"],
        threshold=float(config["pipeline"]["threshold"]),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

