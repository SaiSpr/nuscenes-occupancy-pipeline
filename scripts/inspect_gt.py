#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from occ_pipeline.utils.io import load_occupancy_npz
from occ_pipeline.visualization.ascii import ascii_topdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one generated occupancy artifact.")
    parser.add_argument("artifact", help="Path to .npz file from scripts/generate_gt.py")
    args = parser.parse_args()

    artifact = load_occupancy_npz(args.artifact)
    print(json.dumps(artifact["metadata"], indent=2, sort_keys=True))
    print("\nTop-down occupancy preview (# = occupied):")
    print(ascii_topdown(artifact["occupancy"]))


if __name__ == "__main__":
    main()

