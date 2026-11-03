#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from occ_pipeline.config import load_config
from occ_pipeline.data.nuscenes_adapter import NuScenesSampleProvider, SyntheticSampleProvider
from occ_pipeline.gt_generation.generate import generate_ground_truth
from occ_pipeline.gt_generation.voxel_grid import VoxelGridSpec
from occ_pipeline.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate occupancy GT artifacts.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--nuscenes-root", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--synthetic", action="store_true", help="Use deterministic synthetic point clouds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_samples is not None and args.max_samples<1:raise ValueError('max-samples must be positive')
    config = load_config(args.config)
    set_seed(int(config["project"]["seed"]))

    output_dir = args.output_dir or config["paths"]["output_dir"]
    max_samples = args.max_samples or int(config["nuscenes"]["max_samples"])
    spec = VoxelGridSpec.from_config(config["grid"])

    nuscenes_root = Path(args.nuscenes_root or config["paths"]["nuscenes_root"])
    use_synthetic = args.synthetic

    if use_synthetic:
        provider = SyntheticSampleProvider(max_samples=max_samples, seed=int(config["project"]["seed"]))
    else:
        provider = NuScenesSampleProvider(
            dataroot=nuscenes_root,
            version=args.version or config["nuscenes"]["version"],
            split=args.split or config["nuscenes"]["split"],
            max_samples=max_samples,
            num_sweeps=int(config["nuscenes"]["num_sweeps"]),
            gt_config=config['gt'],
        )

    written = generate_ground_truth(
        provider.iter_samples(),
        spec=spec,
        output_dir=output_dir,
        min_points_per_voxel=int(config["grid"]["min_points_per_voxel"]),
        enable_ray_casting=bool(config["gt"]["ray_casting"]),
    )

    print(f"Wrote {len(written)} occupancy GT artifact(s) to {Path(output_dir).resolve()}")
    for path in written[:5]:
        print(f"  - {path}")


if __name__ == "__main__":
    main()

