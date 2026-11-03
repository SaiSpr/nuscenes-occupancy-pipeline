# Reproduction guide

The project includes the modular implementation, combined GT profile, Docker environment, design report, walkthrough and experimental evidence. The [design report](../DESIGN_REPORT.pdf) and [video](../walkthrough.mp4) are available directly; `submission.zip` includes checkpoints, derived grids, captions and complete execution logs.

After cloning, run `python3 scripts/unpack_evidence.py` to verify and restore the bundled artifacts before checkpoint evaluation. Raw nuScenes data is provided separately.

## Run from a clean Docker environment

From the repository root:

```bash
docker build --target reconstruction -t occupancy-pipeline:submission .
docker run --rm occupancy-pipeline:submission python -m pytest
mkdir -p artifacts/reproduction
docker run --rm \
  -v "$PWD/artifacts/accuracy-benchmark-v2:/evidence:ro" \
  -v "$PWD/artifacts/reproduction:/outputs" \
  occupancy-pipeline:submission python scripts/evaluate.py \
  --checkpoint /evidence/recommended-checkpoint.pt \
  --gt-dir /evidence/test --output /outputs/checkpoint-evaluation.json
```

Open3D is pinned to 0.20.0 on Linux ARM64 because 0.19.0 has no matching CPython 3.11 wheel there; other supported platforms retain 0.19.0. The archived geometric benchmarks used macOS Open3D 0.19.0. The Docker demonstration records its own actual version and results; cross-version reconstruction is not claimed bit-identical.

Build natively for your CPU architecture. Verification on this ARM Mac uses Linux ARM64; the AMD64 Open3D wheel crashed under emulation. The runtime-only default image supports the synthetic demo without reconstruction extras; the reconstruction target installs Open3D and plotting/report dependencies.

For real ground-truth generation, download nuScenes separately and mount its root read only:

```bash
docker run --rm \
  -v /absolute/path/to/nuscenes:/data/nuscenes:ro \
  -v "$PWD/artifacts/reproduction:/outputs" \
  occupancy-pipeline:submission python scripts/generate_gt.py \
  --config configs/submission.yaml --nuscenes-root /data/nuscenes \
  --split mini_val --max-samples 3 --output-dir /outputs/targets
```

The selected profile enables both motion correction and Poisson, with box filling off. Poisson proposals remain separate from measured labels and are excluded from current training targets. A finite forward-pass loss shows interface integration; it is separate from trained-checkpoint performance.

## Evidence and interpretation

Read `transcript.json` for command arguments, complete output, exit status and duration of each recorded Docker stage. It also records the installed dependencies and image ID. `video-manifest.json` records video provenance and full decode verification. No verification claim should be inferred solely from the presence of a Dockerfile.

Read `artifacts/accuracy-benchmark-v2/plan.json` and `results.json` for fixed scene partitions, the three-seed architecture comparison and development-only decision fitting. The checkpoint performs GT-derived occupancy completion; it is not a camera or fresh raw-LiDAR prediction model. Poisson's separate two-frame geometric comparison is too small to establish general prediction accuracy.

## Development attribution

[ASSISTANCE.md](ASSISTANCE.md) records development assistance and the origin of the source snapshot. `SNAPSHOT_PROVENANCE.json` retains the original file hashes.
