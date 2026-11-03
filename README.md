# nuScenes Occupancy Pipeline

Private submission repository: **SaiSpr/nuscenes-occupancy-pipeline**.

- [Design report](DESIGN_REPORT.pdf): five-page architecture, evidence, tradeoffs and limitations.
- [Walkthrough video](walkthrough.mp4): about 6½ minutes, edited from actual Docker commands with synthesized narration.
- [Complete submission bundle](submission.zip): source snapshot, checkpoint, derived grids, experiment results, full Docker logs, captions and checksums.
- [Submission and reproduction guide](docs/SUBMISSION.md).

Source, configuration and tests are available directly in this repository. To restore the bundled experiment artifacts at the paths used by the commands below, run:

```sh
python3 scripts/unpack_evidence.py
```

This verifies every archived file against its bundled SHA-256 digest, then restores only `artifacts/` and `output/`; it does not replace repository source files. Raw nuScenes data is not included. Native Linux ARM64 Docker verification passed all **27 tests**, real combined GT generation, forward/loss and checkpoint evaluation. The checkpoint reproduced **75.14% occupied IoU** on 16 archived test frames. This is GT-derived occupancy completion, not raw sensor prediction; Poisson proposals are excluded from learned supervision.

The project is substantially LLM-assisted. Review the provenance, assistance disclosure and limitations in the report before submission. Private repository access for the evaluator must be arranged separately.

---

## Selected submission profile: both methods enabled

Use `configs/submission.yaml` to run **motion correction and Poisson reconstruction together**.
Box-volume filling is disabled in this profile to isolate the two methods. Poisson
surface proposals remain identifiable and separate from measured learning targets.
This is the selected design configuration; earlier optional/default-promotion
wording describes historical experiments and does not disable this explicit profile.

The [five-page design report](output/pdf/Occupancy_Pipeline_Design_Report.pdf) explains
the architecture, GT logic, code modules, learning task, experiments, tradeoffs and
assistance disclosure. A direct no-box-fill comparison measured geometric IoU
67.28% for motion correction and 67.47% for motion plus Poisson on two eligible
held-out scans. These are geometric agreement scores, separate from the model's
74.99% mean completion IoU.

```sh
python scripts/generate_gt.py --config configs/submission.yaml \
  --nuscenes-root /path/to/nuscenes --split mini_val \
  --max-samples 24 --output-dir artifacts/submission/val
```

> **November 2026 local update:** This source snapshot was restored from the supplied
> branch walkthrough and extended with scene-disjoint evaluation, a U-Net, temporal
> GT and optional Poisson proposals. Read [the update report](docs/IMPLEMENTATION_UPDATES.md)
> for current commands, validation evidence and limitations. Original instructions
> below describe the older branch; GT filenames now include fingerprints and a manifest.

# Occupancy Perception Pipeline

CPU-first starter project for generating occupancy ground truth (GT) from nuScenes-style LiDAR data and validating a training-ready deep learning pipeline.

The goal is project design, not model quality. The repository demonstrates:

- Occupancy GT generation from point clouds.
- A nuScenes-mini adapter for real data.
- A deterministic synthetic mode for laptop and Docker demos.
- A PyTorch `Dataset`, tiny 3D CNN baseline, loss, and metrics.
- Tests, Docker, and a design report.

## Design Scope

Initial GT is binary LiDAR occupancy:

```text
1 = occupied voxel observed by LiDAR
0 = unknown / not observed
```

The code intentionally does not claim that `0` means definitely free. Sparse LiDAR only proves observed surfaces unless ray casting or multi-frame free-space reasoning is added.

Default laptop grid:

```text
x: [-40m, 40m)
y: [-40m, 40m)
z: [-3m, 3m)
voxel_size: 1.0m
shape: [Z, Y, X] = [6, 80, 80]
```

This is small enough to run comfortably on CPU and still exercises the full pipeline.

## Quick Start: Local Laptop Demo

From this project directory:

```bash
python scripts/generate_gt.py --config configs/default.yaml --synthetic --max-samples 3
python scripts/run_pipeline.py --config configs/default.yaml
python scripts/inspect_gt.py artifacts/occupancy_gt/synthetic_0000.npz
pytest
```

Expected flow:

```text
synthetic point cloud
-> voxel occupancy GT .npz files
-> OccupancyDataset
-> tiny 3D CNN forward pass
-> BCE loss + IoU/precision/recall metrics
```

## Real nuScenes-mini Run

Install optional dependencies:

```bash
pip install -e ".[nuscenes,dev]"
```

Then run with a mounted or local nuScenes root:

```bash
python scripts/generate_gt.py \
  --config configs/default.yaml \
  --nuscenes-root /path/to/nuscenes \
  --version v1.0-mini \
  --split mini_train \
  --max-samples 5

python scripts/run_pipeline.py --config configs/default.yaml
```

The nuScenes root should contain folders such as `samples/`, `sweeps/`, `maps/`, and `v1.0-mini/`.

## Docker

For the final motion-correction + Poisson profile, complete submission commands,
walkthrough provenance and evaluator handoff, see [Submission guide](docs/SUBMISSION.md).
Use the `reconstruction` target; the default image below is the lighter synthetic runtime.

Build:

```bash
docker build -t occupancy-perception .
```

Synthetic end-to-end demo:

```bash
docker run --rm \
  -v "$PWD/artifacts:/app/artifacts" \
  occupancy-perception \
  python scripts/generate_gt.py --synthetic --max-samples 3 --output-dir artifacts/occupancy_gt

docker run --rm \
  -v "$PWD/artifacts:/app/artifacts" \
  occupancy-perception \
  python scripts/run_pipeline.py --gt-dir artifacts/occupancy_gt
```

Real nuScenes-mini demo:

```bash
docker run --rm \
  -v /path/to/nuscenes:/data/nuscenes:ro \
  -v "$PWD/artifacts:/app/artifacts" \
  occupancy-perception \
  python scripts/generate_gt.py --nuscenes-root /data/nuscenes --max-samples 5
```

## Repository Layout

```text
configs/default.yaml
scripts/generate_gt.py
scripts/run_pipeline.py
scripts/inspect_gt.py
src/occ_pipeline/
  data/
  geometry/
  gt_generation/
  models/
    occupancy_cnn.py
  training/
  utils/
  visualization/
tests/
docs/design_report.md
Dockerfile
```

## Generated Artifact Format

Each `.npz` file contains:

```text
occupancy: uint8 [Z, Y, X]
point_counts: uint16 [Z, Y, X]
metadata: JSON string
```

Metadata includes grid range, voxel size, point counts, occupancy ratio, sample token, source, and runtime.

## Metrics

GT generation metrics:

- input point count
- finite point count
- in-range and out-of-range point count
- occupied voxel count
- occupancy ratio
- runtime per sample

Pipeline metrics:

- BCE loss
- IoU
- precision
- recall
- predicted occupancy ratio
- target occupancy ratio

These metrics validate the data and pipeline contract. They are not presented as model-performance claims.

## CNN Baseline

The configured model is `tiny_3d_cnn`, a small `Conv3d -> ReLU -> Conv3d -> ReLU -> Conv3d` network.

It is included to prove the deep learning path is real:

```text
occupancy tensor [B, 1, Z, Y, X]
-> TinyOccupancyCNN
-> logits [B, 1, Z, Y, X]
-> BCE loss and occupancy metrics
```

It is intentionally small for laptop/4GB-GPU demos and is not presented as a trained model.

## Edge Cases Handled

- Points with NaN or infinite coordinates are ignored.
- Points outside the configured voxel range are counted and skipped.
- Max-boundary points are excluded with half-open ranges `[min, max)`.
- Empty point clouds produce valid all-zero occupancy grids.
- Class imbalance is handled in the demo loss with capped positive weighting.
- CUDA is optional; CPU is the default for reproducibility.

## LLM Disclosure

This starter implementation was scaffolded with LLM assistance. The candidate should review, understand, and be prepared to defend every design choice before submission.


### Measuring and improving completion accuracy

The accuracy extension adds a controlled baseline/U-Net comparison, fixed evaluation
masks across optimizer seeds, development-only decision fitting, and an input-evidence
preservation rule. See [measured results](docs/ACCURACY_RESULTS.md) for the actual
numbers and their limits. This evaluates **GT mask completion**, not raw-sensor or
camera prediction. Poisson geometry is evaluated separately and stays optional
unless its development evidence gate passes.

```sh
# Three seeds, fixed scene-disjoint train/development/test selections.
python scripts/benchmark_accuracy.py --data-root /path/to/nuscenes \
  --output artifacts/accuracy-benchmark-v2 --epochs 30 --seeds 42,43,44
python scripts/benchmark_poisson.py --data-root /path/to/nuscenes \
  --plan artifacts/accuracy-benchmark-v2/plan.json --output artifacts/poisson-benchmark
python scripts/report_accuracy.py

# Fit a decision setting for another trained checkpoint using its recorded dev set.
# The script rejects different scenes/samples and preserves the original checkpoint.
python scripts/tune_decision.py --checkpoint artifacts/checkpoints/checkpoint_best.pt \
  --dev-dir artifacts/development --output artifacts/checkpoints/checkpoint_decision.pt
```

The resulting decision checkpoint is consumed automatically by `evaluate.py` and
`visualize.py`. The occupied-score offset optimizes development IoU; it does not
provide calibrated uncertainty. Preserving input occupancy assumes those input
measurements are reliable, and can retain false positives from noisy sensors.
