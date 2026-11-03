# Implemented updates to the supplied three-class branch

## Source and scope

This folder was reconstructed from all 39 full-file code blocks in the user's
`occupancy-branch-explained.md`. The source identifies
`gh-sprakash/occupancy-perception-pipeline`, branch
`worktree-i-would-like-sue-pure-boole`, last described commit `0ae1e43`.
`SNAPSHOT_PROVENANCE.json` records the source document's SHA256 and every original
file hash. This is a local reconstructed source snapshot, not a live Git checkout.
GitHub's connected API returned 404, so no current branch state, merge or remote
upload is claimed. The document's statement about branch divergence is historical.

The implementation preserves genuine three-class prediction. It does not inherit
the older binary model's practice of copying free labels from GT into predictions.
New features were implemented locally; they are not represented as a merge of the
unavailable current `main` branch. The source document's embedded Claude skills and
SageMaker instructions were treated as reference data, not executed.

## Changes

### Reproducible data and honest targets

`OccupancyDataset` now creates a stable per-sample validation mask from a seed and
filename hash. Access order, repeated calls and DataLoader batch size do not change
it. Training masks change at explicitly assigned epochs. Optional training-only XY
flips are applied before constructing both inputs and targets. The single input
channel remains a masked occupied grid; no full-GT free channel is supplied.

This remains **GT mask completion**, not prediction from cameras or raw LiDAR.
A high completion score would not establish raw-sensor perception quality.

Binary loss and metrics operate only on observed occupied/free cells, excluding
unknown. The three-class model explicitly predicts unknown, occupied and free using
cross-entropy. Its unknown class represents observation status, not proof of empty
space. Class weights are computed once from the training partition and reused for
validation. They no longer fluctuate with each validation batch.

### Scene separation and metrics

The trainer groups samples by scene and keeps complete scenes together. Explicit
`--val-dir` supports separate official mini_train/mini_val targets and rejects scene
overlap. Real targets without scene metadata fail clearly. Independent synthetic
samples receive synthetic groups. A multi-frame, single-scene real dataset needs
external validation or an explicit `val_split: 0`.

Validation uses a separate non-augmented dataset, every validation sample, and one
accumulated confusion matrix. Per-class IoU and mIoU are calculated after accumulation,
not averaged per batch. Absent classes receive null and are excluded from mIoU.
Binary TP/FP/FN are similarly pooled. Validation loss uses the appropriate global
weight denominator. Reports/checkpoints record scenes, files, seed and fixed weights.

`scripts/evaluate.py` loads state dictionaries with `weights_only=True`, requires a
recorded protocol, rejects training-scene overlap, and evaluates the entire supplied
directory/manifest. Metrics derive from model logits, including free and unknown.

### Model and training

A small `OccupancyUNet3D` adds three XY pooling levels while retaining height. Decoder
features are resized to skip sizes, so odd XY dimensions are supported (minimum 8).
Optional auxiliary heads use downsampled targets; the trainer, demo and visualization
unwrap their outputs consistently. The existing tiny CNN remains available.

Binary BCE/focal and three-class CE are supported. Optional cosine/step LR schedules
are configurable. `train.py` now accepts `--epochs` and `--val-dir`. Visualization
uses model configuration and mask seed saved in the checkpoint, supports auxiliary
outputs, and prioritizes occupied over free in height projections. A raw numeric
max over 0/1/2 would incorrectly hide occupied cells behind label 2.

### Temporal GT

The raw nuScenes table reader follows past-only LiDAR chains, retaining each sweep's
actual sensor origin. It validates time order, sensor identity, scenes, binary records
and cycles. Points are transformed through their own ego pose to keyframe ego space.
Optional body filtering is applied in each sweep's own ego coordinates.

Tracked annotation boxes are interpolated with linear translation/size and shortest-
path quaternion SLERP. Both translation and rotation can trigger reposing. Reposed
points provide occupied hits only; invented rays to their new positions are never
carved. Unbracketed intervals or unmatched instance sets are conservatively skipped
and reported. A nominal ten-sweep run can therefore contain only one usable sweep.
Missing sensor files fail explicitly instead of silently reducing dataset coverage.

Batched clipped DDA replaces the large rays-by-steps sampling allocation. It advances
all tied axes at corners and processes bounded chunks. Across sweeps, hits are applied
last so a later free ray cannot erase them. Counts are saturated before uint16 storage
to avoid overflow. A configurable minimum hit threshold can suppress sparse noise,
with the usual risk of losing thin/far objects.

`gt_generation/evidence` contains the self-contained temporal core ported from the
previously validated local implementation. The branch-facing wrapper converts its
XYZ / -1,0,1 convention to this branch's **ZYX / 0,1,2** convention explicitly.

### Poisson and annotation proposals

Poisson remains optional and disabled in `temporal_3class.yaml`. In-range points
outside annotated boxes form the static reconstruction input. Locally estimated
normals are flipped toward acquisition origins. Input is deterministically capped;
Open3D uses one reconstruction thread. Low-density mesh vertices are trimmed, the
mesh is cropped to the grid, and surface voxels require nearby point support.
Unannotated moving objects and normal errors remain possible.

Reconstruction adds a surface shell proposal, not verified solid occupancy or new
free space. Annotation filling uses voxel centers inside boxes, an approximate shape
prior. Both are restricted to unknown cells and stored separately as:

- `proposal_kind`: 0 none, 4 box prior, 5 Poisson shell.
- `proposed_occupancy`: original labels plus the inferred occupied proposals.
- `provenance`: 0 unknown, 1 ray free, 2 measured hit, 3 reposed hit.

The dataset only trains on `occupancy`; inferred proposals never silently enter
supervision. Their storage consistency is checked on load. Source sweep hashes,
transforms, annotation hash, actual sweep counts and skipped tokens are recorded.
GT filenames include a generation fingerprint and `manifest.json` lists exactly the
current generation's targets, excluding stale NPZ files. Writes are atomic.

Synthetic generation now requires `--synthetic`. Missing real data never silently
turns a real-data command into a synthetic experiment.

## Commands

Use a new virtual environment with Python 3.11/3.12 on a platform supported by the
PyTorch/Open3D wheels. Torch is now pinned to 2.8.0. Direct reconstruction dependency
is pinned to Open3D 0.19.0; all transitive dependencies are not fully locked.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,reconstruction,visualization]'
.venv/bin/python -m pytest

# Generate official train and validation targets separately.
.venv/bin/python scripts/generate_gt.py --config configs/temporal_3class.yaml \
  --nuscenes-root /path/to/nuscenes --split mini_train --max-samples 24 \
  --output-dir artifacts/temporal/train
.venv/bin/python scripts/generate_gt.py --config configs/temporal_3class.yaml \
  --nuscenes-root /path/to/nuscenes --split mini_val --max-samples 24 \
  --output-dir artifacts/temporal/val

# Train or simply demonstrate forward/loss if training is unnecessary.
.venv/bin/python scripts/train.py --config configs/temporal_3class.yaml \
  --gt-dir artifacts/temporal/train --val-dir artifacts/temporal/val --epochs 10
.venv/bin/python scripts/evaluate.py --checkpoint artifacts/checkpoints/checkpoint_best.pt \
  --gt-dir artifacts/temporal/val --output artifacts/evaluation.json
.venv/bin/python scripts/visualize.py --checkpoint artifacts/checkpoints/checkpoint_best.pt \
  --gt-dir artifacts/temporal/val --n-samples 3 --png-dir artifacts/figures

# Optional Poisson: use poisson_3class.yaml for generation, and compare proposals.
.venv/bin/python scripts/compare_gt.py --config configs/poisson_3class.yaml \
  --nuscenes-root /path/to/nuscenes --max-samples 24 --include-poisson \
  --output artifacts/heldout-comparison.json

# Original CPU baseline image; optional Open3D image on a supported wheel platform.
docker build --target runtime -t occupancy-pipeline:updated .
docker build --platform linux/amd64 --target reconstruction -t occupancy-pipeline:poisson .
```

`compare_gt.py` withholds each current LiDAR scan and builds variants only from older
scans, allowing target-time annotation priors. Use `paired_aggregate` to compare the
same samples. Reference unknown cells are not scored; abstention on a known hit is a
miss. Reports include wrongly-free cells and coverage. These scores measure sparse
scan agreement, not complete-scene accuracy or model prediction quality.

## Validation and limitations

The 13 recovered tests and 10 new regressions passed (23 total) on macOS ARM,
Python 3.12 / PyTorch 2.8.0 / Open3D 0.19.0. Coverage includes reproducible masks,
scene grouping, proposal isolation, unknown-cell gradients, genuine three-class
confusion matrices, fixed CE weights, stale-target exclusion, count saturation,
auxiliary U-Net output shapes, actual Poisson reconstruction and three-class training.
Checkpoint evaluation was repeated with batch sizes 1 and 2, yielding matching metrics.

A deliberately chosen four-frame real nuScenes integration exercise generated GT
with actual Poisson reconstruction. Matched-track cases used ten sweeps; uncertain
cases reported skipped sweeps. A base-width-2, three-class U-Net with auxiliary heads
ran one epoch on two training frames and evaluated two scene-disjoint validation
frames. This is a software smoke run, not a trained accuracy benchmark. In particular,
occupied IoU was 0 in this one-epoch run; no model accuracy improvement is claimed.

Evidence is in `artifacts/update-validation`: selection, targets/manifests, training
summary, checkpoint, evaluation, held-out comparison, test transcript and images.
The paired GT comparison contains only one eligible validation scan. At 1 m resolution:

| GT candidate | Occupied IoU | Precision | Recall | Wrongly free |
|---|---:|---:|---:|---:|
| One past scan | 0.8458 | 0.9297 | 0.9036 | 123 |
| Multiple past scans | 0.7451 | 0.7651 | 0.9661 | 51 |
| Motion corrected | 0.7496 | 0.7682 | 0.9688 | 44 |
| + box proposals | 0.7438 | 0.7621 | 0.9688 | 44 |
| + Poisson proposals | 0.7418 | 0.7587 | 0.9709 | 44 |

Motion correction improved this case relative to naive accumulation. Poisson increased
recall while adding false positives and reducing IoU; it remains experimental.

At the initial smoke-test stage Docker was unavailable. Final verification on 3 November 2026 supersedes that limitation: the native Linux ARM64 reconstruction image built, all 27 tests passed, real combined GT generation and forward/loss ran, and the trained checkpoint evaluation reproduced occupied IoU 0.751392 on all 16 archived test frames. Commands, full logs, dependency versions and image identity are in `artifacts/final-walkthrough/transcript.json`. Linux ARM64 uses Open3D 0.20.0; the archived macOS geometric benchmarks used 0.19.0. The AMD64 emulation attempt failed and is not the verified path.
No GitHub changes, SageMaker commands, full-dataset training, branch merge or remote
publication was performed. The recovered infrastructure/session documents are historical
and are not included in the clean update archive. Original README/EXPLAINER content
is historical; this document is the authoritative description of the local updates.

## Attribution

Codex reconstructed the supplied snapshot and implemented/tested the updates. Claude's
original branch source remains attributed through the source-document provenance.
Reference implementations/docs:

- https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py
- https://www.open3d.org/docs/release/tutorial/geometry/surface_reconstruction.html
- https://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version5.7/

## Accuracy extension

A subsequent controlled experiment adds fixed masks across training seeds,
development-only occupied-score decision fitting, reliable-input preservation,
whole-test and hidden-cell metrics, and a Poisson promotion gate. See
[ACCURACY_RESULTS.md](ACCURACY_RESULTS.md) for measured results. The initial one-epoch
figures above remain historical smoke-test evidence; they are not the trained
benchmark. The latest regression transcript is in the accuracy benchmark folder.
