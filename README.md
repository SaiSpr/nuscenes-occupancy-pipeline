# nuScenes Occupancy Pipeline

A modular occupancy ground-truth and learning demonstration for the **End-to-End Perception for Ground Robotics** assignment.

- [Design report](DESIGN_REPORT.pdf): five pages covering architecture, geometry, learning, evidence, tradeoffs and limitations.
- [Walkthrough video](walkthrough.mp4): about 6½ minutes, edited from actual Docker execution with synthesized narration.
- [Complete submission bundle](submission.zip): source snapshot, trained checkpoints, derived grids, results, Docker logs, captions and checksums.
- [Reproduction guide](docs/SUBMISSION.md).

The repository contains the implementation, configuration, tests and recorded validation results. [Implementation details](docs/IMPLEMENTATION_UPDATES.md) describe the changes; [development assistance](docs/ASSISTANCE.md) and [source provenance](SNAPSHOT_PROVENANCE.json) document attribution.

## Selected design

`configs/submission.yaml` enables **motion correction and Poisson reconstruction together**, with box-volume filling disabled.

1. Read real nuScenes scene, scan, calibration, pose and annotation tables.
2. Transform each scan through its own sensor and ego poses into the keyframe ego frame.
3. Match and interpolate tracked object boxes, then align moving-object points. Reposed points contribute hits only, without artificial free-space rays.
4. Generate measured occupied/free/unknown evidence with clipped 3D ray traversal; apply hits last.
5. Reconstruct supported static Poisson surfaces as separately identified proposals in unknown cells. Never overwrite measured evidence or silently include proposals in learning targets.
6. Load a partial occupied grid, run a CNN or U-Net, and compute the appropriate binary or three-class loss and metrics.

The learning task is **GT-derived occupancy mask completion**, not camera prediction or fresh raw-LiDAR inference. The three-class U-Net hides 30% of occupied input cells and predicts unknown, occupied and free classes. No complete ground-truth free-space channel is supplied to the model.

## Restore experiment evidence

Source, configuration and tests are directly browsable. Checkpoints and derived data are kept in the supplied ZIP. From the repository root:

```sh
python3 scripts/unpack_evidence.py
```

This verifies the bundled SHA-256 checksums, then restores only `artifacts/` and `output/`. It does not overwrite repository source. Raw nuScenes files are excluded and must be obtained separately under the dataset's terms.

## Docker: reproducible runtime

Build natively for your CPU architecture:

```sh
docker build --target reconstruction -t occupancy-pipeline:submission .
docker run --rm occupancy-pipeline:submission python -m pytest
```

The reconstruction target includes Open3D, plotting and report dependencies. Linux ARM64 uses Open3D 0.20.0 because no matching 0.19.0 CPython 3.11 wheel was available; other supported platforms retain 0.19.0. The archived macOS geometric comparisons used 0.19.0. Reconstruction across versions is not claimed bit-identical. AMD64 emulation on this ARM Mac failed; native Linux ARM64 was verified.

After restoring evidence, evaluate the supplied trained checkpoint:

```sh
mkdir -p artifacts/reproduction
docker run --rm \
  -v "$PWD/artifacts/accuracy-benchmark-v2:/evidence:ro" \
  -v "$PWD/artifacts/reproduction:/outputs" \
  occupancy-pipeline:submission python scripts/evaluate.py \
  --checkpoint /evidence/recommended-checkpoint.pt \
  --gt-dir /evidence/test --output /outputs/checkpoint-evaluation.json
```

For real GT generation, supply the nuScenes root containing `samples/`, `sweeps/` and `v1.0-mini/`:

```sh
docker run --rm \
  -v /absolute/path/to/nuscenes:/data/nuscenes:ro \
  -v "$PWD/artifacts/reproduction:/outputs" \
  occupancy-pipeline:submission python scripts/generate_gt.py \
  --config configs/submission.yaml --nuscenes-root /data/nuscenes \
  --split mini_val --max-samples 3 --output-dir /outputs/targets
```

The lighter default image supports a clearly labeled synthetic GT demo:

```sh
docker build -t occupancy-pipeline:runtime .
docker run --rm occupancy-pipeline:runtime
```

## Local development

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev,reconstruction,visualization]'
python -m pytest
```

The video renderer is a separate macOS authoring tool requiring the installed Samantha voice, `say`, `afinfo`, Pillow and `imageio-ffmpeg==0.6.0`. It is not needed for the Docker perception runtime. The delivered video includes its synthesized-narration disclosure, captions and complete command transcript.

## Repository modules

| Location | Responsibility |
|---|---|
| `configs/` | Grid, data, GT and learning profiles |
| `src/occ_pipeline/data/` | Raw-table adapter, temporal scans and completion dataset |
| `src/occ_pipeline/geometry/` | Coordinate transformations |
| `src/occ_pipeline/gt_generation/` | Voxel grids, ray evidence, motion correction, Poisson proposals and manifests |
| `src/occ_pipeline/models/` | Tiny CNN and height-preserving U-Net |
| `src/occ_pipeline/training/` | Losses, training, inference decisions and pooled metrics |
| `src/occ_pipeline/utils/` | Storage, atomic writes and deterministic seeds |
| `scripts/` | Generation, inspection, training, evaluation, comparisons and packaging |
| `tests/` | Geometry, evidence isolation, scene separation, masking, models and metrics |
| `docs/` | Implementation details, accuracy protocol and reproduction instructions |

## Stored data contract

The default grid spans x/y `[-40,40)` m and z `[-3,3)` m at 1 m resolution. Saved arrays use `[Z,Y,X] = [6,80,80]`.

| Field | Meaning |
|---|---|
| `occupancy` | `uint8`: 0 unknown, 1 occupied, 2 observed free |
| `point_counts` | Saturated `uint16` point counts |
| `provenance` | Unknown, ray-free, measured-hit or motion-reposed-hit origin |
| `proposal_kind` | Separate box/Poisson proposal identifier |
| `proposed_occupancy` | Measured labels plus inferred proposals; excluded from current training targets |
| `metadata` | Grid, configuration, input hashes, transforms, used/skipped scans and statistics |

Fingerprint filenames and exact manifests prevent stale targets from silently entering a new generation.

## Recorded validation and limitations

Native Linux ARM64 Docker verification passed **all 27 tests**, real-data GT generation with both methods active, forward/loss computation, checkpoint evaluation and prediction visualization. Full commands, exit statuses, dependencies and image identity are in `artifacts/final-walkthrough/transcript.json` after evidence restoration.

The recommended checkpoint reproduced **75.14% occupied IoU** on all 16 archived test frames. The three-seed comparison averaged **74.99%** for the U-Net and **73.63%** for the CNN under the same development-selection protocol. The benchmark used 42 training frames from seven scenes, six development frames from one other scene, and 16 test frames from two official mini validation scenes. Earlier smoke runs inspected some validation frames, so this is not a pristine external test. [Detailed protocol and results](docs/ACCURACY_RESULTS.md).

A separate two-frame geometric comparison measured **67.28% IoU** for motion correction alone and **67.47%** with Poisson: recall increased and precision decreased. This small sparse-reference comparison does not establish general prediction accuracy. Poisson proposals are excluded from current model inputs and supervision, so the model scores must not be attributed to Poisson.

Further limits include coarse 1 m voxels, approximate object-box interpolation, reduced temporal coverage under conservative track filtering, possible reconstructed false surfaces and only two test scenes. Preserving known occupied input also preserves input false positives. The assignment emphasizes design and integration; a trained model is supplementary evidence.

## Implemented deliverables

- Modular nuScenes ingestion, coordinate transforms and temporal occupancy generation.
- Motion correction and separate Poisson surface proposals, enabled in the selected profile.
- CNN and U-Net completion models, loss computation and scene-disjoint evaluation.
- Docker environment with 27 passing regression tests and recorded real-data execution.
- Design report, narrated walkthrough, trained checkpoints and checksummed experiment evidence.

Raw nuScenes data is supplied separately. The design report and accuracy results describe the measured behavior and its limitations.
