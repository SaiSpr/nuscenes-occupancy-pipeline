# Submission guide

Submit a private GitHub repository invitation, the brief design report and the walkthrough video. The ZIP is a transport copy of the source and evidence; it does not itself give evaluators access to a private repository.

## What to share

1. Private repository URL and access for the evaluator's GitHub account.
2. `output/pdf/Occupancy_Pipeline_Design_Report.pdf`: approach, architecture, tradeoffs, results, limitations and attribution.
3. `artifacts/final-walkthrough/walkthrough.mp4`: the final implementation's walkthrough. This is an edited presentation of actual recorded Docker commands and result images, narrated with macOS Samantha synthesized speech. It is not a live screen capture. Captions, narration text and unabridged execution logs accompany it.
4. Optionally, the checksummed submission ZIP as a convenient downloadable copy.

No credentials or raw nuScenes files belong in the repository. Dataset access remains subject to nuScenes terms. A small set of derived target grids is supplied for reproducing the completion demonstration.

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

## Before sending

Confirm that the actual submission repository contains this final source, report and video, and that the evaluator can open it. A locally completed archive does not establish that the remote repository has been updated. The original branch was reconstructed from the supplied Claude walkthrough; see `SNAPSHOT_PROVENANCE.json`. Changes, experiments, report and generated walkthrough are substantially LLM-assisted and require the submitting author's review.
