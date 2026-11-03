# Delivery contents verification

Verified 4 November 2026.

- Source, configuration, tests, Dockerfile, dependency metadata and reproduction scripts are present.
- The design report contains five pages and matches the archived report.
- The walkthrough has 11 chapters, H.264 video at 1280×720 and English captions. It is 5 minutes 51 seconds long, contains **no audio stream**, and passes a full decode check. The repository and experiment-folder copies match.
- The experiment bundle includes all 42 training, 6 development and 16 test target files named by their manifests, plus the selected checkpoint, six training-run checkpoints, model decisions and benchmark results.
- All seven recorded Docker stages succeeded. The regression transcript records 27 passed tests; combined motion/Poisson GT generation, forward/loss, checkpoint evaluation and plots are included.
- The video manifest, captions, walkthrough notes and full command transcripts accompany the silent video. The bundle contains no speech audio files or narrated video.
- The comparison chart is included directly so the accuracy report renders on GitHub.
- Relative README/document links resolve to repository files, apart from the ZIP download link within the archive itself. When the ZIP is extracted, experiment data is already restored.
- The packaging command validates every bundled file against its SHA-256 manifest and checks ZIP integrity. Raw nuScenes scans, credentials, virtual environments and historical cloud setup files are excluded.

These checks verify delivery consistency and the archived execution evidence. They do not represent a new model benchmark or a new Docker run. The documented accuracy limitations and development-assistance disclosure still apply.
