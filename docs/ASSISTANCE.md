# Development assistance and attribution

LLM assistance was used for the source implementation, subsequent changes, experiment automation and documentation.

- **Claude:** the initial branch implementation was supplied in a complete-code walkthrough. Its source-document hash and original file hashes are retained in `SNAPSHOT_PROVENANCE.json`.
- **Codex:** reconstructed that snapshot; implemented temporal motion correction, Poisson proposal isolation, data/metric corrections and the U-Net/evaluation extensions; automated the recorded experiments, tests, Docker validation, report, walkthrough generation and repository publication.
- **Walkthrough narration:** generated with the installed macOS Samantha synthetic voice. The video is an edited presentation of actual command logs and result images, not a live screen recording.

The recorded test and experiment results describe executions performed in the documented environments.

The project uses PyTorch and Open3D. Dataset conventions follow the nuScenes schema; references are listed in the design report and implementation notes.
