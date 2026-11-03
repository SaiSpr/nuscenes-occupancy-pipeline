# Measured occupancy-completion results

This is a small real nuScenes mini **GT mask-completion** experiment. It does not establish camera/raw-LiDAR perception performance or production safety.

## Protocol

- 42 training frames from 7 scenes; 6 development frames from one different scene.
- 16 held-out frames from the two official mini validation scenes.
- Frames selected at evenly spaced timestamps by a fixed rule, before training.
- Three training seeds (42, 43, 44), 30 epochs each; identical scene split, target artifacts, loss weights and validation mask seed (42).
- Both architectures use the same CE objective, Adam learning rate, cosine schedule, augmentation, training data and epoch budget.
- Checkpoints selected by development loss. Occupied decision bias selected on development occupied IoU using the same fixed grid for both models.
- Test was scored after all six training runs and development selection completed.
- Raw, evidence-preserving and development-selected inference are reported separately. Hidden-cell IoU excludes occupied cells already visible in the input.
- Earlier smoke runs inspected a few official validation frames; this is not a pristine external test dataset.
- No Poisson proposals were fed into training or prediction.

## Held-out results

| Model / inference | Occupied IoU mean | SD across seeds | Hidden occupied IoU mean |
|---|---:|---:|---:|
| tiny_cnn/raw | 0.6023 | 0.0607 | 0.2595 |
| tiny_cnn/preserve_input | 0.6028 | 0.0603 | 0.2595 |
| tiny_cnn/development_selected | 0.7363 | 0.0033 | 0.1804 |
| unet/raw | 0.6130 | 0.0106 | 0.2780 |
| unet/preserve_input | 0.6139 | 0.0107 | 0.2780 |
| unet/development_selected | 0.7499 | 0.0014 | 0.2522 |

Copying only visible input occupancy scores 0.6979 occupied IoU; its hidden-cell IoU is zero.

The spread is across three optimizer seeds, not a confidence interval across independent driving environments. Only two test scenes were evaluated.

## Paired model differences

| Seed | Raw U-Net minus raw CNN | Development-selected U-Net minus development-selected CNN |
|---|---:|---:|
| 42 | +0.0002 | +0.0168 |
| 43 | -0.0483 | +0.0096 |
| 44 | +0.0803 | +0.0143 |

Raw architecture comparison: U-Net mean occupied-IoU difference +0.0107.
With the same development-selection procedure applied to both: difference +0.0136.
These differences describe this fixed subset and training budget. They do not prove universal gains or isolate every architectural factor.

## Usable checkpoint

The exported model is **unet, seed 42**, selected using development occupied IoU only. The test score was not used to choose its architecture, seed, checkpoint epoch or decision setting.

```sh
python scripts/evaluate.py --checkpoint artifacts/accuracy-benchmark-v2/recommended-checkpoint.pt --gt-dir artifacts/accuracy-benchmark-v2/test --output artifacts/accuracy-benchmark-v2/recommended-evaluation.json
python scripts/visualize.py --checkpoint artifacts/accuracy-benchmark-v2/recommended-checkpoint.pt --gt-dir artifacts/accuracy-benchmark-v2/test --n-samples 3 --png-dir artifacts/accuracy-benchmark-v2/figures
```

Decision bias is an IoU-oriented classification setting, not calibrated uncertainty. Input preservation assumes input occupied cells are reliable; false positive inputs would also be retained.

## Poisson decision

Six support-distance/density-trim combinations were compared using development scans only. A gate was fixed before evaluation: at least three paired development scans, a positive pooled IoU difference and no negative per-scan difference.
Best development candidate: support=1.0 m, density trim=0.3, paired scans=2, IoU difference=+0.000380.
On 2 eligible held-out scans: annotation-proposal IoU=0.669378, Poisson-proposal IoU=0.671194; recall 0.941137 -> 0.947528, precision 0.698627 -> 0.697105.
**Historical assistant-defined promotion gate: False.** This was an engineering heuristic, not an assignment rule or significance test. The final user-selected submission profile keeps both motion correction and Poisson enabled, with proposals isolated from measured supervision. See the combined validation results and design report.
These are sparse geometric agreement scores, separate from the learned completion-model scores. Poisson cannot improve the model when its proposals are intentionally excluded from model inputs and supervision.

## Evidence and reproducibility

- `artifacts/accuracy-benchmark-v2/plan.json`: predeclared selections, models, seeds and decision grid.
- `artifacts/accuracy-benchmark-v2/results.json`: pooled, per-scene and hidden-cell metrics for every run.
- `artifacts/accuracy-benchmark-v2/*-seed-*/training-summary.json`: training histories and actual scene/file lists.
- `artifacts/accuracy-benchmark-v2/*-seed-*/decision-selection.json`: complete development-only bias search.
- `artifacts/poisson-benchmark/decision.json`: Poisson gate and paired geometric results.
- The initial driver run retained a five-sample loader limit. It was invalidated, excluded from these results, and corrected with exact sample-ID assertions before training.
- CPU macOS execution and native Linux ARM64 Docker execution were verified. Docker passed all 27 tests, ran real combined GT and reproduced the recommended checkpoint score on 16 frames. See `artifacts/final-walkthrough/transcript.json`. Full-dataset benchmarks remain unverified.
- See `IMPLEMENTATION_UPDATES.md` for the inherited pipeline and assistance disclosure.

![Held-out completion comparison](../artifacts/accuracy-benchmark-v2/accuracy-comparison.png)
