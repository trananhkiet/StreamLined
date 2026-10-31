# SKD loss-weight comparison

Recorded: 2026-10-18 (Asia/Ho_Chi_Minh).
Results checked: 2026-10-26 (Asia/Ho_Chi_Minh).

## Controlled comparison

| Setting | Baseline A | New run B |
|---|---|---|
| CE weight | 0.1 | 0.5 |
| KD weight | 0.9 | 0.5 |
| SKD/BC weight (shares KD weight) | 0.9 | 0.5 |
| Notebook version | Original notebook v7 | Separate notebook v1 |
| Final status | COMPLETE | COMPLETE |
| Best test top-1 | 68.49% at epoch 235 | **70.26% at epoch 235** |
| Best test top-5 | 91.72% | **92.32%** |
| Best test loss | 1.140646 | **1.073034** |
| Final-epoch test top-1 | 68.16% | **70.04%** |
| Training duration | 13,552.55 s | 13,809.70 s |

Baseline A is reused from the completed experiment; it is not rerun.
Run B starts from scratch using the same seed and initialization procedure.

- Baseline: https://www.kaggle.com/code/trananhkiet/streamlined-kd-resnet56-to-resnet20-on-cifar-100
- Run B: https://www.kaggle.com/code/trananhkiet/streamlined-kd-cifar100-ce05-kd05
- Run B kernel ID: `139386905`, version `1`, private notebook.
- Submitted artifact: [resnet56_resnet20_skd_ce05_kd05.ipynb](resnet56_resnet20_skd_ce05_kd05.ipynb).

## Controls and verification

The variant was generated directly from the server's baseline v7 notebook.
Cell-source comparison confirmed exactly two changed lines: `ce_weight` and
`kd_weight`. All code cells passed Python compilation before upload.

Unchanged: ResNet56 teacher checkpoint and its SHA-256, ResNet20 architecture,
seed 2021, CIFAR-100 data/augmentation, batch size 64, workers 2, 240 epochs,
SGD LR 0.1, momentum 0.9, weight decay 0.0005, LR milestones 150/180/210,
temperature 4, Tikhonov factor 0.1, single T4, and best-checkpoint selection.
Both training and evaluation batch limits remain disabled.

Same Docker image requested and confirmed in server metadata:
`gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f`.
Dataset attachment requested as `fedesoriano/cifar100/versions/1`; REST returned
no invalid sources and server metadata confirmed `fedesoriano/cifar100` attached.

Run B uses the official REST push endpoint to avoid the MCP data-source bug.
MCP initially confirmed RUNNING after submission and later confirmed COMPLETE.
Run B output/session ID is `359636575`. Outputs are isolated in the separate
notebook; baseline v7 is preserved. Run B produced the best-model checkpoint,
epoch-240 checkpoint, complete 240-row history, summary, and teacher checkpoint.

## Run B results

Teacher evaluation was unchanged: top-1 71.59%, top-5 92.51%, test loss
1.049947. The best student checkpoint occurred at epoch 235:

| Metric | Value |
|---|---:|
| Learning rate | 0.0001 |
| Train loss | 0.949018 |
| Train CE component | 0.597931 |
| Train KD component | 0.342646 |
| Train SKD/BC component | 0.008440 |
| Train top-1 | 66.687% |
| Test loss | 1.073034 |
| Test top-1 | **70.26%** |
| Test top-5 | **92.32%** |

The final epoch reached 70.04% top-1 and 92.27% top-5. Total training time was
13,809.70 seconds (3 h 50 min 9.70 s).

Selected milestones:

| Epoch | LR | Train top-1 | Test top-1 | Best so far |
|---:|---:|---:|---:|---:|
| 1 | 0.1 | 9.363% | 20.30% | 20.30% |
| 50 | 0.1 | 41.916% | 44.04% | 47.18% |
| 100 | 0.1 | 42.501% | 41.50% | 47.82% |
| 150 | 0.1 | 42.259% | 38.06% | 47.88% |
| 151 | 0.01 | 52.739% | 62.63% | 62.63% |
| 180 | 0.01 | 58.053% | 63.01% | 64.41% |
| 181 | 0.001 | 61.992% | 67.66% | 67.66% |
| 210 | 0.001 | 65.360% | 69.48% | 69.65% |
| 211 | 0.0001 | 66.243% | 70.05% | 70.05% |
| 235 | 0.0001 | 66.687% | **70.26%** | **70.26%** |
| 240 | 0.0001 | 66.735% | 70.04% | **70.26%** |

## Interpretation

Changing only the loss weights from 0.1/0.9 to 0.5/0.5 improved best top-1 by
**1.77 percentage points**, best top-5 by **0.60 points**, and final-epoch top-1
by **1.88 points**. Run B is 4.47 points above the README's separately reported
65.79% supervised ResNet20 baseline, but that is not a matched baseline run.

The 0.5/0.5 result remains **2.24 points below** the paper/repository's 72.50%
SKD result. It also remains 1.33 points below this run's released teacher
checkpoint (71.59%). The comparison indicates that the `mdistiller` default
weights fit this implementation better than 0.1/0.9, but one seed cannot show
statistical significance or conclusively identify the paper's exact setting.

Both runs evaluate on the CIFAR-100 test set every epoch and select the best
checkpoint on that test set. Therefore, 70.26% is test-selected and should not
be treated as an unbiased final test estimate. A separate validation split and
multiple seeds would be needed for a stronger experimental claim.
