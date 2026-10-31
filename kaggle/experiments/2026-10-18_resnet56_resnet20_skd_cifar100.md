# Experiment log: SKD ResNet56 to ResNet20 on CIFAR-100

## Run identity

| Field | Value |
|---|---|
| Recorded date | 2026-10-18 (Asia/Ho_Chi_Minh) |
| Kaggle notebook | [Streamlined KD: ResNet56 to ResNet20 on CIFAR-100](https://www.kaggle.com/code/trananhkiet/streamlined-kd-resnet56-to-resnet20-on-cifar-100) |
| Successful notebook version | 7 |
| Kaggle kernel ID | `139223435` |
| Kaggle output/session ID | `359410027` |
| Final status | `COMPLETE` |
| Accelerator | NVIDIA Tesla T4 |
| Framework | PyTorch `2.11.0+cu128`, torchvision `0.26.0+cu128` |
| Training duration | 13,552.55 seconds (3 h 45 min 52.55 s) |

## Experiment objective

Reproduce the first homogeneous-architecture result in the repository README:

- teacher: ResNet56;
- student: ResNet20;
- dataset: CIFAR-100;
- distiller: Streamlined Knowledge Distillation (SKD);
- reported teacher top-1: 72.40%;
- reported supervised student baseline top-1: 65.79%;
- reported distilled student top-1 (`Ours`): 72.50%.

The supervised student baseline is a separate reference experiment. Distillation
does **not** load the 65.79% ResNet20 checkpoint and fine-tune it. It initializes a
new ResNet20 and trains it from scratch using labels plus signals from the frozen
teacher.

## Data and checkpoints

### CIFAR-100

- Kaggle dataset: `fedesoriano/cifar100`, version 1.
- Server-side source attachment was confirmed in notebook metadata before the
  successful run.
- Input path used by the run: `/kaggle/input/cifar100`.
- Files: `train` (50,000 images), `test` (10,000 images), and `meta`.
- `torchvision.datasets.CIFAR100` verified the official file MD5 checksums.
- No external CIFAR download was permitted in the successful notebook
  (`download=False`).

The training dataset concatenates two views of the complete training set:

1. weak augmentation: random crop with padding 4 and random horizontal flip;
2. strong augmentation: the weak pipeline plus the repository's two-operation
   RandAugment and Cutout.

This produces 100,000 training samples per epoch. Test images use only tensor
conversion and normalization. All views use CIFAR-100 normalization:

```text
mean = (0.5071, 0.4867, 0.4408)
std  = (0.2675, 0.2565, 0.2761)
```

### Teacher checkpoint

- File: `best_model_weights_resnet56.pth`.
- SHA-256: `2c8c5778c296ebf2bc3b226b5a2e5d4c93345c38676e48bb6e865206cc31d954`.
- The files under `CIFAR-100/Model_Pth` and `CIFAR-100/SDD_Models_Pth` have the
  same SHA-256.
- The teacher was frozen and used only for inference during distillation.
- Re-evaluated checkpoint result: 71.59% top-1 and 92.51% top-5.

The README reports 72.40% for the teacher, but the released checkpoint used here
evaluates to 71.59% under the canonical test set, repository normalization, and
notebook model implementation. The repository does not include enough original
run metadata to determine whether the table came from another checkpoint/run or
from a different evaluation environment.

## Training configuration

| Parameter | Value |
|---|---:|
| Seed | 2021 |
| Epochs | 240 |
| Batch size | 64 |
| Training samples per epoch | 100,000 |
| Batches per epoch | 1,563 |
| Approximate optimizer steps | 375,120 |
| DataLoader workers | 2 |
| Optimizer | SGD |
| Initial learning rate | 0.1 |
| Momentum | 0.9 |
| Weight decay | 0.0005 |
| KD temperature | 4.0 |
| Tikhonov factor | 0.1 |
| Cross-entropy weight | 0.1 |
| KD weight | 0.9 |
| SKD/batch-correlation weight | 0.9 |
| Train batch limit | Disabled (`0`) |
| Evaluation batch limit | Disabled (`0`) |

Learning-rate schedule:

| Epochs | Learning rate |
|---|---:|
| 1–150 | 0.1 |
| 151–180 | 0.01 |
| 181–210 | 0.001 |
| 211–240 | 0.0001 |

Determinism settings included Python, NumPy, PyTorch, CUDA and DataLoader seeds,
with cuDNN deterministic mode enabled and benchmark mode disabled.

## Results

### Final summary

| Metric | Result |
|---|---:|
| Teacher test loss | 1.049947 |
| Teacher top-1 | 71.59% |
| Teacher top-5 | 92.51% |
| Best student epoch | **235** |
| Best student test loss | 1.140646 |
| Best student top-1 | **68.49%** |
| Best student top-5 | 91.72% |
| Final epoch student top-1 | 68.16% |
| Final epoch student top-5 | 91.72% |

At the best epoch (235):

| Metric | Value |
|---|---:|
| Learning rate | 0.0001 |
| Train loss | 0.684753 |
| Train CE component | 0.133179 |
| Train KD component | 0.537389 |
| Train SKD/BC component | 0.014185 |
| Train top-1 | 63.046% |
| Test loss | 1.140646 |
| Test top-1 | **68.49%** |
| Test top-5 | 91.72% |

### Selected training milestones

| Epoch | LR | Train top-1 | Test top-1 | Best so far |
|---:|---:|---:|---:|---:|
| 1 | 0.1 | 10.681% | 16.43% | 16.43% |
| 50 | 0.1 | 43.603% | 46.79% | 49.16% |
| 100 | 0.1 | 44.317% | 41.31% | 49.86% |
| 133 | 0.1 | 44.156% | 50.93% | 50.93% |
| 150 | 0.1 | 44.323% | 45.16% | 50.98% |
| 151 | 0.01 | 53.080% | 63.13% | 63.13% |
| 180 | 0.01 | 57.340% | 63.24% | 64.38% |
| 181 | 0.001 | 60.228% | 66.66% | 66.66% |
| 210 | 0.001 | 62.316% | 67.80% | 67.98% |
| 211 | 0.0001 | 62.705% | 67.92% | 67.98% |
| 216 | 0.0001 | 63.005% | 68.28% | 68.28% |
| 235 | 0.0001 | 63.046% | **68.49%** | **68.49%** |
| 240 | 0.0001 | 63.008% | 68.16% | **68.49%** |

The large accuracy jumps at epochs 151 and 181 coincide with the first two
learning-rate reductions. Training completed normally; there was no runtime
failure or truncated epoch.

## Comparison with repository report

| Quantity | README | This run | Difference |
|---|---:|---:|---:|
| Teacher ResNet56 top-1 | 72.40% | 71.59% | -0.81 pp |
| Student supervised baseline top-1 | 65.79% | Not rerun | N/A |
| Distilled ResNet20 top-1 | 72.50% | 68.49% | **-4.01 pp** |

Relative to the README's separately reported student baseline, this run is
2.70 percentage points higher. That is not a strictly controlled improvement
measurement because the 65.79% baseline was not rerun with this notebook's exact
seed, environment, initialization, and data-loader behavior.

The run therefore verifies that the pipeline trains and improves a small model,
but it does **not** reproduce the reported 72.50% result.

## Reproduction caveats and follow-up

1. The released teacher checkpoint itself is 0.81 percentage points below the
   teacher number shown in the README.
2. The original script constructs every model in its student/teacher selection
   lists before selecting one. Those eager constructions consume RNG state. The
   notebook seeds and directly constructs ResNet20, so its initial weights are
   deterministic but may not match the original script's exact RNG position.
3. The notebook uses a globally tracked best validation checkpoint. The original
   trainer resets `best_acc` inside every epoch, effectively retaining each
   current epoch rather than a true global best.
4. The notebook uses one T4 rather than the original `DataParallel` wrapper.
5. A controlled supervised ResNet20 run with the exact same environment and seed
   is required for a fair baseline comparison.

Recommended next checks, in order:

1. reproduce the original eager model-construction order before initializing the
   selected ResNet20;
2. run the matched supervised ResNet20 baseline;
3. evaluate whether the paper used another teacher checkpoint or reported an
   average/best-of-seeds result;
4. run multiple seeds and report mean plus standard deviation.

## Kaggle execution history

| Version | Outcome |
|---:|---|
| 1 | Began slow external CIFAR-100 download; cancelled. |
| 2 | Kaggle MCP silently dropped the dataset attachment; external download began; cancelled. |
| 3 | Attachment diagnostic/fail-fast attempt; cancelled. |
| 4–5 | Quick-save diagnostics for Kaggle MCP data-source fields; not used for training. |
| 6 | Failed because KaggleHub cannot attach a new dataset from a non-interactive batch session. |
| 7 | Dataset attached through the official REST push endpoint; completed successfully. |

## Saved artifacts

The successful Kaggle version contains:

- `streamlined_kd/resnet20_skd_best.pth` — global-best student at epoch 235;
- `streamlined_kd/last_checkpoint.pt` — epoch 240 model, optimizer, and history;
- `streamlined_kd/history.csv` — all 240 epoch metrics;
- `streamlined_kd/summary.json` — configuration and final summary;
- `streamlined_kd/best_model_weights_resnet56.pth` — downloaded, checksum-verified teacher.
