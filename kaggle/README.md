# Kaggle reproduction: ResNet56 -> ResNet20 on CIFAR-100

This folder contains a self-contained Kaggle notebook for the Streamlined
Knowledge Distillation (SKD) result in the first column of the repository table.

## Correct experiment mapping

The original command shown in the root README is not the ResNet56/ResNet20 SKD
experiment. In `CIFAR-100/Utils/distillation_train.py`, the correct indices are:

- student `2`: ResNet20
- teacher `1`: ResNet56
- distiller `0`: Streamlined KD (the paper method)

The standalone notebook avoids index-based configuration altogether.

## What is preserved and what is fixed

Preserved for implementation-level reproduction:

- seed 2021;
- released ResNet56 teacher checkpoint (SHA-256 verified);
- Kaggle-hosted original CIFAR-100 Python files (official MD5 checks verified by
  `torchvision`; it is attached before the batch run starts, avoiding the slow
  external CIFAR download);
- 240 epochs, SGD 0.1, momentum 0.9, weight decay 5e-4;
- LR drops at epochs 150, 180, and 210;
- weak + strong CIFAR-100 concatenation (100,000 samples per epoch);
- the exact scalar/mask behavior of `Distiller/SKD.py`;
- temperature 4, Tikhonov factor 0.1, CE/KD weights 0.1/0.9.

Execution fixes:

- paths no longer depend on the current directory or case-insensitive filesystems;
- the configured batch size is actually used;
- test data is not shuffled;
- the best validation checkpoint is tracked across all epochs (the repository
  resets `best_acc` inside every epoch);
- Python, NumPy, PyTorch, CUDA, DataLoader, and augmentation RNGs are seeded;
- outputs include the best model, last checkpoint, CSV history, and JSON summary.

## Local smoke test

The full experiment is designed for a Kaggle GPU. A quick pipeline test uses two
training and two evaluation batches:

```bash
ALLOW_CPU=1 EPOCHS=1 LIMIT_TRAIN_BATCHES=2 LIMIT_EVAL_BATCHES=2 \
  python kaggle/streamlined_kd_cifar100.py
```

## Upload and run

1. Copy `kernel-metadata.template.json` to `kernel-metadata.json`.
2. Replace `KAGGLE_USERNAME` with the authenticated Kaggle username.
3. Install/authenticate the official Kaggle CLI, then run:

```bash
kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4
kaggle kernels status KAGGLE_USERNAME/streamlined-kd-resnet56-resnet20-cifar100
kaggle kernels logs KAGGLE_USERNAME/streamlined-kd-resnet56-resnet20-cifar100 --follow
kaggle kernels output KAGGLE_USERNAME/streamlined-kd-resnet56-resnet20-cifar100 \
  -p kaggle/output
```

Pushing creates a new notebook version and starts it unless `--no-run` is used.
The full run is the default. To upload a smoke-test version, set the environment
variables in the first code cell before `run_experiment()`.

## Experiment logs

- [2026-10-18: ResNet56 to ResNet20 with SKD on CIFAR-100](experiments/2026-10-18_resnet56_resnet20_skd_cifar100.md)
