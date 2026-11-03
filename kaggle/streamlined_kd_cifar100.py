# %% [markdown]
# # Streamlined KD: ResNet56 -> ResNet20 on CIFAR-100
#
# This notebook-ready script reproduces the homogeneous-architecture experiment
# reported in the repository README. It intentionally keeps the paper code's
# two-view training set and SKD loss semantics, while fixing execution issues
# such as path handling, global-best checkpointing, and deterministic seeding.

# %%
from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import random
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm


@dataclass(frozen=True)
class Config:
    seed: int = 2021
    epochs: int = int(os.getenv("EPOCHS", "240"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "64"))
    num_workers: int = int(os.getenv("NUM_WORKERS", "2"))
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    lr_milestones: tuple[int, ...] = (150, 180, 210)
    temperature: float = float(os.getenv("TEMPERATURE", "4.0"))
    tik: float = 1e-1
    ce_weight: float = float(os.getenv("CE_WEIGHT", "0.5"))
    kd_weight: float = float(os.getenv("KD_WEIGHT", "0.5"))
    # "scalar": repo/run-B behavior, mean(KL) * mask  (mask has no real selection effect)
    # "per_sample": experiment A1, mean(mask_i * KL_i)
    kd_mask_mode: str = os.getenv("KD_MASK_MODE", "scalar")
    # Which samples the confidence mask keeps: "low" (repo/A1: teacher confidence <= median) or
    # "high" (experiment A2: teacher confidence >= median). Only changes the KD mask, not the BC mask.
    kd_mask_select: str = os.getenv("KD_MASK_SELECT", "low")
    # Weight of the SKD/BC (Gram-matrix) term. Negative (default) = share kd_weight, i.e. runs A/B/A1/A2 behavior.
    # Experiment A3 sweeps this independently of kd_weight (0 / 0.5 / 2.0).
    lambda_bc: float = float(os.getenv("LAMBDA_BC", "-1"))
    # Plan item "B" (Logit Standardization, CVPR 2024): z_hat = (z - mean(z)) / (std(z) + 1e-7) per sample, applied to
    # BOTH student and teacher logits inside the KL branch only (CE, Gram/BC and the confidence masks keep raw logits).
    logit_std: bool = os.getenv("LOGIT_STD", "0") == "1"
    # Plan item "C" (DKD, Zhao et al. CVPR 2022): replaces the KL + BC terms by
    #   L = dkd_ce_weight * CE + min(epoch / dkd_warmup, 1) * (dkd_alpha * TCKD + dkd_beta * NCKD)
    # following megvii-research/mdistiller (configs/cifar100/dkd/res56_res20.yaml: CE 1.0, alpha 1, beta 2, T 4, warmup 20).
    # "skd" (default) keeps every previous experiment unchanged.
    loss_mode: str = os.getenv("LOSS_MODE", "skd")
    dkd_alpha: float = float(os.getenv("DKD_ALPHA", "1.0"))
    dkd_beta: float = float(os.getenv("DKD_BETA", "2.0"))
    dkd_warmup: int = int(os.getenv("DKD_WARMUP", "20"))
    dkd_ce_weight: float = float(os.getenv("DKD_CE_WEIGHT", "1.0"))
    # Plan item "D" (DOT, Wang et al. ICCV 2023): "dot" keeps SEPARATE momentum buffers for the task loss (CE) and the
    # distillation loss (KD + BC), with momentum 0.9 -/+ dot_delta, following megvii-research/mdistiller (DELTA 0.075).
    # "sgd" (default) is the plain optimizer used by every previous experiment.
    optim_mode: str = os.getenv("OPTIM_MODE", "sgd")
    dot_delta: float = float(os.getenv("DOT_DELTA", "0.075"))
    # Every N batches log the L2 norm of d(loss term)/d(last-layer weight) for CE/KD/BC (0 disables).
    grad_log_every: int = int(os.getenv("GRAD_LOG_EVERY", "100"))

    @property
    def bc_weight(self) -> float:
        return self.kd_weight if self.lambda_bc < 0 else self.lambda_bc
    # Set to a positive integer only for a quick pipeline smoke test.
    limit_train_batches: int = int(os.getenv("LIMIT_TRAIN_BATCHES", "0"))
    limit_eval_batches: int = int(os.getenv("LIMIT_EVAL_BATCHES", "0"))


CFG = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORK_DIR = Path("/kaggle/working/streamlined_kd") if Path("/kaggle/working").exists() else Path.cwd() / "kaggle_outputs"
DATA_DIR = Path("/kaggle/working/data") if Path("/kaggle/working").exists() else Path.cwd() / "data"
KAGGLE_CIFAR_DIR = Path("/kaggle/input/cifar100")
WORK_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

print(f"PyTorch: {torch.__version__}")
print(f"torchvision: {torchvision.__version__}")
print(f"device: {DEVICE}")
if DEVICE.type != "cuda" and os.getenv("ALLOW_CPU", "0") != "1":
    raise RuntimeError("This 240-epoch experiment requires a Kaggle GPU. Enable GPU in Notebook settings.")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = (CFG.seed + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


seed_everything(CFG.seed)

# %% [markdown]
# ## Minimal CIFAR ResNet implementation
#
# Only the ResNet20 student and ResNet56 teacher are retained from the original
# repository. Both return logits directly because SKD does not consume features.

# %%
def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual, inplace=True)


class CifarResNet(nn.Module):
    def __init__(self, depth: int, num_classes: int = 100):
        super().__init__()
        if (depth - 2) % 6 != 0:
            raise ValueError("CIFAR ResNet depth must be 6n+2")
        blocks_per_stage = (depth - 2) // 6
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, blocks_per_stage, stride=1)
        self.layer2 = self._make_layer(32, blocks_per_stage, stride=2)
        self.layer3 = self._make_layer(64, blocks_per_stage, stride=2)
        self.fc = nn.Linear(64, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        layers.extend(BasicBlock(planes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.avg_pool2d(x, 8)
        return self.fc(torch.flatten(x, 1))


def resnet20() -> CifarResNet:
    return CifarResNet(20, num_classes=100)


def resnet56() -> CifarResNet:
    return CifarResNet(56, num_classes=100)

# %% [markdown]
# ## Paper data pipeline
#
# The repository concatenates two complete CIFAR-100 training sets: one weakly
# augmented and one strongly augmented with its custom RandAugment. This means
# each epoch contains 100,000 samples, not 50,000.

# %%
def _autocontrast(img: Image.Image, _: float) -> Image.Image:
    return ImageOps.autocontrast(img)


def _brightness(img: Image.Image, value: float) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(value)


def _color(img: Image.Image, value: float) -> Image.Image:
    return ImageEnhance.Color(img).enhance(value)


def _contrast(img: Image.Image, value: float) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(value)


def _equalize(img: Image.Image, _: float) -> Image.Image:
    return ImageOps.equalize(img)


def _identity(img: Image.Image, _: float) -> Image.Image:
    return img


def _posterize(img: Image.Image, value: float) -> Image.Image:
    return ImageOps.posterize(img, max(1, int(value)))


def _rotate(img: Image.Image, value: float) -> Image.Image:
    return img.rotate(value)


def _sharpness(img: Image.Image, value: float) -> Image.Image:
    return ImageEnhance.Sharpness(img).enhance(value)


def _shear_x(img: Image.Image, value: float) -> Image.Image:
    return img.transform(img.size, Image.Transform.AFFINE, (1, value, 0, 0, 1, 0))


def _shear_y(img: Image.Image, value: float) -> Image.Image:
    return img.transform(img.size, Image.Transform.AFFINE, (1, 0, 0, value, 1, 0))


def _solarize(img: Image.Image, value: float) -> Image.Image:
    return ImageOps.solarize(img, value)


def _translate_x(img: Image.Image, value: float) -> Image.Image:
    return img.transform(img.size, Image.Transform.AFFINE, (1, 0, value * img.size[0], 0, 1, 0))


def _translate_y(img: Image.Image, value: float) -> Image.Image:
    return img.transform(img.size, Image.Transform.AFFINE, (1, 0, 0, 0, 1, value * img.size[1]))


def _cutout(img: Image.Image, value: float) -> Image.Image:
    size = value * img.size[0]
    x0 = int(max(0, np.random.uniform(img.size[0]) - size / 2))
    y0 = int(max(0, np.random.uniform(img.size[1]) - size / 2))
    result = img.copy()
    ImageDraw.Draw(result).rectangle((x0, y0, min(img.size[0], x0 + size), min(img.size[1], y0 + size)), fill=(125, 123, 114))
    return result


class PaperRandAugment:
    def __init__(self, n: int = 2):
        self.n = n
        self.operations = [
            (_autocontrast, 0, 1), (_brightness, 0.05, 0.95), (_color, 0.05, 0.95),
            (_contrast, 0.05, 0.95), (_equalize, 0, 1), (_identity, 0, 1),
            (_posterize, 0, 1), (_rotate, -30, 30), (_sharpness, 0.05, 0.95),
            (_shear_x, -0.3, 0.3), (_shear_y, -0.3, 0.3), (_solarize, 0, 256),
            (_translate_x, -0.3, 0.3), (_translate_y, -0.3, 0.3),
        ]

    def __call__(self, img: Image.Image) -> Image.Image:
        for operation, low, high in random.choices(self.operations, k=self.n):
            img = operation(img, low + (high - low) * random.random())
        return _cutout(img, random.random() * 0.5)


def prepare_cifar100() -> None:
    """Stage Kaggle's original CIFAR-100 Python files for torchvision.

    CIFAR100 still verifies the official MD5 checksums after these links are made.
    """
    candidates = [KAGGLE_CIFAR_DIR]
    input_root = Path("/kaggle/input")
    if input_root.is_dir():
        candidates.extend(path.parent for path in input_root.rglob("train"))
    source_dir = next(
        (path for path in candidates if all((path / name).is_file() for name in ("train", "test", "meta"))),
        None,
    )
    if source_dir is None:
        raise RuntimeError(
            "Attached CIFAR-100 files train/test/meta were not found under /kaggle/input. "
            "Refusing to use the slow external download."
        )

    target_dir = DATA_DIR / torchvision.datasets.CIFAR100.base_folder
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "test", "meta"):
        source = source_dir / name
        target = target_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"Required Kaggle CIFAR-100 file is missing: {source}")
        if not target.exists():
            target.symlink_to(source)
    print(f"Using checksum-verified CIFAR-100 from {source_dir}")


def make_loaders() -> tuple[DataLoader, DataLoader]:
    prepare_cifar100()
    normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    weak = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize,
    ])
    strong = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        PaperRandAugment(2), transforms.ToTensor(), normalize,
    ])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    weak_set = torchvision.datasets.CIFAR100(DATA_DIR, train=True, download=False, transform=weak)
    strong_set = torchvision.datasets.CIFAR100(DATA_DIR, train=True, download=False, transform=strong)
    test_set = torchvision.datasets.CIFAR100(DATA_DIR, train=False, download=False, transform=test_transform)
    generator = torch.Generator().manual_seed(CFG.seed)
    train_loader = DataLoader(
        ConcatDataset([weak_set, strong_set]), batch_size=CFG.batch_size, shuffle=True,
        num_workers=CFG.num_workers, pin_memory=True, persistent_workers=CFG.num_workers > 0,
        worker_init_fn=seed_worker, generator=generator,
    )
    test_loader = DataLoader(
        test_set, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers,
        pin_memory=True, persistent_workers=CFG.num_workers > 0, worker_init_fn=seed_worker,
    )
    return train_loader, test_loader

# %% [markdown]
# ## Streamlined Knowledge Distillation loss
#
# `paper_skd_loss` deliberately matches `Distiller/SKD.py`. In particular, the
# repository first reduces KD/SKD to scalars and then multiplies those scalars by
# the median masks. Keeping that behavior is important for an implementation-level
# reproduction of the reported result.

# %%
def standardize_logits(logits: torch.Tensor) -> torch.Tensor:
    """Per-sample logit standardization (Sun et al., CVPR 2024): zero mean, unit std over the class dimension."""
    return (logits - logits.mean(dim=1, keepdim=True)) / (1e-7 + logits.std(dim=1, keepdim=True))


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float, per_sample: bool = False) -> torch.Tensor:
    if CFG.logit_std:
        student_logits = standardize_logits(student_logits)
        teacher_logits = standardize_logits(teacher_logits)
    student_log_prob = F.log_softmax(student_logits / temperature, dim=1)
    teacher_prob = F.softmax(teacher_logits / temperature, dim=1)
    per = F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(1) * temperature**2
    return per if per_sample else per.mean()


def paper_skd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, tik: float) -> torch.Tensor:
    num_classes = student_logits.size(1)
    student_logits = F.normalize(student_logits, p=2, dim=1)
    teacher_logits = F.normalize(teacher_logits, p=2, dim=1)
    difference = (student_logits @ student_logits.T - teacher_logits @ teacher_logits.T) / num_classes
    covariance = torch.cov(difference.T) + tik * torch.eye(difference.size(1), device=difference.device)
    covariance_inv = torch.cholesky_inverse(torch.linalg.cholesky(covariance))
    distances = torch.einsum("bi,ij,bj->b", difference, covariance_inv, difference)
    return torch.sqrt(distances).mean()


LOGIT_STATS = {"student_std": 0.0, "teacher_std": 0.0, "count": 0.0}


def pop_logit_stats() -> dict[str, float]:
    """Mean per-sample std of the RAW student/teacher logits since the last call, then reset."""
    count = max(LOGIT_STATS["count"], 1.0)
    stats = {"student_logit_std": LOGIT_STATS["student_std"] / count, "teacher_logit_std": LOGIT_STATS["teacher_std"] / count}
    for key in LOGIT_STATS:
        LOGIT_STATS[key] = 0.0
    return stats


MASK_STATS = {"selected": 0.0, "selected_correct": 0.0, "rest": 0.0, "rest_correct": 0.0}


def pop_mask_stats() -> dict[str, float]:
    """Teacher top-1 (%) on the KD-selected vs. non-selected samples since the last call, then reset."""
    stats = dict(MASK_STATS)
    for key in MASK_STATS:
        MASK_STATS[key] = 0.0
    total = stats["selected"] + stats["rest"]
    return {
        "teacher_top1_selected": 100.0 * stats["selected_correct"] / max(stats["selected"], 1.0),
        "teacher_top1_rest": 100.0 * stats["rest_correct"] / max(stats["rest"], 1.0),
        "kd_selected_fraction": stats["selected"] / max(total, 1.0),
    }


CURRENT_EPOCH = {"value": 1}  # 1-indexed epoch, used by the DKD warm-up weight
DKD_STATS = {"tckd": 0.0, "nckd": 0.0, "count": 0.0}


def pop_dkd_stats() -> dict[str, float]:
    """Mean unweighted TCKD / NCKD (before alpha, beta and warm-up) since the last call, then reset."""
    count = max(DKD_STATS["count"], 1.0)
    stats = {"dkd_tckd": DKD_STATS["tckd"] / count, "dkd_nckd": DKD_STATS["nckd"] / count}
    for key in DKD_STATS:
        DKD_STATS[key] = 0.0
    return stats


def dkd_terms(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor, track_stats: bool = True
) -> dict[str, torch.Tensor]:
    """Decoupled KD, same maths as mdistiller's `dkd_loss` (TCKD: target vs. rest; NCKD: among non-target classes)."""
    temperature = CFG.temperature
    batch = labels.size(0)
    gt_mask = torch.zeros_like(student_logits).scatter_(1, labels.reshape(-1, 1), 1).bool()
    other_mask = torch.ones_like(student_logits).scatter_(1, labels.reshape(-1, 1), 0).bool()
    pred_student = F.softmax(student_logits / temperature, dim=1)
    pred_teacher = F.softmax(teacher_logits / temperature, dim=1)

    def binary(prob: torch.Tensor) -> torch.Tensor:
        return torch.cat([(prob * gt_mask).sum(1, keepdim=True), (prob * other_mask).sum(1, keepdim=True)], dim=1)

    tckd = F.kl_div(torch.log(binary(pred_student)), binary(pred_teacher), reduction="sum") * temperature**2 / batch
    pred_teacher_rest = F.softmax(teacher_logits / temperature - 1000.0 * gt_mask, dim=1)
    log_pred_student_rest = F.log_softmax(student_logits / temperature - 1000.0 * gt_mask, dim=1)
    nckd = F.kl_div(log_pred_student_rest, pred_teacher_rest, reduction="sum") * temperature**2 / batch
    if track_stats:
        DKD_STATS["tckd"] += tckd.item()
        DKD_STATS["nckd"] += nckd.item()
        DKD_STATS["count"] += 1.0
    return {
        "ce": F.cross_entropy(student_logits, labels),
        "kd": CFG.dkd_alpha * tckd + CFG.dkd_beta * nckd,
        "bc": student_logits.new_zeros(()),
    }


def raw_loss_terms(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor, track_stats: bool = True
) -> dict[str, torch.Tensor]:
    """Unweighted CE / KD / BC terms (KD and BC already include their masks)."""
    if track_stats:
        LOGIT_STATS["student_std"] += student_logits.detach().std(dim=1).mean().item()
        LOGIT_STATS["teacher_std"] += teacher_logits.detach().std(dim=1).mean().item()
        LOGIT_STATS["count"] += 1.0
    if CFG.loss_mode == "dkd":
        return dkd_terms(student_logits, teacher_logits, labels, track_stats)
    teacher_probability = F.softmax(teacher_logits.detach(), dim=1)
    confidence = teacher_probability.max(dim=1).values.detach()
    confidence_threshold = np.percentile(confidence.cpu().numpy().flatten(), 50)
    if CFG.kd_mask_select == "high":
        confidence_mask = confidence.ge(confidence_threshold)
    else:
        confidence_mask = confidence.le(confidence_threshold)
    if track_stats:
        teacher_correct = teacher_logits.detach().argmax(dim=1).eq(labels)
        MASK_STATS["selected"] += confidence_mask.sum().item()
        MASK_STATS["selected_correct"] += (teacher_correct & confidence_mask).sum().item()
        MASK_STATS["rest"] += (~confidence_mask).sum().item()
        MASK_STATS["rest_correct"] += (teacher_correct & ~confidence_mask).sum().item()
    class_confidence = teacher_probability.sum(dim=0).detach()
    class_threshold = np.percentile(class_confidence.cpu().numpy().flatten(), 50)
    class_mask = class_confidence.le(class_threshold)
    if CFG.kd_mask_mode == "per_sample":
        kd_term = (kd_loss(student_logits, teacher_logits, CFG.temperature, per_sample=True) * confidence_mask).mean()
    else:
        kd_term = (kd_loss(student_logits, teacher_logits, CFG.temperature) * confidence_mask).mean()
    return {
        "ce": F.cross_entropy(student_logits, labels),
        "kd": kd_term,
        "bc": (paper_skd_loss(student_logits, teacher_logits, CFG.tik) * class_mask).mean(),
    }


def distillation_losses(student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
    raw = raw_loss_terms(student_logits, teacher_logits, labels)
    if CFG.loss_mode == "dkd":
        warmup = min(CURRENT_EPOCH["value"] / max(CFG.dkd_warmup, 1), 1.0)
        return {"ce": CFG.dkd_ce_weight * raw["ce"], "kd": warmup * raw["kd"], "bc": 0.0 * raw["bc"]}
    return {
        "ce": CFG.ce_weight * raw["ce"],
        "kd": CFG.kd_weight * raw["kd"],
        "bc": CFG.bc_weight * raw["bc"],
    }


GRAD_STATS = {"ce": 0.0, "kd": 0.0, "bc": 0.0, "count": 0.0}


def probe_last_layer_grads(
    student: nn.Module, student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor
) -> None:
    """Accumulate ||d term / d fc.weight||_2 of each UNWEIGHTED term (does not touch the real backward pass)."""
    raw = raw_loss_terms(student_logits, teacher_logits, labels, track_stats=False)
    for name, term in raw.items():
        if not term.requires_grad:
            continue
        grad = torch.autograd.grad(term, student.fc.weight, retain_graph=True, allow_unused=True)[0]
        GRAD_STATS[name] += 0.0 if grad is None else grad.norm().item()
    GRAD_STATS["count"] += 1.0


def pop_grad_stats() -> dict[str, float]:
    count = max(GRAD_STATS["count"], 1.0)
    stats = {f"gradnorm_{name}": GRAD_STATS[name] / count for name in ("ce", "kd", "bc")}
    for key in GRAD_STATS:
        GRAD_STATS[key] = 0.0
    return stats


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, limit_batches: int = 0) -> dict[str, float]:
    model.eval()
    total_loss = total_top1 = total_top5 = total_examples = 0.0
    for batch_index, (images, labels) in enumerate(loader):
        if limit_batches and batch_index >= limit_batches:
            break
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        logits = model(images)
        batch_size = labels.size(0)
        total_loss += F.cross_entropy(logits, labels).item() * batch_size
        predictions = logits.topk(5, dim=1).indices
        matches = predictions.eq(labels[:, None])
        total_top1 += matches[:, :1].sum().item()
        total_top5 += matches.sum().item()
        total_examples += batch_size
    return {
        "loss": total_loss / total_examples,
        "top1": 100.0 * total_top1 / total_examples,
        "top5": 100.0 * total_top5 / total_examples,
    }


# --- Distillation-Oriented Trainer (DOT) ---
# Vendored from megvii-research/mdistiller `mdistiller/engine/dot.py` (MIT license) so the notebook stays self-contained.
# Only change: the unused `torch.optim._functional` import is dropped (it no longer exists in recent PyTorch).
def _check_in(t, tensors):
    for i in tensors:
        if t is i:
            return True
    return False


def _dot_update(params, d_p_list, momentum_buffer_list, kd_grad_buffer, kd_momentum_buffer, kd_params,
                *, weight_decay, momentum, momentum_kd, lr, dampening):
    for i, param in enumerate(params):
        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)
        if momentum != 0:
            buf = momentum_buffer_list[i]
            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            elif _check_in(param, kd_params):
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
            else:
                buf.mul_((momentum_kd + momentum) / 2.0).add_(d_p, alpha=1 - dampening)
            d_p = buf
        # update params with task-loss grad
        param.add_(d_p, alpha=-lr)

    for i, (d_p, buf, p) in enumerate(zip(kd_grad_buffer, kd_momentum_buffer, kd_params)):
        # update params with kd-loss grad
        if buf is None:
            buf = torch.clone(d_p).detach()
            kd_momentum_buffer[i] = buf
        elif _check_in(p, params):
            buf.mul_(momentum_kd).add_(d_p, alpha=1 - dampening)
        else:
            if weight_decay != 0:
                d_p = d_p.add(p, alpha=weight_decay)
            buf.mul_((momentum_kd + momentum) / 2.0).add_(d_p, alpha=1 - dampening)
        p.add_(buf, alpha=-lr)


class DistillationOrientedTrainer(torch.optim.Optimizer):
    """Usage per iteration: zero_grad; loss_kd.backward(retain_graph=True); step_kd(); zero_grad; loss_task.backward(); step()."""

    def __init__(self, params, lr, momentum=0, momentum_kd=0, dampening=0, weight_decay=0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum_kd < 0.0:
            raise ValueError("Invalid momentum value")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, momentum=momentum, momentum_kd=momentum_kd, dampening=dampening, weight_decay=weight_decay)
        self.kd_grad_buffer = []
        self.kd_grad_params = []
        self.kd_momentum_buffer = []
        super().__init__(params, defaults)

    @torch.no_grad()
    def step_kd(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        assert len(self.param_groups) == 1, "Only implemented for one parameter group."
        for group in self.param_groups:
            params_with_grad, d_p_list, momentum_kd_buffer_list = [], [], []
            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    state = self.state[p]
                    momentum_kd_buffer_list.append(state.get("momentum_kd_buffer"))
        self.kd_momentum_buffer = momentum_kd_buffer_list
        self.kd_grad_buffer = d_p_list
        self.kd_grad_params = params_with_grad
        return loss

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        assert len(self.param_groups) == 1, "Only implemented for one parameter group."
        for group in self.param_groups:
            params_with_grad, d_p_list, momentum_buffer_list = [], [], []
            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    momentum_buffer_list.append(self.state[p].get("momentum_buffer"))
            _dot_update(
                params_with_grad, d_p_list, momentum_buffer_list, self.kd_grad_buffer, self.kd_momentum_buffer, self.kd_grad_params,
                weight_decay=group["weight_decay"], momentum=group["momentum"], momentum_kd=group["momentum_kd"],
                lr=group["lr"], dampening=group["dampening"],
            )
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                self.state[p]["momentum_buffer"] = momentum_buffer
            for p, momentum_kd_buffer in zip(self.kd_grad_params, self.kd_momentum_buffer):
                self.state[p]["momentum_kd_buffer"] = momentum_kd_buffer
            self.kd_grad_buffer = []
            self.kd_grad_params = []
            self.kd_momentum_buffer = []
        return loss


DOT_STATS = {"cosine": 0.0, "count": 0.0}


def pop_dot_stats() -> dict[str, float]:
    """Mean cosine between the task (CE) and distillation (KD+BC) gradients of ALL parameters, since the last call."""
    stats = {"dot_grad_cosine": DOT_STATS["cosine"] / max(DOT_STATS["count"], 1.0)}
    for key in DOT_STATS:
        DOT_STATS[key] = 0.0
    return stats


def _flat_grad(parameters) -> torch.Tensor:
    return torch.cat([p.grad.detach().reshape(-1) for p in parameters if p.grad is not None])


def train_one_epoch(student: nn.Module, teacher: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer) -> dict[str, float]:
    student.train()
    teacher.eval()
    totals = {"loss": 0.0, "ce": 0.0, "kd": 0.0, "bc": 0.0, "top1": 0.0, "examples": 0.0}
    progress = tqdm(loader, leave=False)
    for batch_index, (images, labels) in enumerate(progress):
        if CFG.limit_train_batches and batch_index >= CFG.limit_train_batches:
            break
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        with torch.no_grad():
            teacher_logits = teacher(images)
        student_logits = student(images)
        losses = distillation_losses(student_logits, teacher_logits, labels)
        if CFG.grad_log_every and batch_index % CFG.grad_log_every == 0:
            probe_last_layer_grads(student, student_logits, teacher_logits, labels)
        loss = sum(losses.values())
        if CFG.optim_mode == "dot":
            # DOT: task loss (CE) and distillation loss (KD + BC) get their own backward pass and momentum buffer.
            loss_task = losses["ce"]
            loss_distill = losses["kd"] + losses["bc"]
            optimizer.zero_grad(set_to_none=True)
            loss_distill.backward(retain_graph=True)
            probe_cosine = bool(CFG.grad_log_every) and batch_index % CFG.grad_log_every == 0
            if probe_cosine:
                distill_grad = _flat_grad(student.parameters()).clone()
            optimizer.step_kd()
            optimizer.zero_grad(set_to_none=True)
            loss_task.backward()
            if probe_cosine:
                task_grad = _flat_grad(student.parameters())
                DOT_STATS["cosine"] += F.cosine_similarity(task_grad, distill_grad, dim=0).item()
                DOT_STATS["count"] += 1.0
            optimizer.step()
        else:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        totals["loss"] += loss.item() * batch_size
        for name, value in losses.items():
            totals[name] += value.item() * batch_size
        totals["top1"] += student_logits.argmax(1).eq(labels).sum().item()
        totals["examples"] += batch_size
        progress.set_postfix(loss=f"{loss.item():.3f}")

    examples = totals.pop("examples")
    return {name: (100.0 * value / examples if name == "top1" else value / examples) for name, value in totals.items()}

# %% [markdown]
# ## Download and verify the released teacher

# %%
TEACHER_URL = "https://raw.githubusercontent.com/HyunJunSik/StreamLined/main/CIFAR-100/Model_Pth/best_model_weights_resnet56.pth"
TEACHER_SHA256 = "2c8c5778c296ebf2bc3b226b5a2e5d4c93345c38676e48bb6e865206cc31d954"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def obtain_teacher_checkpoint() -> Path:
    candidates = [
        Path("/kaggle/input/streamlined-kd-cifar100-assets/best_model_weights_resnet56.pth"),
        Path.cwd() / "CIFAR-100" / "Model_Pth" / "best_model_weights_resnet56.pth",
        Path.cwd().parent / "CIFAR-100" / "Model_Pth" / "best_model_weights_resnet56.pth",
        WORK_DIR / "best_model_weights_resnet56.pth",
    ]
    for candidate in candidates:
        if candidate.exists() and sha256(candidate) == TEACHER_SHA256:
            return candidate
    target = WORK_DIR / "best_model_weights_resnet56.pth"
    print("Downloading released ResNet56 teacher...")
    urllib.request.urlretrieve(TEACHER_URL, target)
    actual_hash = sha256(target)
    if actual_hash != TEACHER_SHA256:
        raise RuntimeError(f"Teacher checkpoint checksum mismatch: {actual_hash}")
    return target


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state

# %% [markdown]
# ## Run the 240-epoch experiment

# %%
def run_experiment() -> dict[str, object]:
    train_loader, test_loader = make_loaders()
    teacher = resnet56().to(DEVICE)
    teacher.load_state_dict(load_state_dict(obtain_teacher_checkpoint()), strict=True)
    teacher.requires_grad_(False).eval()
    teacher_metrics = evaluate(teacher, test_loader, CFG.limit_eval_batches)
    print(f"Teacher ResNet56: top1={teacher_metrics['top1']:.2f}, top5={teacher_metrics['top5']:.2f}")

    # Re-seed after teacher validation so student initialization is independent of loader iteration.
    seed_everything(CFG.seed)
    student = resnet20().to(DEVICE)
    if CFG.optim_mode == "dot":
        optimizer = DistillationOrientedTrainer(
            student.parameters(), lr=CFG.learning_rate, momentum=CFG.momentum - CFG.dot_delta,
            momentum_kd=CFG.momentum + CFG.dot_delta, weight_decay=CFG.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            student.parameters(), lr=CFG.learning_rate, momentum=CFG.momentum, weight_decay=CFG.weight_decay
        )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(CFG.lr_milestones), gamma=0.1)
    best_top1 = float("-inf")
    best_state = None
    history: list[dict[str, float | int]] = []
    started_at = time.time()

    for epoch in range(CFG.epochs):
        CURRENT_EPOCH["value"] = epoch + 1
        train_metrics = train_one_epoch(student, teacher, train_loader, optimizer)
        test_metrics = evaluate(student, test_loader, CFG.limit_eval_batches)
        row = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **pop_mask_stats(),
            **pop_grad_stats(),
            **pop_logit_stats(),
            **pop_dkd_stats(),
            **pop_dot_stats(),
            "elapsed_seconds": time.time() - started_at,
            **{f"test_{key}": value for key, value in test_metrics.items()},
        }
        history.append(row)
        print(
            f"epoch {epoch + 1:03d}/{CFG.epochs} | lr={row['lr']:.4g} | "
            f"train top1={row['train_top1']:.2f} | test top1={row['test_top1']:.2f} | "
            f"best={max(best_top1, row['test_top1']):.2f}"
        )
        if test_metrics["top1"] > best_top1:
            best_top1 = test_metrics["top1"]
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in student.state_dict().items()})
            torch.save(best_state, WORK_DIR / "resnet20_skd_best.pth")
        torch.save(
            {"epoch": epoch + 1, "student": student.state_dict(), "optimizer": optimizer.state_dict(), "history": history},
            WORK_DIR / "last_checkpoint.pt",
        )
        scheduler.step()

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    student.load_state_dict(best_state)
    final_metrics = evaluate(student, test_loader, CFG.limit_eval_batches)
    elapsed_seconds = time.time() - started_at

    history_path = WORK_DIR / "history.csv"
    with history_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    summary: dict[str, object] = {
        "config": asdict(CFG),
        "teacher_metrics": teacher_metrics,
        "best_student_metrics": final_metrics,
        "best_student_top1_during_training": best_top1,
        "elapsed_seconds": elapsed_seconds,
        "artifacts": ["resnet20_skd_best.pth", "last_checkpoint.pt", "history.csv", "summary.json"],
    }
    with (WORK_DIR / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    SUMMARY = run_experiment()
