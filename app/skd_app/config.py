"""Static choices for the demo app: models, recipes, paths, and the env-var mapping of the training script."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = APP_DIR.parent
TRAIN_SCRIPT = REPO_DIR / "kaggle" / "streamlined_kd_cifar100.py"
RUNS_DIR = APP_DIR / "runs"

# Pinned Kaggle image used by every experiment in this repo. Do not change.
DOCKER_IMAGE = (
    "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"
)
CIFAR100_KAGGLE_DATASET = "fedesoriano/cifar100"

TEACHERS = {
    "resnet56": "ResNet56 (mdistiller, 72.41%)",
    "resnet110": "ResNet110 (mdistiller, 74.31%)",
}
STUDENTS = {
    "resnet20": "ResNet20",
    "resnet32": "ResNet32",
}
DATASETS = {
    "cifar100": ("CIFAR-100", True),
    "cifar10": ("CIFAR-10 (chưa có teacher pretrained)", False),
    "tiny-imagenet": ("Tiny-ImageNet (chưa hỗ trợ)", False),
}
RECIPES = {
    "E4": "E4 – tái lập SKD (KD 0.5 cho mọi mẫu)",
    "F1b": "F1b – KD 0.6 khi teacher đúng, 0.5 khi sai",
}

# Paper numbers (CIFAR-100 top-1) and our reproduction, shown next to results.
REFERENCE = {
    ("resnet56", "resnet20"): {"paper_skd": 72.50, "paper_kd": 71.74, "ours_e4": 72.11, "ours_f1b": 72.44},
    ("resnet110", "resnet32"): {"paper_skd": 74.84, "paper_kd": 74.01, "ours_e4": 74.71, "ours_f1b": 74.70},
}

SHORT = {"resnet20": "r20", "resnet32": "r32", "resnet56": "r56", "resnet110": "r110"}


@dataclass
class TrainConfig:
    teacher: str = "resnet56"
    student: str = "resnet20"
    dataset: str = "cifar100"
    recipe: str = "F1b"
    epochs: int = 300
    learning_rate: float = 0.1
    lr_milestones: str = "100,188,250"
    batch_size: int = 64
    temperature: float = 4.0
    seed: int = 2022
    ce_weight: float = 0.5
    kd_weight: float = 0.5
    smoke_test: bool = False
    extra_env: dict[str, str] = field(default_factory=dict)

    def validate(self) -> list[str]:
        errors = []
        if self.teacher not in TEACHERS:
            errors.append(f"teacher không hợp lệ: {self.teacher}")
        if self.student not in STUDENTS:
            errors.append(f"student không hợp lệ: {self.student}")
        if not DATASETS.get(self.dataset, ("", False))[1]:
            errors.append(f"dataset chưa hỗ trợ: {self.dataset}")
        if self.recipe not in RECIPES:
            errors.append(f"recipe không hợp lệ: {self.recipe}")
        if not 1 <= self.epochs <= 400:
            errors.append("epochs phải trong [1, 400]")
        if not 0 < self.learning_rate <= 1:
            errors.append("learning rate phải trong (0, 1]")
        try:
            milestones = [int(m) for m in self.lr_milestones.split(",") if m.strip()]
        except ValueError:
            errors.append("LR milestones phải là các số nguyên cách nhau bởi dấu phẩy")
        else:
            if milestones != sorted(milestones) or any(m <= 0 for m in milestones):
                errors.append("LR milestones phải tăng dần và > 0")
        if self.batch_size not in (32, 64, 128, 256):
            errors.append("batch size phải là 32/64/128/256")
        if not 0.5 <= self.temperature <= 20:
            errors.append("temperature phải trong [0.5, 20]")
        return errors

    def to_env(self) -> dict[str, str]:
        """Map the form to the training script's environment variables (see kaggle/streamlined_kd_cifar100.py)."""
        env = {
            "SEED": str(self.seed),
            "EPOCHS": str(self.epochs),
            "TEACHER_SOURCE": "mdistiller",
            "TEACHER_ARCH": self.teacher,
            "STUDENT_ARCH": self.student,
            "LEARNING_RATE": f"{self.learning_rate:g}",
            "LR_MILESTONES": ",".join(m.strip() for m in self.lr_milestones.split(",") if m.strip()),
            "BATCH_SIZE": str(self.batch_size),
            "TEMPERATURE": f"{self.temperature:g}",
        }
        if self.recipe == "F1b":
            env.update({"SAMPLE_WEIGHT_MODE": "teacher_correct", "SW_TRUST": "0.6", "SW_DISTRUST": "0.5"})
        else:
            env.update({"CE_WEIGHT": f"{self.ce_weight:g}", "KD_WEIGHT": f"{self.kd_weight:g}"})
        if self.smoke_test:
            env.update({"LIMIT_TRAIN_BATCHES": "20", "LIMIT_EVAL_BATCHES": "5"})
        env.update(self.extra_env)
        return env

    def slug(self, stamp: str) -> str:
        """Kaggle kernel slug, e.g. skd-app-r56-r20-f1b-s2022-1127-0715 (<= 50 chars)."""
        parts = ["skd-app", SHORT[self.teacher], SHORT[self.student], self.recipe.lower(), f"s{self.seed}"]
        if self.smoke_test:
            parts.append("smoke")
        return "-".join(parts + [stamp])[:50].rstrip("-")


def estimated_hours(cfg: TrainConfig) -> float:
    """Rough T4 wall time measured on our runs (R56->R20 ~ 63 s/epoch, R110->R32 ~ 78 s/epoch)."""
    if cfg.smoke_test:
        return 0.15
    per_epoch = {"resnet56": 63.0, "resnet110": 78.0}[cfg.teacher] * (64 / cfg.batch_size) ** 0.3
    return cfg.epochs * per_epoch / 3600 + 0.1
