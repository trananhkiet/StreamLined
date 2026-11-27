"""CIFAR ResNets (same layer names as kaggle/streamlined_kd_cifar100.py) + checkpoint loading + prediction."""

from __future__ import annotations

import hashlib
import io
import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .config import RUNS_DIR

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
]

MDISTILLER_TAR_URL = "https://github.com/megvii-research/mdistiller/releases/download/checkpoints/cifar_teachers.tar"
MDISTILLER_TEACHERS = {
    "resnet56": ("cifar_teachers/resnet56_vanilla/ckpt_epoch_240.pth",
                 "4143895c82105da0a73236230db11aee5be39b660fe21e1323037c774f994155"),
    "resnet110": ("cifar_teachers/resnet110_vanilla/ckpt_epoch_240.pth",
                  "75dc04faf468c078f1dfa010b51069fa3a8a4b91145379574e05c8e63e7441a9"),
}
DEPTHS = {"resnet20": 20, "resnet32": 32, "resnet56": 56, "resnet110": 110}


# ---------------------------------------------------------------- architecture (copied from the training script)


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
        blocks = (depth - 2) // 6
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, blocks, 1)
        self.layer2 = self._make_layer(32, blocks, 2)
        self.layer3 = self._make_layer(64, blocks, 2)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes, 1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes))
        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        layers.extend(BasicBlock(planes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer3(self.layer2(self.layer1(x)))
        x = F.avg_pool2d(x, 8)
        return self.fc(torch.flatten(x, 1))


def build_model(arch: str) -> CifarResNet:
    return CifarResNet(DEPTHS[arch])


# ---------------------------------------------------------------- checkpoints


def _torch_load(source):
    try:
        import numpy as np
        try:
            from numpy._core.multiarray import scalar
        except ImportError:
            from numpy.core.multiarray import scalar
        # CRD/mdistiller checkpoints store their accuracy as a numpy scalar next to the weights.
        safe = [(scalar, "numpy.core.multiarray.scalar"), scalar, np.dtype, type(np.dtype("float64")),
                type(np.dtype("float32"))]
        with torch.serialization.safe_globals(safe):
            return torch.load(source, map_location="cpu", weights_only=True)
    except (TypeError, AttributeError):
        return torch.load(source, map_location="cpu", weights_only=True)


def extract_state_dict(obj) -> dict[str, torch.Tensor]:
    """Accept plain state_dicts, mdistiller/CRD {'model': ...}, our last_checkpoint {'student': ...}."""
    state = obj
    for key in ("student", "model", "state_dict"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
    if not isinstance(state, dict):
        raise ValueError("Checkpoint không chứa state_dict hợp lệ")
    return {k.removeprefix("module."): v for k, v in state.items() if isinstance(v, torch.Tensor)}


def infer_arch(state: dict[str, torch.Tensor]) -> str:
    blocks = len({k.split(".")[1] for k in state if k.startswith("layer1.")})
    depth = 6 * blocks + 2
    for arch, d in DEPTHS.items():
        if d == depth:
            return arch
    raise ValueError(f"Không nhận ra kiến trúc (layer1 có {blocks} block)")


def load_model(source, arch: str | None = None) -> tuple[CifarResNet, str]:
    """`source`: path or bytes of a .pth/.pt file. Returns (model in eval mode, arch)."""
    raw = _torch_load(io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source)
    state = extract_state_dict(raw)
    arch = arch or infer_arch(state)
    model = build_model(arch)
    for key, value in model.state_dict().items():
        if key.endswith("num_batches_tracked") and key not in state:
            state[key] = torch.zeros_like(value)
    model.load_state_dict(state, strict=True)
    return model.eval(), arch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def teacher_path(arch: str, root: Path = RUNS_DIR / "teachers") -> Path:
    return root / f"mdistiller_{arch}.pth"


def download_teacher(arch: str, root: Path = RUNS_DIR / "teachers") -> Path:
    """Stream the 364 MB mdistiller tar from GitHub and keep only the requested teacher (checksum verified)."""
    member, expected = MDISTILLER_TEACHERS[arch]
    target = teacher_path(arch, root)
    if target.exists() and sha256(target) == expected:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")
    with urllib.request.urlopen(MDISTILLER_TAR_URL) as response, tarfile.open(fileobj=response, mode="r|") as tar:
        for info in tar:
            if info.name == member and info.isfile():
                with tar.extractfile(info) as src, tmp.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                break
        else:
            raise RuntimeError(f"{member} không có trong {MDISTILLER_TAR_URL}")
    if sha256(tmp) != expected:
        tmp.unlink()
        raise RuntimeError("Checksum teacher không khớp")
    tmp.replace(target)
    return target


# ---------------------------------------------------------------- inference


def preprocess(image: Image.Image) -> torch.Tensor:
    img = image.convert("RGB").resize((32, 32), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
    mean = torch.tensor(CIFAR100_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR100_STD).view(3, 1, 1)
    return ((x - mean) / std).unsqueeze(0)


@torch.no_grad()
def predict(model: nn.Module, image: Image.Image, k: int = 5) -> list[tuple[str, float]]:
    probs = F.softmax(model(preprocess(image)), dim=1)[0]
    values, indices = probs.topk(k)
    return [(CIFAR100_CLASSES[i], float(p)) for p, i in zip(values.tolist(), indices.tolist())]


def load_cifar100_test(root: Path = RUNS_DIR / "data"):
    """torchvision CIFAR-100 test split (downloads ~160 MB the first time)."""
    import torchvision

    return torchvision.datasets.CIFAR100(root=str(root), train=False, download=True)
