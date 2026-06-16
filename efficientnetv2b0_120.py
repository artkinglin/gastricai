"""Train and evaluate EfficientNetV2-B0 on 120px gastric histology images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_SIZE = 120
CLASS_NAMES = ("benign", "malignant")
LABEL_ALIASES = {
    "benign": ("benign", "normal", "negative"),
    "malignant": ("malignant", "abnormal", "tumor", "positive"),
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class TrainConfig:
    data_dir: Path = Path("data/GasHisSDB/120")
    output_dir: Path = Path("runs/efficientnetv2b0_120")
    batch_size: int = 32
    epochs: int = 25
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    validation_size: float = 0.2
    seed: int = 42
    num_workers: int = 2


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _candidate_label_dirs(data_dir: Path, aliases: Iterable[str]) -> list[Path]:
    candidates: list[Path] = []
    for alias in aliases:
        candidates.extend(data_dir.glob(alias))
        candidates.extend(data_dir.glob(alias.capitalize()))
        candidates.extend(data_dir.glob(alias.upper()))
    return [path for path in candidates if path.is_dir()]


def discover_image_paths(data_dir: Path) -> tuple[list[Path], list[int]]:
    image_paths: list[Path] = []
    labels: list[int] = []

    for label_index, class_name in enumerate(CLASS_NAMES):
        class_dirs = _candidate_label_dirs(data_dir, LABEL_ALIASES[class_name])
        for class_dir in class_dirs:
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    image_paths.append(image_path)
                    labels.append(label_index)

    if not image_paths:
        raise FileNotFoundError(
            f"No images found under {data_dir}. Expected class folders like Normal/Abnormal."
        )

    return image_paths, labels


class GastricImageDataset(Dataset):
    def __init__(self, image_paths: list[Path], labels: list[int], transform=None) -> None:
        if len(image_paths) != len(labels):
            raise ValueError("image_paths and labels must have the same length")
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.image_paths[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return image, label
