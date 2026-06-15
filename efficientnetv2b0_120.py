"""Train and evaluate EfficientNetV2-B0 on 120px gastric histology images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch


IMAGE_SIZE = 120
CLASS_NAMES = ("benign", "malignant")


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
