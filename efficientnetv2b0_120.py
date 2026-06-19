"""Train and evaluate EfficientNetV2-B0 on 120px gastric histology images."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import json
import random
from typing import Iterable

import numpy as np
import torch
from torch import nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision import transforms


IMAGE_SIZE = 120
CLASS_NAMES = ("benign", "malignant")
LABEL_ALIASES = {
    "benign": ("benign", "normal", "negative"),
    "malignant": ("malignant", "abnormal", "tumor", "positive"),
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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
    pretrained: bool = True


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


def stratified_split(
    image_paths: list[Path],
    labels: list[int],
    validation_size: float,
    seed: int,
) -> tuple[list[Path], list[Path], list[int], list[int]]:
    return train_test_split(
        image_paths,
        labels,
        test_size=validation_size,
        random_state=seed,
        stratify=labels,
    )


def build_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_model(pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_V2_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_v2_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, len(CLASS_NAMES)),
    )
    return model


def class_weights(labels: list[int], device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.float32)
    if np.any(counts == 0):
        raise ValueError(f"Every class must have at least one sample, got counts={counts.tolist()}")
    weights = counts.sum() / (len(CLASS_NAMES) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_metrics(labels: list[int], probabilities: list[float], threshold: float = 0.5) -> dict[str, float]:
    predictions = [1 if probability >= threshold else 0 for probability in probabilities]
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, zero_division=0),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
    }
    if len(set(labels)) == 2:
        metrics["roc_auc"] = roc_auc_score(labels, probabilities)
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float = 0.5) -> dict[str, float]:
    model.eval()
    labels: list[int] = []
    probabilities: list[float] = []

    for images, targets in loader:
        images = images.to(device)
        logits = model(images)
        batch_probabilities = torch.softmax(logits, dim=1)[:, 1]
        probabilities.extend(batch_probabilities.cpu().tolist())
        labels.extend(targets.tolist())

    return compute_metrics(labels, probabilities, threshold=threshold)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    seen = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        seen += batch_size

    return running_loss / max(seen, 1)


def train(config: TrainConfig) -> dict[str, float]:
    seed_everything(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_paths, labels = discover_image_paths(config.data_dir)
    train_paths, val_paths, train_labels, val_labels = stratified_split(
        image_paths,
        labels,
        config.validation_size,
        config.seed,
    )

    train_dataset = GastricImageDataset(train_paths, train_labels, build_transforms(train=True))
    val_dataset = GastricImageDataset(val_paths, val_labels, build_transforms(train=False))
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(pretrained=config.pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_labels, device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_metrics: dict[str, float] = {"f1": -1.0}
    checkpoint_path = config.output_dir / "best_model.pt"

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        metrics = evaluate(model, val_loader, device)
        metrics["train_loss"] = train_loss
        metrics["epoch"] = float(epoch)
        print(json.dumps(metrics, sort_keys=True))

        if metrics["f1"] > best_metrics["f1"]:
            best_metrics = metrics
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "config": {key: str(value) for key, value in config.__dict__.items()},
                    "metrics": metrics,
                },
                checkpoint_path,
            )

    return best_metrics


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=TrainConfig.data_dir)
    parser.add_argument("--output-dir", type=Path, default=TrainConfig.output_dir)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--validation-size", type=float, default=TrainConfig.validation_size)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    return TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_size=args.validation_size,
        seed=args.seed,
        num_workers=args.num_workers,
        pretrained=not args.no_pretrained,
    )


if __name__ == "__main__":
    final_metrics = train(parse_args())
    print("best_metrics=" + json.dumps(final_metrics, sort_keys=True))
