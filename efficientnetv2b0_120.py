"""Train and evaluate EfficientNetV2-B0 on 120px gastric histology images."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import json
import random

import numpy as np
import torch
from torch import nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision import transforms

from gastric_common import (
    CLASS_NAMES,
    compute_metrics,
    confusion_counts,
    discover_image_paths,
    stratified_split,
    tune_threshold,
    write_history_csv,
    write_json,
)


IMAGE_SIZE = 120
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TrainConfig:
    data_dir: Path = Path("data/GasHisSDB/120")
    test_dir: Path | None = None
    output_dir: Path = Path("runs/efficientnetv2b0_120")
    batch_size: int = 32
    epochs: int = 25
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    validation_size: float = 0.2
    seed: int = 42
    num_workers: int = 2
    early_stopping_patience: int = 5
    pretrained: bool = True


def validate_config(config: TrainConfig) -> None:
    if config.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if config.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.weight_decay < 0:
        raise ValueError("weight_decay cannot be negative")
    if not 0 < config.validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1")
    if config.num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if config.early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be at least 1")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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


@torch.no_grad()
def predict_probabilities(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[float], list[int]]:
    model.eval()
    labels: list[int] = []
    probabilities: list[float] = []

    for images, targets in loader:
        images = images.to(device)
        logits = model(images)
        batch_probabilities = torch.softmax(logits, dim=1)[:, 1]
        probabilities.extend(batch_probabilities.cpu().tolist())
        labels.extend(targets.tolist())

    return probabilities, labels


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float = 0.5) -> dict[str, object]:
    probabilities, labels = predict_probabilities(model, loader, device)
    return {
        "confusion_matrix": confusion_counts(labels, probabilities, threshold=threshold),
        "labels": labels,
        "metrics": compute_metrics(labels, probabilities, threshold=threshold),
        "probabilities": probabilities,
    }


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


def make_loader(
    image_paths: list[Path],
    labels: list[int],
    batch_size: int,
    transform: transforms.Compose,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        GastricImageDataset(image_paths, labels, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train(config: TrainConfig) -> dict[str, float]:
    validate_config(config)
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

    train_loader = make_loader(
        train_paths, train_labels, config.batch_size, build_transforms(train=True), True, config.num_workers
    )
    val_loader = make_loader(
        val_paths, val_labels, config.batch_size, build_transforms(train=False), False, config.num_workers
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
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    checkpoint_path = config.output_dir / "best_model.pt"

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        val_result = evaluate(model, val_loader, device)
        metrics = dict(val_result["metrics"])
        tuned_threshold, tuned_metrics = tune_threshold(val_result["labels"], val_result["probabilities"])
        metrics["tuned_threshold"] = tuned_threshold
        metrics["tuned_f1"] = tuned_metrics["f1"]
        metrics["train_loss"] = train_loss
        metrics["epoch"] = float(epoch)
        history.append(metrics)
        print(json.dumps(metrics, sort_keys=True))

        if metrics["f1"] > best_metrics["f1"]:
            best_metrics = metrics
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "config": {key: str(value) for key, value in config.__dict__.items()},
                    "metrics": metrics,
                    "threshold": tuned_threshold,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                print(json.dumps({"early_stopped_at_epoch": epoch}, sort_keys=True))
                break

    write_json(config.output_dir / "history.json", history)
    write_history_csv(config.output_dir / "history.csv", history)
    return best_metrics


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=TrainConfig.data_dir)
    parser.add_argument("--test-dir", type=Path, default=TrainConfig.test_dir)
    parser.add_argument("--output-dir", type=Path, default=TrainConfig.output_dir)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--validation-size", type=float, default=TrainConfig.validation_size)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--early-stopping-patience", type=int, default=TrainConfig.early_stopping_patience)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    return TrainConfig(
        data_dir=args.data_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_size=args.validation_size,
        seed=args.seed,
        num_workers=args.num_workers,
        early_stopping_patience=args.early_stopping_patience,
        pretrained=not args.no_pretrained,
    )


if __name__ == "__main__":
    final_metrics = train(parse_args())
    print("best_metrics=" + json.dumps(final_metrics, sort_keys=True))
