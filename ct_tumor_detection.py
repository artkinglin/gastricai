"""Train and evaluate a simple CNN for CT tumor screening PNG images."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


IMAGE_SIZE = 224
CLASS_NAMES = ("normal", "tumor")
LABEL_ALIASES = {
    "normal": ("normal", "negative", "benign", "no_tumor", "no-tumor"),
    "tumor": ("tumor", "positive", "malignant", "abnormal"),
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class TrainConfig:
    train_dir: Path = Path("data/train")
    test_dir: Path = Path("data/test")
    output_dir: Path = Path("runs/ct_tumor_detection")
    batch_size: int = 8
    epochs: int = 5
    learning_rate: float = 1e-3
    threshold: float = 0.5
    num_workers: int = 0
    plot: bool = False


def _candidate_label_dirs(data_dir: Path, aliases: tuple[str, ...]) -> list[Path]:
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
        for class_dir in _candidate_label_dirs(data_dir, LABEL_ALIASES[class_name]):
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    image_paths.append(image_path)
                    labels.append(label_index)

    if not image_paths:
        expected = ", ".join(CLASS_NAMES)
        raise FileNotFoundError(f"No images found under {data_dir}. Expected folders for: {expected}")

    if len(set(labels)) != len(CLASS_NAMES):
        counts = {name: labels.count(index) for index, name in enumerate(CLASS_NAMES)}
        raise ValueError(f"Every class must have at least one image, got counts={counts}")

    return image_paths, labels


def read_grayscale_image(path: Path, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return torch.from_numpy(image)


class CTScanDataset(Dataset):
    def __init__(self, image_paths: list[Path], labels: list[int]) -> None:
        if len(image_paths) != len(labels):
            raise ValueError("image_paths and labels must have the same length")
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = read_grayscale_image(self.image_paths[index])
        label = torch.tensor([self.labels[index]], dtype=torch.float32)
        return image, label


class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(128, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    seen = 0

    for inputs, labels in tqdm(dataloader, desc="train", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        seen += batch_size

    return running_loss / max(seen, 1)


@torch.no_grad()
def predict_probabilities(model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[list[float], list[int]]:
    model.eval()
    probabilities: list[float] = []
    labels: list[int] = []

    for inputs, targets in tqdm(dataloader, desc="eval", leave=False):
        logits = model(inputs.to(device))
        batch_probabilities = torch.sigmoid(logits).squeeze(1).cpu().tolist()
        probabilities.extend(float(value) for value in batch_probabilities)
        labels.extend(int(value.item()) for value in targets.squeeze(1))

    return probabilities, labels


def compute_metrics(labels: list[int], probabilities: list[float], threshold: float = 0.5) -> dict[str, float]:
    predictions = [1 if probability >= threshold else 0 for probability in probabilities]
    true_positive = sum(1 for label, prediction in zip(labels, predictions) if label == 1 and prediction == 1)
    true_negative = sum(1 for label, prediction in zip(labels, predictions) if label == 0 and prediction == 0)
    false_positive = sum(1 for label, prediction in zip(labels, predictions) if label == 0 and prediction == 1)
    false_negative = sum(1 for label, prediction in zip(labels, predictions) if label == 1 and prediction == 0)

    total = max(len(labels), 1)
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "accuracy": (true_positive + true_negative) / total,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "false_positive": float(false_positive),
        "false_negative": float(false_negative),
    }


def estimate_tumor_size_pixels(image_path: Path, threshold: int = 127) -> int:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    _, mask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return int(np.count_nonzero(mask))


def evaluate(model: nn.Module, dataloader: DataLoader, image_paths: list[Path], device: torch.device, threshold: float) -> dict[str, object]:
    probabilities, labels = predict_probabilities(model, dataloader, device)
    metrics = compute_metrics(labels, probabilities, threshold=threshold)
    tumor_sizes = [estimate_tumor_size_pixels(path) for path in image_paths]
    return {
        "metrics": metrics,
        "probabilities": probabilities,
        "tumor_sizes": tumor_sizes,
    }


def plot_results(probabilities: list[float], tumor_sizes: list[int], output_path: Path | None = None) -> None:
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(probabilities, bins=20, color="skyblue", edgecolor="black")
    plt.title("Tumor Probabilities")
    plt.xlabel("Probability")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    plt.hist(tumor_sizes, bins=20, color="salmon", edgecolor="black")
    plt.title("Thresholded Area")
    plt.xlabel("Pixels")
    plt.ylabel("Count")

    plt.tight_layout()
    if output_path is None:
        plt.show()
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
    plt.close()


def make_loader(image_paths: list[Path], labels: list[int], batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        CTScanDataset(image_paths, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train(config: TrainConfig) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_paths, train_labels = discover_image_paths(config.train_dir)
    test_paths, test_labels = discover_image_paths(config.test_dir)
    train_loader = make_loader(train_paths, train_labels, config.batch_size, True, config.num_workers)
    test_loader = make_loader(test_paths, test_labels, config.batch_size, False, config.num_workers)

    model = SimpleCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        result = evaluate(model, test_loader, test_paths, device, config.threshold)
        metrics = dict(result["metrics"])
        metrics["epoch"] = float(epoch)
        metrics["train_loss"] = train_loss
        history.append(metrics)
        print(json.dumps(metrics, sort_keys=True))

    checkpoint_path = config.output_dir / "ct_tumor_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "config": {key: str(value) for key, value in config.__dict__.items()},
            "history": history,
        },
        checkpoint_path,
    )

    final_result = evaluate(model, test_loader, test_paths, device, config.threshold)
    if config.plot:
        plot_results(
            final_result["probabilities"],
            final_result["tumor_sizes"],
            config.output_dir / "evaluation_histograms.png",
        )

    return {
        "checkpoint": str(checkpoint_path),
        "metrics": final_result["metrics"],
    }


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=TrainConfig.train_dir)
    parser.add_argument("--test-dir", type=Path, default=TrainConfig.test_dir)
    parser.add_argument("--output-dir", type=Path, default=TrainConfig.output_dir)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--threshold", type=float, default=TrainConfig.threshold)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    return TrainConfig(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        threshold=args.threshold,
        num_workers=args.num_workers,
        plot=args.plot,
    )


if __name__ == "__main__":
    final_result = train(parse_args())
    print("final_result=" + json.dumps(final_result, sort_keys=True))
