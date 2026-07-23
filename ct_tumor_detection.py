"""Train and evaluate a simple CNN for CT tumor screening PNG images."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from experiment_config import coerce_path, get_config_value, load_config_file
from gastric_common import read_manifest, stratified_split
from ct_segmentation import segment_threshold_area


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
    train_manifest: Path | None = None
    test_dir: Path | None = Path("data/test")
    test_manifest: Path | None = None
    output_dir: Path = Path("runs/ct_tumor_detection")
    run_name: str | None = None
    batch_size: int = 8
    epochs: int = 5
    learning_rate: float = 1e-3
    validation_size: float = 0.2
    threshold: float = 0.5
    tumor_area_threshold: int = 127
    pixel_spacing_mm: float | None = None
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"
    max_images_per_class: int | None = None
    plot: bool = False


def validate_config(config: TrainConfig) -> None:
    if config.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if config.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0 < config.validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1")
    if not 0.0 <= config.threshold <= 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0")
    if not 0 <= config.tumor_area_threshold <= 255:
        raise ValueError("tumor_area_threshold must be between 0 and 255")
    if config.pixel_spacing_mm is not None and config.pixel_spacing_mm <= 0:
        raise ValueError("pixel_spacing_mm must be positive")
    if config.num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if config.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if config.max_images_per_class is not None and config.max_images_per_class < 1:
        raise ValueError("max_images_per_class must be at least 1")


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(device_name)


def optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _candidate_label_dirs(data_dir: Path, aliases: tuple[str, ...]) -> list[Path]:
    candidates: list[Path] = []
    for alias in aliases:
        candidates.extend(data_dir.glob(alias))
        candidates.extend(data_dir.glob(alias.capitalize()))
        candidates.extend(data_dir.glob(alias.upper()))
    return [path for path in candidates if path.is_dir()]


def discover_image_paths(data_dir: Path, max_images_per_class: int | None = None) -> tuple[list[Path], list[int]]:
    image_paths: list[Path] = []
    labels: list[int] = []

    for label_index, class_name in enumerate(CLASS_NAMES):
        class_paths: list[Path] = []
        for class_dir in _candidate_label_dirs(data_dir, LABEL_ALIASES[class_name]):
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    class_paths.append(image_path)
        if max_images_per_class is not None:
            class_paths = class_paths[:max_images_per_class]
        image_paths.extend(class_paths)
        labels.extend([label_index] * len(class_paths))

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


def positive_class_weight(labels: list[int], device: torch.device) -> torch.Tensor:
    positive_count = sum(1 for label in labels if label == 1)
    negative_count = sum(1 for label in labels if label == 0)
    if positive_count == 0 or negative_count == 0:
        raise ValueError(
            f"Both classes are required to compute class weight, got normal={negative_count}, tumor={positive_count}"
        )
    return torch.tensor([negative_count / positive_count], dtype=torch.float32, device=device)


def estimate_tumor_area(image_path: Path, threshold: int = 127, pixel_spacing_mm: float | None = None) -> tuple[int, float | None]:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    spacing = (pixel_spacing_mm, pixel_spacing_mm) if pixel_spacing_mm is not None else None
    result = segment_threshold_area(image, threshold=threshold, pixel_spacing=spacing)
    return result.pixel_count, result.area_mm2


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    image_paths: list[Path],
    device: torch.device,
    threshold: float,
    tumor_area_threshold: int,
    pixel_spacing_mm: float | None,
) -> dict[str, object]:
    probabilities, labels = predict_probabilities(model, dataloader, device)
    metrics = compute_metrics(labels, probabilities, threshold=threshold)
    tumor_areas = [estimate_tumor_area(path, threshold=tumor_area_threshold, pixel_spacing_mm=pixel_spacing_mm) for path in image_paths]
    tumor_sizes = [area[0] for area in tumor_areas]
    tumor_area_mm2 = [area[1] for area in tumor_areas]
    return {
        "labels": labels,
        "metrics": metrics,
        "probabilities": probabilities,
        "tumor_area_mm2": tumor_area_mm2,
        "tumor_sizes": tumor_sizes,
    }


def write_prediction_report(
    output_path: Path,
    image_paths: list[Path],
    labels: list[int],
    probabilities: list[float],
    tumor_sizes: list[int],
    tumor_area_mm2: list[float | None],
    threshold: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image_path", "label", "probability", "prediction", "tumor_size_pixels", "tumor_area_mm2"],
        )
        writer.writeheader()
        for image_path, label, probability, tumor_size, area_mm2 in zip(
            image_paths, labels, probabilities, tumor_sizes, tumor_area_mm2
        ):
            writer.writerow(
                {
                    "image_path": str(image_path),
                    "label": CLASS_NAMES[label],
                    "probability": probability,
                    "prediction": CLASS_NAMES[1 if probability >= threshold else 0],
                    "tumor_size_pixels": tumor_size,
                    "tumor_area_mm2": area_mm2,
                }
            )


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
    validate_config(config)
    seed_everything(config.seed)
    output_dir = config.output_dir / config.run_name if config.run_name else config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(config.device)

    train_paths, train_labels = (
        read_manifest(config.train_manifest, root_dir=config.train_dir, class_names=CLASS_NAMES)
        if config.train_manifest
        else discover_image_paths(config.train_dir, config.max_images_per_class)
    )
    if config.test_manifest is not None:
        test_paths, test_labels = read_manifest(config.test_manifest, root_dir=config.test_dir, class_names=CLASS_NAMES)
    elif config.test_dir is not None:
        test_paths, test_labels = discover_image_paths(config.test_dir, config.max_images_per_class)
    else:
        train_paths, test_paths, train_labels, test_labels = stratified_split(
            train_paths, train_labels, config.validation_size, config.seed
        )
    train_loader = make_loader(train_paths, train_labels, config.batch_size, True, config.num_workers)
    test_loader = make_loader(test_paths, test_labels, config.batch_size, False, config.num_workers)

    model = SimpleCNN().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_class_weight(train_labels, device))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history: list[dict[str, float]] = []
    best_f1 = -1.0
    checkpoint_path = output_dir / "best_ct_tumor_model.pt"
    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        result = evaluate(
            model, test_loader, test_paths, device, config.threshold, config.tumor_area_threshold, config.pixel_spacing_mm
        )
        metrics = dict(result["metrics"])
        metrics["epoch"] = float(epoch)
        metrics["train_loss"] = train_loss
        history.append(metrics)
        print(json.dumps(metrics, sort_keys=True))

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "config": {key: str(value) for key, value in config.__dict__.items()},
                    "metrics": metrics,
                },
                checkpoint_path,
            )
    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2, sort_keys=True), encoding="utf-8")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_result = evaluate(
        model, test_loader, test_paths, device, config.threshold, config.tumor_area_threshold, config.pixel_spacing_mm
    )
    predictions_path = output_dir / "predictions.csv"
    write_prediction_report(
        predictions_path,
        test_paths,
        final_result["labels"],
        final_result["probabilities"],
        final_result["tumor_sizes"],
        final_result["tumor_area_mm2"],
        config.threshold,
    )
    if config.plot:
        plot_results(
            final_result["probabilities"],
            final_result["tumor_sizes"],
            output_dir / "evaluation_histograms.png",
        )

    return {
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
        "predictions": str(predictions_path),
        "metrics": final_result["metrics"],
    }


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--train-dir", type=Path, default=TrainConfig.train_dir)
    parser.add_argument("--train-manifest", type=Path, default=TrainConfig.train_manifest)
    parser.add_argument("--test-dir", type=Path, default=TrainConfig.test_dir)
    parser.add_argument("--test-manifest", type=Path, default=TrainConfig.test_manifest)
    parser.add_argument("--output-dir", type=Path, default=TrainConfig.output_dir)
    parser.add_argument("--run-name", type=str, default=TrainConfig.run_name)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--validation-size", type=float, default=TrainConfig.validation_size)
    parser.add_argument("--threshold", type=float, default=TrainConfig.threshold)
    parser.add_argument("--tumor-area-threshold", type=int, default=TrainConfig.tumor_area_threshold)
    parser.add_argument("--pixel-spacing-mm", type=float, default=TrainConfig.pixel_spacing_mm)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=TrainConfig.device)
    parser.add_argument("--max-images-per-class", type=int, default=TrainConfig.max_images_per_class)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    config_values = load_config_file(args.config)
    return TrainConfig(
        train_dir=coerce_path(config_values.get("train_dir", args.train_dir)) or args.train_dir,
        train_manifest=coerce_path(config_values.get("train_manifest", args.train_manifest)),
        test_dir=coerce_path(get_config_value(config_values, "test_dir", args.test_dir)),
        test_manifest=coerce_path(config_values.get("test_manifest", args.test_manifest)),
        output_dir=coerce_path(config_values.get("output_dir", args.output_dir)) or args.output_dir,
        run_name=config_values.get("run_name", args.run_name),
        batch_size=int(config_values.get("batch_size", args.batch_size)),
        epochs=int(config_values.get("epochs", args.epochs)),
        learning_rate=float(config_values.get("learning_rate", args.learning_rate)),
        validation_size=float(config_values.get("validation_size", args.validation_size)),
        threshold=float(config_values.get("threshold", args.threshold)),
        tumor_area_threshold=int(config_values.get("tumor_area_threshold", args.tumor_area_threshold)),
        pixel_spacing_mm=optional_float(config_values.get("pixel_spacing_mm", args.pixel_spacing_mm)),
        num_workers=int(config_values.get("num_workers", args.num_workers)),
        seed=int(config_values.get("seed", args.seed)),
        device=str(config_values.get("device", args.device)),
        max_images_per_class=config_values.get("max_images_per_class", args.max_images_per_class),
        plot=bool(config_values.get("plot", args.plot)),
    )


if __name__ == "__main__":
    final_result = train(parse_args())
    print("final_result=" + json.dumps(final_result, sort_keys=True))
