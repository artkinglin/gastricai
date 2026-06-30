"""Shared pure-Python helpers for gastric image workflows."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Iterable


CLASS_NAMES = ("benign", "malignant")
LABEL_ALIASES = {
    "benign": ("benign", "normal", "negative"),
    "malignant": ("malignant", "abnormal", "tumor", "positive"),
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def candidate_label_dirs(data_dir: Path, aliases: Iterable[str]) -> list[Path]:
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
        class_dirs = candidate_label_dirs(data_dir, LABEL_ALIASES[class_name])
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


def stratified_split(
    image_paths: list[Path],
    labels: list[int],
    validation_size: float,
    seed: int,
) -> tuple[list[Path], list[Path], list[int], list[int]]:
    if len(image_paths) != len(labels):
        raise ValueError("image_paths and labels must have the same length")
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1")

    rng = random.Random(seed)
    by_label: dict[int, list[Path]] = {}
    for image_path, label in zip(image_paths, labels):
        by_label.setdefault(label, []).append(image_path)

    train_paths: list[Path] = []
    val_paths: list[Path] = []
    train_labels: list[int] = []
    val_labels: list[int] = []

    for label, paths in sorted(by_label.items()):
        shuffled_paths = paths[:]
        rng.shuffle(shuffled_paths)
        val_count = max(1, round(len(shuffled_paths) * validation_size))
        if len(shuffled_paths) - val_count < 1:
            raise ValueError(f"Class {label} needs at least two samples for a stratified split")
        val_group = shuffled_paths[:val_count]
        train_group = shuffled_paths[val_count:]
        val_paths.extend(val_group)
        val_labels.extend([label] * len(val_group))
        train_paths.extend(train_group)
        train_labels.extend([label] * len(train_group))

    return train_paths, val_paths, train_labels, val_labels


def confusion_counts(labels: list[int], probabilities: list[float], threshold: float = 0.5) -> dict[str, int]:
    predictions = [1 if probability >= threshold else 0 for probability in probabilities]
    return {
        "true_negative": sum(1 for label, prediction in zip(labels, predictions) if label == 0 and prediction == 0),
        "false_positive": sum(1 for label, prediction in zip(labels, predictions) if label == 0 and prediction == 1),
        "false_negative": sum(1 for label, prediction in zip(labels, predictions) if label == 1 and prediction == 0),
        "true_positive": sum(1 for label, prediction in zip(labels, predictions) if label == 1 and prediction == 1),
    }


def roc_auc_score(labels: list[int], probabilities: list[float]) -> float:
    positives = [score for label, score in zip(labels, probabilities) if label == 1]
    negatives = [score for label, score in zip(labels, probabilities) if label == 0]
    if not positives or not negatives:
        return float("nan")

    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def compute_metrics(labels: list[int], probabilities: list[float], threshold: float = 0.5) -> dict[str, float]:
    counts = confusion_counts(labels, probabilities, threshold)
    true_positive = counts["true_positive"]
    true_negative = counts["true_negative"]
    false_positive = counts["false_positive"]
    false_negative = counts["false_negative"]
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
        "roc_auc": roc_auc_score(labels, probabilities),
    }


def candidate_thresholds(probabilities: list[float]) -> list[float]:
    thresholds = {0.5}
    thresholds.update(max(0.0, min(1.0, probability)) for probability in probabilities)
    return sorted(thresholds)


def tune_threshold(labels: list[int], probabilities: list[float]) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics = compute_metrics(labels, probabilities, threshold=best_threshold)
    for threshold in candidate_thresholds(probabilities):
        metrics = compute_metrics(labels, probabilities, threshold=threshold)
        if (metrics["f1"], metrics["recall"], metrics["precision"]) > (
            best_metrics["f1"],
            best_metrics["recall"],
            best_metrics["precision"],
        ):
            best_threshold = threshold
            best_metrics = metrics
    return best_threshold, best_metrics


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_history_csv(path: Path, history: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in history for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
