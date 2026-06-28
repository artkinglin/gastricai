"""Shared pure-Python helpers for gastric image workflows."""

from __future__ import annotations

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
