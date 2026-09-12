"""Train and evaluate EfficientNetV2-B0 pipelines on gastric histology images."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import json
import logging
import random
from typing import Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset

from experiment_config import coerce_path, get_config_value, load_config_file
from gastric_common import (
    CLASS_NAMES,
    calibration_bins,
    compute_metrics,
    confusion_counts,
    discover_image_paths,
    log_event,
    read_manifest,
    save_confusion_matrix_image,
    stratified_split,
    threshold_sweep,
    tune_threshold,
    write_history_csv,
    write_calibration_csv,
    write_json,
    write_prediction_csv,
    write_threshold_sweep_csv,
)


LOGGER = logging.getLogger("efficientnetv2b0_120")
IMAGE_SIZE = 120
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TrainConfig:
    mode: str = "hybrid"
    data_dir: Path = Path("data/GasHisSDB/120")
    manifest: Path | None = None
    test_dir: Path | None = None
    test_manifest: Path | None = None
    output_dir: Path = Path("runs/efficientnetv2b0_120")
    run_name: str | None = None
    resume_checkpoint: Path | None = None
    batch_size: int = 32
    epochs: int = 25
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    validation_size: float = 0.2
    seed: int = 42
    num_workers: int = 2
    early_stopping_patience: int = 5
    pretrained: bool = True
    catboost_iterations: int = 500
    catboost_learning_rate: float = 0.03
    catboost_depth: int = 6
    tsne_perplexity: float = 30.0
    grad_cam_samples: int = 8
    checkpoint: Path | None = None


def validate_config(config: TrainConfig) -> None:
    if config.mode not in {"hybrid", "finetune"}:
        raise ValueError("mode must be 'hybrid' or 'finetune'")
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
    if config.catboost_iterations < 1:
        raise ValueError("catboost_iterations must be at least 1")
    if config.catboost_learning_rate <= 0:
        raise ValueError("catboost_learning_rate must be positive")
    if config.catboost_depth < 1:
        raise ValueError("catboost_depth must be at least 1")
    if config.tsne_perplexity <= 0:
        raise ValueError("tsne_perplexity must be positive")
    if config.grad_cam_samples < 0:
        raise ValueError("grad_cam_samples cannot be negative")


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


def build_transforms(train: bool):
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise ImportError(
            "torchvision is required to build image transforms. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

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
    try:
        from torchvision import models
    except ImportError as exc:
        raise ImportError(
            "torchvision is required to build EfficientNetV2-B0. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    weights = models.EfficientNet_V2_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_v2_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, len(CLASS_NAMES)),
    )
    return model


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return model


def forward_features(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    x = model.features(images)
    x = model.avgpool(x)
    return torch.flatten(x, 1)


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
    transform,
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


def compute_metrics_with_confusion(
    labels: Sequence[int],
    probabilities: Sequence[float],
    threshold: float = 0.5,
) -> dict[str, object]:
    metrics = compute_metrics(list(labels), list(probabilities), threshold=threshold)
    metrics["confusion_matrix"] = confusion_counts(labels, probabilities, threshold=threshold)
    return metrics


@torch.no_grad()
def extract_deep_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    model.eval()
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    paths: list[Path] = []

    dataset = loader.dataset
    dataset_paths = getattr(dataset, "image_paths", None)
    offset = 0

    for images, targets in loader:
        images = images.to(device)
        batch_features = forward_features(model, images)
        features.append(batch_features.cpu().numpy())
        labels.append(targets.numpy())
        if dataset_paths is not None:
            paths.extend(dataset_paths[offset : offset + len(targets)])
        offset += len(targets)

    return np.vstack(features), np.concatenate(labels), paths


def train_catboost_classifier(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    config: TrainConfig,
):
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ImportError(
            "CatBoost is required for the hybrid pipeline. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    classifier = CatBoostClassifier(
        iterations=config.catboost_iterations,
        learning_rate=config.catboost_learning_rate,
        depth=config.catboost_depth,
        loss_function="Logloss",
        eval_metric="F1",
        random_seed=config.seed,
        verbose=False,
        allow_writing_files=False,
    )
    classifier.fit(train_features, train_labels)
    return classifier


def save_tsne_plot(
    features: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    perplexity: float,
    seed: int,
) -> None:
    import matplotlib.pyplot as plt

    if len(features) < 3:
        return

    effective_perplexity = min(perplexity, max(1.0, (len(features) - 1) / 3))
    embeddings = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(features)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    for class_index, class_name in enumerate(CLASS_NAMES):
        mask = labels == class_index
        plt.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            s=18,
            alpha=0.78,
            label=class_name,
        )
    plt.title("t-SNE of EfficientNetV2-B0 Features")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def denormalize_image(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN, dtype=image.dtype, device=image.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=image.dtype, device=image.device).view(3, 1, 1)
    image = image * std + mean
    image = image.clamp(0, 1).permute(1, 2, 0)
    return image.cpu().numpy()


def make_grad_cam_overlay(
    model: nn.Module,
    image: torch.Tensor,
    target_class: int,
    device: torch.device,
) -> np.ndarray:
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    target_layer = model.features[-1]

    def save_activation(_module, _inputs, output):
        activations.append(output)

    def save_gradient(_module, _grad_inputs, grad_outputs):
        gradients.append(grad_outputs[0])

    forward_handle = target_layer.register_forward_hook(save_activation)
    backward_handle = target_layer.register_full_backward_hook(save_gradient)

    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        batch = image.unsqueeze(0).to(device)
        logits = model(batch)
        score = logits[:, target_class].sum()
        score.backward()

        weights = gradients[-1].mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations[-1]).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze()
        cam = (cam - cam.min()) / (cam.max() - cam.min()).clamp_min(1e-8)
    finally:
        forward_handle.remove()
        backward_handle.remove()

    base = denormalize_image(image)
    heat = cam.detach().cpu().numpy()
    heat_rgb = np.zeros_like(base)
    heat_rgb[..., 0] = heat
    heat_rgb[..., 1] = np.clip(1.0 - np.abs(heat - 0.5) * 2.0, 0.0, 1.0)
    return np.clip((0.58 * base) + (0.42 * heat_rgb), 0.0, 1.0)


def save_grad_cam_examples(
    model: nn.Module,
    dataset: GastricImageDataset,
    output_dir: Path,
    device: torch.device,
    sample_count: int,
) -> None:
    if sample_count <= 0:
        return

    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(min(sample_count, len(dataset))):
        image, label = dataset[index]
        overlay = make_grad_cam_overlay(model, image, int(label.item()), device)
        image_name = dataset.image_paths[index].stem
        output_path = output_dir / f"{index:03d}_{image_name}_{CLASS_NAMES[int(label.item())]}.png"
        plt.imsave(output_path, overlay)


def train(config: TrainConfig) -> dict[str, float]:
    validate_config(config)
    seed_everything(config.seed)
    output_dir = config.output_dir / config.run_name if config.run_name else config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_paths, labels = (
        read_manifest(config.manifest, root_dir=config.data_dir) if config.manifest else discover_image_paths(config.data_dir)
    )
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
    if config.resume_checkpoint is not None:
        checkpoint = torch.load(config.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
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
    checkpoint_path = output_dir / "best_model.pt"

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
        log_event(LOGGER, "efficientnet_epoch", **metrics)
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

    write_json(output_dir / "history.json", history)
    write_history_csv(output_dir / "history.csv", history)
    write_calibration_csv(output_dir / "validation_calibration.csv", calibration_bins(val_result["labels"], val_result["probabilities"]))
    write_threshold_sweep_csv(output_dir / "validation_threshold_sweep.csv", threshold_sweep(val_result["labels"], val_result["probabilities"]))
    if config.test_dir is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if config.test_manifest is not None:
            test_paths, test_labels = read_manifest(config.test_manifest, root_dir=config.test_dir)
        else:
            test_paths, test_labels = discover_image_paths(config.test_dir)
        test_loader = make_loader(
            test_paths, test_labels, config.batch_size, build_transforms(train=False), False, config.num_workers
        )
        threshold = float(checkpoint.get("threshold", 0.5))
        test_result = evaluate(model, test_loader, device, threshold=threshold)
        write_json(output_dir / "test_metrics.json", test_result["metrics"])
        write_json(output_dir / "test_confusion_matrix.json", test_result["confusion_matrix"])
        save_confusion_matrix_image(output_dir / "test_confusion_matrix.png", test_result["confusion_matrix"])
        write_prediction_csv(
            output_dir / "test_predictions.csv",
            test_paths,
            test_result["labels"],
            test_result["probabilities"],
            threshold,
        )
        write_threshold_sweep_csv(
            output_dir / "test_threshold_sweep.csv",
            threshold_sweep(test_result["labels"], test_result["probabilities"]),
        )
        write_calibration_csv(
            output_dir / "test_calibration.csv",
            calibration_bins(test_result["labels"], test_result["probabilities"]),
        )
    return best_metrics


def run_hybrid_pipeline(config: TrainConfig) -> dict[str, object]:
    validate_config(config)
    seed_everything(config.seed)
    output_dir = config.output_dir / config.run_name if config.run_name else config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_paths, labels = (
        read_manifest(config.manifest, root_dir=config.data_dir) if config.manifest else discover_image_paths(config.data_dir)
    )
    train_paths, val_paths, train_labels, val_labels = stratified_split(
        image_paths,
        labels,
        config.validation_size,
        config.seed,
    )

    train_dataset = GastricImageDataset(train_paths, train_labels, build_transforms(train=False))
    val_dataset = GastricImageDataset(val_paths, val_labels, build_transforms(train=False))
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
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
    checkpoint_path = config.checkpoint or config.resume_checkpoint
    if checkpoint_path is not None:
        model = load_checkpoint(model, checkpoint_path, device)
    elif config.grad_cam_samples > 0:
        print(
            "warning=Grad-CAM examples use the EfficientNet classifier head. "
            "Pass --checkpoint from a fine-tuned EfficientNet run for class-specific heatmaps."
        )

    train_features, train_targets, train_feature_paths = extract_deep_features(model, train_loader, device)
    val_features, val_targets, val_feature_paths = extract_deep_features(model, val_loader, device)

    np.save(output_dir / "train_features.npy", train_features)
    np.save(output_dir / "train_labels.npy", train_targets)
    np.save(output_dir / "val_features.npy", val_features)
    np.save(output_dir / "val_labels.npy", val_targets)
    (output_dir / "train_paths.txt").write_text(
        "\n".join(str(path) for path in train_feature_paths),
        encoding="utf-8",
    )
    (output_dir / "val_paths.txt").write_text(
        "\n".join(str(path) for path in val_feature_paths),
        encoding="utf-8",
    )

    classifier = train_catboost_classifier(train_features, train_targets, config)
    probabilities = classifier.predict_proba(val_features)[:, 1].tolist()
    metrics = compute_metrics_with_confusion(val_targets.tolist(), probabilities)
    classifier.save_model(str(output_dir / "catboost_model.cbm"))

    all_features = np.vstack([train_features, val_features])
    all_labels = np.concatenate([train_targets, val_targets])
    save_tsne_plot(
        all_features,
        all_labels,
        output_dir / "tsne_features.png",
        config.tsne_perplexity,
        config.seed,
    )
    save_grad_cam_examples(
        model,
        val_dataset,
        output_dir / "grad_cam",
        device,
        config.grad_cam_samples,
    )

    artifact_summary = {
        "metrics": metrics,
        "class_names": CLASS_NAMES,
        "artifacts": {
            "catboost_model": str(output_dir / "catboost_model.cbm"),
            "tsne": str(output_dir / "tsne_features.png"),
            "grad_cam_dir": str(output_dir / "grad_cam"),
            "train_features": str(output_dir / "train_features.npy"),
            "val_features": str(output_dir / "val_features.npy"),
        },
    }
    write_json(output_dir / "hybrid_metrics.json", artifact_summary)
    log_event(LOGGER, "efficientnet_catboost_complete", **metrics)
    print(json.dumps(artifact_summary, sort_keys=True))
    return artifact_summary


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=("hybrid", "finetune"),
        default=TrainConfig.mode,
        help="Use `hybrid` for EfficientNetV2-B0 features + CatBoost, or `finetune` for end-to-end EfficientNet.",
    )
    parser.add_argument("--data-dir", type=Path, default=TrainConfig.data_dir)
    parser.add_argument("--manifest", type=Path, default=TrainConfig.manifest)
    parser.add_argument("--test-dir", type=Path, default=TrainConfig.test_dir)
    parser.add_argument("--test-manifest", type=Path, default=TrainConfig.test_manifest)
    parser.add_argument("--output-dir", type=Path, default=TrainConfig.output_dir)
    parser.add_argument("--run-name", type=str, default=TrainConfig.run_name)
    parser.add_argument("--resume-checkpoint", type=Path, default=TrainConfig.resume_checkpoint)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--validation-size", type=float, default=TrainConfig.validation_size)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--early-stopping-patience", type=int, default=TrainConfig.early_stopping_patience)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--catboost-iterations", type=int, default=TrainConfig.catboost_iterations)
    parser.add_argument("--catboost-learning-rate", type=float, default=TrainConfig.catboost_learning_rate)
    parser.add_argument("--catboost-depth", type=int, default=TrainConfig.catboost_depth)
    parser.add_argument("--tsne-perplexity", type=float, default=TrainConfig.tsne_perplexity)
    parser.add_argument("--grad-cam-samples", type=int, default=TrainConfig.grad_cam_samples)
    parser.add_argument("--checkpoint", type=Path, default=TrainConfig.checkpoint)
    args = parser.parse_args()
    config_values = load_config_file(args.config)
    return TrainConfig(
        mode=str(config_values.get("mode", args.mode)),
        data_dir=coerce_path(config_values.get("data_dir", args.data_dir)) or args.data_dir,
        manifest=coerce_path(config_values.get("manifest", args.manifest)),
        test_dir=coerce_path(get_config_value(config_values, "test_dir", args.test_dir)),
        test_manifest=coerce_path(config_values.get("test_manifest", args.test_manifest)),
        output_dir=coerce_path(config_values.get("output_dir", args.output_dir)) or args.output_dir,
        run_name=config_values.get("run_name", args.run_name),
        resume_checkpoint=coerce_path(config_values.get("resume_checkpoint", args.resume_checkpoint)),
        batch_size=int(config_values.get("batch_size", args.batch_size)),
        epochs=int(config_values.get("epochs", args.epochs)),
        learning_rate=float(config_values.get("learning_rate", args.learning_rate)),
        weight_decay=float(config_values.get("weight_decay", args.weight_decay)),
        validation_size=float(config_values.get("validation_size", args.validation_size)),
        seed=int(config_values.get("seed", args.seed)),
        num_workers=int(config_values.get("num_workers", args.num_workers)),
        early_stopping_patience=int(config_values.get("early_stopping_patience", args.early_stopping_patience)),
        pretrained=bool(config_values.get("pretrained", not args.no_pretrained)),
        catboost_iterations=int(config_values.get("catboost_iterations", args.catboost_iterations)),
        catboost_learning_rate=float(config_values.get("catboost_learning_rate", args.catboost_learning_rate)),
        catboost_depth=int(config_values.get("catboost_depth", args.catboost_depth)),
        tsne_perplexity=float(config_values.get("tsne_perplexity", args.tsne_perplexity)),
        grad_cam_samples=int(config_values.get("grad_cam_samples", args.grad_cam_samples)),
        checkpoint=coerce_path(config_values.get("checkpoint", args.checkpoint)),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = parse_args()
    if config.mode == "finetune":
        final_metrics = train(config)
        print("best_metrics=" + json.dumps(final_metrics, sort_keys=True))
    else:
        run_hybrid_pipeline(config)
