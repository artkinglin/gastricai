"""Run inference with a trained EfficientNetV2-B0 gastric classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from efficientnetv2b0_120 import GastricImageDataset, build_model, build_transforms, predict_probabilities
from gastric_common import CLASS_NAMES, IMAGE_EXTENSIONS, write_prediction_csv


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {input_path}")
        return [input_path]
    if input_path.is_dir():
        image_paths = sorted(path for path in input_path.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
        if not image_paths:
            raise FileNotFoundError(f"No supported images found under: {input_path}")
        return image_paths
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    threshold = float(checkpoint.get("threshold", 0.5))
    return model, threshold


def run_inference(checkpoint_path: Path, input_path: Path, output_csv: Path, batch_size: int) -> dict[str, object]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_paths = collect_images(input_path)
    labels = [0] * len(image_paths)
    loader = DataLoader(
        GastricImageDataset(image_paths, labels, build_transforms(train=False)),
        batch_size=batch_size,
        shuffle=False,
    )
    model, threshold = load_checkpoint(checkpoint_path, device)
    probabilities, _ = predict_probabilities(model, loader, device)
    write_prediction_csv(output_csv, image_paths, labels, probabilities, threshold, CLASS_NAMES)
    return {
        "checkpoint": str(checkpoint_path),
        "input": str(input_path),
        "output_csv": str(output_csv),
        "images": len(image_paths),
        "threshold": threshold,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/efficientnetv2b0_120/best_model.pt"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=Path("runs/efficientnetv2b0_120/inference_predictions.csv"))
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_inference(args.checkpoint, args.input, args.output_csv, args.batch_size)
    print(json.dumps(result, sort_keys=True))
