"""Convert DICOM images to PNG with CT rescale and window handling."""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pydicom
from tqdm import tqdm


LOGGER = logging.getLogger("dicom_to_png")


@dataclass(frozen=True)
class ConvertConfig:
    input_dir: Path = Path("data/raw_dicom")
    output_dir: Path = Path("data/processed_png")
    metadata_csv: Path | None = None
    recursive: bool = True
    overwrite: bool = False


def _first_numeric(value: object, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return float(value[0]) if value else default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def apply_rescale(pixel_array: np.ndarray, dataset: pydicom.Dataset) -> np.ndarray:
    slope = _first_numeric(getattr(dataset, "RescaleSlope", 1.0), 1.0)
    intercept = _first_numeric(getattr(dataset, "RescaleIntercept", 0.0), 0.0)
    return pixel_array.astype(np.float32) * slope + intercept


def window_to_uint8(image: np.ndarray, dataset: pydicom.Dataset) -> np.ndarray:
    center = _first_numeric(getattr(dataset, "WindowCenter", None), float(np.mean(image)))
    width = _first_numeric(getattr(dataset, "WindowWidth", None), float(np.ptp(image)))
    if width <= 0:
        width = float(np.ptp(image)) or 1.0
    lower = center - width / 2
    upper = center + width / 2
    clipped = np.clip(image, lower, upper)
    normalized = (clipped - lower) / max(upper - lower, 1.0)
    return (normalized * 255).astype(np.uint8)


def discover_dicom_files(input_dir: Path, recursive: bool = True) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(path for path in input_dir.glob(pattern) if path.is_file() and path.suffix.lower() == ".dcm")


def output_path_for(dicom_path: Path, input_dir: Path, output_dir: Path) -> Path:
    relative_path = dicom_path.relative_to(input_dir)
    return output_dir / relative_path.with_suffix(".png")


def convert_one(dicom_path: Path, output_path: Path, overwrite: bool = False) -> dict[str, object]:
    if output_path.exists() and not overwrite:
        return {"input": str(dicom_path), "output": str(output_path), "status": "skipped"}

    try:
        dataset = pydicom.dcmread(dicom_path)
        image = apply_rescale(dataset.pixel_array, dataset)
        png = window_to_uint8(image, dataset)
    except Exception as exc:
        LOGGER.warning("Skipping invalid DICOM %s: %s", dicom_path, exc)
        return {"input": str(dicom_path), "output": str(output_path), "status": "failed", "error": str(exc)}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), png):
        return {"input": str(dicom_path), "output": str(output_path), "status": "failed", "error": "cv2.imwrite failed"}

    return {
        "input": str(dicom_path),
        "output": str(output_path),
        "status": "converted",
        "rows": int(png.shape[0]),
        "columns": int(png.shape[1]),
    }


def parse_args() -> ConvertConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ConvertConfig.input_dir)
    parser.add_argument("--output-dir", type=Path, default=ConvertConfig.output_dir)
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    return ConvertConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        metadata_csv=args.metadata_csv,
        recursive=not args.no_recursive,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = parse_args()
    LOGGER.info("Found %s DICOM files", len(discover_dicom_files(config.input_dir, config.recursive)))
