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


def discover_dicom_files(input_dir: Path, recursive: bool = True) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(path for path in input_dir.glob(pattern) if path.is_file() and path.suffix.lower() == ".dcm")


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
