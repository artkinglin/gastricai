"""Lightweight CT threshold segmentation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SegmentationResult:
    pixel_count: int
    area_mm2: float | None


def binary_mask(image: np.ndarray, threshold: int) -> np.ndarray:
    return image > threshold


def clean_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    if kernel_size <= 1:
        return mask.astype(bool)
    try:
        import cv2

        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        return closed.astype(bool)
    except Exception:
        return mask.astype(bool)


def largest_component(mask: np.ndarray) -> np.ndarray:
    try:
        import cv2

        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if count <= 1:
            return mask.astype(bool)
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return labels == largest_label
    except Exception:
        return mask.astype(bool)


def segment_threshold_area(
    image: np.ndarray,
    threshold: int,
    pixel_spacing: tuple[float, float] | None = None,
    kernel_size: int = 3,
    keep_largest: bool = True,
) -> SegmentationResult:
    mask = clean_mask(binary_mask(image, threshold), kernel_size=kernel_size)
    if keep_largest:
        mask = largest_component(mask)
    pixel_count = int(np.count_nonzero(mask))
    area_mm2 = None
    if pixel_spacing is not None:
        area_mm2 = pixel_count * float(pixel_spacing[0]) * float(pixel_spacing[1])
    return SegmentationResult(pixel_count=pixel_count, area_mm2=area_mm2)
