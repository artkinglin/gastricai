import numpy as np

from ct_segmentation import binary_mask, segment_threshold_area


def test_binary_mask_thresholds_pixels() -> None:
    mask = binary_mask(np.array([[10, 200]], dtype=np.uint8), threshold=127)

    assert mask.tolist() == [[False, True]]


def test_segment_threshold_area_counts_pixels_without_morphology() -> None:
    image = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    result = segment_threshold_area(image, threshold=127, kernel_size=1, keep_largest=False)

    assert result.pixel_count == 2
    assert result.area_mm2 is None


def test_segment_threshold_area_uses_pixel_spacing() -> None:
    image = np.array([[255, 255]], dtype=np.uint8)
    result = segment_threshold_area(image, threshold=127, pixel_spacing=(0.5, 0.25), kernel_size=1)

    assert result.pixel_count == 2
    assert result.area_mm2 == 0.25
