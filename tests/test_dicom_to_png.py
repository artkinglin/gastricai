from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dicom_to_png import apply_rescale, output_path_for, safe_metadata, window_to_uint8


def test_output_path_preserves_nested_structure() -> None:
    output_path = output_path_for(
        Path("raw/study_1/series_2/image.dcm"),
        Path("raw"),
        Path("png"),
    )

    assert output_path == Path("png/study_1/series_2/image.png")


def test_apply_rescale_uses_slope_and_intercept() -> None:
    dataset = SimpleNamespace(RescaleSlope=2, RescaleIntercept=-1000)
    image = apply_rescale(np.array([[0, 10]], dtype=np.int16), dataset)

    assert image.tolist() == [[-1000.0, -980.0]]


def test_window_to_uint8_clips_to_window() -> None:
    dataset = SimpleNamespace(WindowCenter=0, WindowWidth=100)
    image = np.array([[-100, 0, 100]], dtype=np.float32)
    png = window_to_uint8(image, dataset)

    assert png.tolist() == [[0, 127, 255]]


def test_safe_metadata_excludes_patient_identity() -> None:
    dataset = SimpleNamespace(PatientName="Jane Doe", PatientID="123", Modality="CT", Rows=2, Columns=3)

    metadata = safe_metadata(dataset)

    assert metadata["modality"] == "CT"
    assert metadata["rows"] == 2
    assert "PatientName" not in metadata
    assert "PatientID" not in metadata
