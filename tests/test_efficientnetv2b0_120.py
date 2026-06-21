from pathlib import Path

from PIL import Image

from efficientnetv2b0_120 import compute_metrics, discover_image_paths, stratified_split


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (120, 120), color=(128, 64, 32)).save(path)


def test_discovers_normal_and_abnormal_folders(tmp_path: Path) -> None:
    _write_image(tmp_path / "Normal" / "case_001.png")
    _write_image(tmp_path / "Abnormal" / "case_002.png")

    image_paths, labels = discover_image_paths(tmp_path)

    assert len(image_paths) == 2
    assert sorted(labels) == [0, 1]


def test_stratified_split_keeps_both_classes() -> None:
    image_paths = [Path(f"case_{index}.png") for index in range(10)]
    labels = [0] * 5 + [1] * 5

    _, _, train_labels, val_labels = stratified_split(image_paths, labels, 0.4, seed=7)

    assert set(train_labels) == {0, 1}
    assert set(val_labels) == {0, 1}


def test_metrics_respect_threshold() -> None:
    metrics = compute_metrics([0, 0, 1, 1], [0.1, 0.4, 0.8, 0.9], threshold=0.5)

    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
