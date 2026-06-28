from pathlib import Path

from gastric_common import discover_image_paths, stratified_split


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a real image")


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
