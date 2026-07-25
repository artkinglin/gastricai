from pathlib import Path

from gastric_common import (
    compute_metrics,
    confusion_counts,
    discover_image_paths,
    read_manifest,
    stratified_split,
    threshold_sweep,
    tune_threshold,
)


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


def test_metrics_respect_threshold() -> None:
    metrics = compute_metrics([0, 0, 1, 1], [0.1, 0.4, 0.8, 0.9], threshold=0.5)

    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_confusion_counts_include_all_cells() -> None:
    counts = confusion_counts([0, 0, 1, 1], [0.2, 0.7, 0.4, 0.8], threshold=0.5)

    assert counts == {
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_positive": 1,
    }


def test_threshold_tuning_prefers_best_f1() -> None:
    threshold, metrics = tune_threshold([0, 0, 1, 1], [0.1, 0.2, 0.7, 0.8])

    assert threshold >= 0.5
    assert metrics["f1"] == 1.0


def test_manifest_reader_accepts_named_labels(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_path,label\ncase_001.png,benign\ncase_002.png,malignant\n", encoding="utf-8")

    image_paths, labels = read_manifest(manifest, root_dir=tmp_path)

    assert image_paths == [tmp_path / "case_001.png", tmp_path / "case_002.png"]
    assert labels == [0, 1]


def test_threshold_sweep_includes_endpoints() -> None:
    rows = threshold_sweep([0, 1], [0.25, 0.75], steps=3)

    assert [row["threshold"] for row in rows] == [0.0, 0.5, 1.0]
