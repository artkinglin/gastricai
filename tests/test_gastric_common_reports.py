import json
from pathlib import Path

from gastric_common import write_history_csv, write_json, write_prediction_csv


def test_write_json_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "report.json"

    write_json(path, {"ok": True})

    assert json.loads(path.read_text(encoding="utf-8")) == {"ok": True}


def test_write_history_csv_writes_union_header(tmp_path: Path) -> None:
    path = tmp_path / "history.csv"

    write_history_csv(path, [{"epoch": 1.0, "f1": 0.5}, {"epoch": 2.0, "recall": 1.0}])

    assert path.read_text(encoding="utf-8").splitlines()[0] == "epoch,f1,recall"


def test_write_prediction_csv_maps_class_names(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"

    write_prediction_csv(path, [Path("case.png")], [1], [0.8], threshold=0.5)

    text = path.read_text(encoding="utf-8")
    assert "case.png,malignant,0.8,malignant" in text
