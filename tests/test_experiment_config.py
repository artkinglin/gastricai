import json
from pathlib import Path

from experiment_config import coerce_path, load_config_file


def test_load_json_config(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"epochs": 3}), encoding="utf-8")

    assert load_config_file(path) == {"epochs": 3}


def test_missing_config_raises(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"

    try:
        load_config_file(missing)
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("missing config should raise")


def test_coerce_path_accepts_none() -> None:
    assert coerce_path(None) is None
    assert coerce_path("data/train") == Path("data/train")
