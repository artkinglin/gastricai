"""Experiment configuration file helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config_file(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported config extension: {path.suffix}")


def coerce_path(value: Any) -> Path | None:
    if value is None:
        return None
    return value if isinstance(value, Path) else Path(value)
