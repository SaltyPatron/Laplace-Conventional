from __future__ import annotations

import tomllib
from pathlib import Path


def load_settings(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def section(settings: dict, name: str) -> dict:
    value = settings.get(name, {})
    if not isinstance(value, dict):
        raise TypeError(f"configuration section {name!r} must be a table")
    return value
