from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


def load_settings(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def section(settings: dict, name: str) -> dict:
    value = settings.get(name, {})
    if not isinstance(value, dict):
        raise TypeError(f"configuration section {name!r} must be a table")
    return value
