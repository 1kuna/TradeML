"""Small helpers for TradeML dotenv-style files."""

from __future__ import annotations

import os
from pathlib import Path


def read_env_file(path: Path) -> dict[str, str]:
    """Read KEY=VALUE lines from a dotenv-style file."""
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key] = value
    return values


def write_env_file(path: Path, values: dict[str, str], *, sort_keys: bool = True) -> None:
    """Write a dotenv-style file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    items = sorted(values.items()) if sort_keys else values.items()
    path.write_text("\n".join(f"{key}={value}" for key, value in items) + "\n", encoding="utf-8")


def load_dotenv(env_path: Path | None, *, override: bool = False) -> None:
    """Load KEY=VALUE entries into process environment."""
    if env_path is None:
        return
    for key, value in read_env_file(env_path).items():
        if override:
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)
