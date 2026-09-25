"""Runtime configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RunConfig:
    """Settings selected for one command invocation."""

    model: str = "mock"
    executor: str = "dry-run"
    log_path: str = "artifacts/episode.jsonl"


def load_config(path: str | Path) -> dict[str, Any]:
    """Load optional YAML runtime settings."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Runtime configuration must be an object")
    return data
