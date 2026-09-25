"""JSON Lines logging for reproducible episodes."""

from __future__ import annotations

import json
from pathlib import Path

from ..results import EpisodeResult, StepResult, to_jsonable


class JsonlLogger:
    """Write one JSON object per step and one summary per episode."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, record: dict) -> None:
        """Append one serialized record to the log."""
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(to_jsonable(record), sort_keys=True) + "\n")

    def write_step(self, step_result: StepResult) -> None:
        """Append one step record."""
        self._write({"record_type": "step", "step": step_result})

    def write_episode(self, episode_result: EpisodeResult) -> None:
        """Append the final episode summary."""
        self._write({"record_type": "episode", "episode": episode_result})
