"""Serializable observations supplied to model providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class Observation:
    """What the agent can see at one point in an episode."""

    episode_id: str
    step: int
    task_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    screenshot_path: str | None = None
    capture_bounds: tuple[int, int, int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def observation_to_dict(observation: Observation) -> dict[str, Any]:
    """Convert an observation into JSON-compatible data."""
    data: dict[str, Any] = {
        "episode_id": observation.episode_id,
        "step": observation.step,
        "task_id": observation.task_id,
        "timestamp": observation.timestamp,
        "screenshot_path": observation.screenshot_path,
        "capture_bounds": observation.capture_bounds,
        "metadata": observation.metadata,
    }
    if observation.capture_bounds is not None:
        data["capture_bounds"] = list(observation.capture_bounds)
    return data
