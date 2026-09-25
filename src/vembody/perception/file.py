"""Observation provider for saved screenshots."""

from __future__ import annotations

from pathlib import Path

from ..observations import Observation
from ..results import StepResult
from ..tasks import TaskSpec


class FileObservationProvider:
    """Reuse one configured screenshot for deterministic development."""

    def __init__(self, screenshot_path: str | None = None) -> None:
        self.screenshot_path = screenshot_path

    def validate_image(self, path: str | None) -> None:
        """Confirm a configured screenshot path exists when supplied."""
        if path is not None and not Path(path).is_file():
            raise FileNotFoundError(f"Screenshot does not exist: {path}")

    def capture(
        self,
        task: TaskSpec,
        episode_id: str,
        step: int,
        history: list[StepResult],
    ) -> Observation:
        """Return an observation referencing the configured screenshot."""
        self.validate_image(self.screenshot_path)
        return Observation(
            episode_id=episode_id,
            step=step,
            task_id=task.id,
            screenshot_path=self.screenshot_path,
            metadata={"provider": "file"},
        )
