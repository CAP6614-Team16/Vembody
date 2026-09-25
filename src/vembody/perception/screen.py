"""Safe placeholder for future live screen capture."""

from __future__ import annotations

from ..observations import Observation
from ..results import StepResult
from ..tasks import TaskSpec


class ScreenObservationProvider:
    """Declare the live capture boundary without enabling it yet."""

    def __init__(self, region: tuple[int, int, int, int]) -> None:
        self.region = region

    def capture(
        self,
        task: TaskSpec,
        episode_id: str,
        step: int,
        history: list[StepResult],
    ) -> Observation:
        """Reject live capture until its platform safety checks exist."""
        raise NotImplementedError(
            "Live screen capture is deferred until the safe desktop milestone"
        )
