"""Interface for observation providers."""

from __future__ import annotations

from typing import Protocol

from ..observations import Observation
from ..results import StepResult
from ..tasks import TaskSpec


class ObservationProvider(Protocol):
    """Contract implemented by file and live capture providers."""

    def capture(
        self,
        task: TaskSpec,
        episode_id: str,
        step: int,
        history: list[StepResult],
    ) -> Observation:
        """Capture one observation for the current step."""
        ...
