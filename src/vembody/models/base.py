"""Interface for model providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..observations import Observation
from ..results import StepResult
from ..tasks import TaskSpec


@dataclass(frozen=True)
class ModelResponse:
    """Raw model output and provider metadata."""

    raw_text: str
    elapsed_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class VisionModel(Protocol):
    """Contract implemented by every action-proposing model."""

    def generate_action(
        self,
        task: TaskSpec,
        observation: Observation,
        history: list[StepResult],
    ) -> ModelResponse:
        """Generate exactly one proposed action response."""
        ...
