"""Interface for independent task evaluators."""

from __future__ import annotations

from typing import Protocol

from ..observations import Observation
from ..results import EvaluationResult, EvaluationStatus
from ..tasks import TaskSpec


class SuccessEvaluator(Protocol):
    """Contract for determining external task success."""

    def evaluate(
        self,
        task: TaskSpec,
        observation: Observation | None,
    ) -> EvaluationResult:
        """Evaluate the current environment state independently of the model."""
        ...


class NullEvaluator:
    """Return unknown when no external success checker exists."""

    def evaluate(
        self,
        task: TaskSpec,
        observation: Observation | None,
    ) -> EvaluationResult:
        """Record that finish was requested but not externally verified."""
        return EvaluationResult(
            status=EvaluationStatus.UNKNOWN,
            message="No external task evaluator is configured.",
        )
