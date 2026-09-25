"""Deterministic model used for Milestone 1."""

from __future__ import annotations

import json
from time import perf_counter

from ..actions import action_from_dict, action_to_dict
from ..observations import Observation
from ..results import StepResult
from ..tasks import TaskSpec
from .base import ModelResponse


class MockModel:
    """Return a predefined action sequence without model inference."""

    def __init__(self, action_payloads: list[dict] | None = None) -> None:
        self._action_payloads = action_payloads or [{"type": "finish"}]
        self._index = 0

    @classmethod
    def from_task(cls, task: TaskSpec) -> MockModel:
        """Create a mock model from task-provided action payloads."""
        return cls(list(task.mock_actions))

    def reset(self) -> None:
        """Restart the deterministic action sequence."""
        self._index = 0

    def generate_action(
        self,
        task: TaskSpec,
        observation: Observation,
        history: list[StepResult],
    ) -> ModelResponse:
        """Return the next action as strict JSON."""
        started = perf_counter()
        payload = self._action_payloads[min(self._index, len(self._action_payloads) - 1)]
        self._index += 1

        action = action_from_dict(payload)
        response = {
            "reasoning_summary": "Deterministic mock response.",
            "action": action_to_dict(action),
            "expected_result": "The configured mock action is recorded.",
            "confidence": 1.0,
        }
        return ModelResponse(
            raw_text=json.dumps(response),
            elapsed_seconds=perf_counter() - started,
            metadata={"provider": "mock"},
        )
