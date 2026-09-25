"""Executor that records actions without controlling the computer."""

from __future__ import annotations

from time import perf_counter

from ..actions import Action, action_to_dict
from ..results import ExecutionResult


class DryRunExecutor:
    """Record every action as a successful simulation."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def execute(self, action: Action) -> ExecutionResult:
        """Record an action without sending input."""
        started = perf_counter()
        self.events.append(action_to_dict(action))
        return ExecutionResult(True, "dry-run: action recorded", perf_counter() - started)

    def shutdown(self) -> None:
        """Finish the dry-run executor."""
        return None
