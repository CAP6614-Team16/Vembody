"""Explicitly disabled live desktop executor boundary."""

from __future__ import annotations

from ..actions import Action
from ..errors import ExecutionError
from ..results import ExecutionResult


class DesktopExecutor:
    """Reject live actions until platform and window checks are implemented."""

    def __init__(self, allow_live_actions: bool = False) -> None:
        self.allow_live_actions = allow_live_actions

    def execute(self, action: Action) -> ExecutionResult:
        """Prevent live input in the initial scaffold."""
        raise ExecutionError("Live desktop execution is not implemented in Milestone 1")

    def release_all_inputs(self) -> None:
        """Provide the future emergency-cleanup boundary."""
        return None

    def emergency_stop(self) -> None:
        """Provide the future emergency-stop boundary."""
        self.release_all_inputs()

    def shutdown(self) -> None:
        """Clean up the future live executor."""
        self.release_all_inputs()
