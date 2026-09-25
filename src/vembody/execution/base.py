"""Interface for action executors."""

from __future__ import annotations

from typing import Protocol

from ..actions import Action
from ..results import ExecutionResult


class ActionExecutor(Protocol):
    """Contract implemented by dry-run and live executors."""

    def execute(self, action: Action) -> ExecutionResult:
        """Execute one validated action."""
        ...

    def shutdown(self) -> None:
        """Release executor resources and held inputs."""
        ...
