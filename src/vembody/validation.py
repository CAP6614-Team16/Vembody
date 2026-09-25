"""Allowlist and safety validation for proposed actions."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .actions import Action, ActionType
from .errors import ActionValidationError
from .tasks import TaskSpec


@dataclass(frozen=True)
class ActionPolicy:
    """Limits applied before an action reaches an executor."""

    allowed_actions: frozenset[ActionType]
    allowed_keys: frozenset[str] = frozenset()
    max_text_length: int = 256
    max_wait_seconds: float = 10.0

    @classmethod
    def from_task(cls, task: TaskSpec) -> ActionPolicy:
        """Build a policy from task action permissions."""
        return cls(frozenset(task.allowed_actions))


def _check_coordinate(value: float | None, field_name: str) -> None:
    """Require a finite normalized coordinate."""
    if value is None or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ActionValidationError(f"{field_name} must be within [0, 1]")


def validate_action(action: Action, policy: ActionPolicy) -> None:
    """Reject actions outside the configured allowlist and bounds."""
    if action.type not in policy.allowed_actions:
        raise ActionValidationError(f"Action type is not allowed: {action.type.value}")

    match action.type:
        case ActionType.MOVE | ActionType.CLICK | ActionType.DOUBLE_CLICK | ActionType.RIGHT_CLICK:
            _check_coordinate(action.x, "x")
            _check_coordinate(action.y, "y")

        case ActionType.DRAG:
            for field_name in ("x", "y", "x2", "y2"):
                _check_coordinate(getattr(action, field_name), field_name)

        case ActionType.KEY:
            if action.key is None or action.key not in policy.allowed_keys:
                raise ActionValidationError("Key is not allowed")

        case ActionType.HOTKEY:
            if not action.keys or not set(action.keys).issubset(policy.allowed_keys):
                raise ActionValidationError("Hotkey contains a disallowed key")

        case ActionType.TYPE_TEXT:
            if action.text is None or len(action.text) > policy.max_text_length:
                raise ActionValidationError("Text is missing or exceeds the configured limit")

        case ActionType.WAIT:
            if action.seconds is None or not 0 <= action.seconds <= policy.max_wait_seconds:
                raise ActionValidationError("Wait duration exceeds the configured limit")
