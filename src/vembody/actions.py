"""Typed, serializable actions proposed by the model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .errors import ModelParseError


class ActionType(str, Enum):
    """Actions that the runtime may consider for execution."""

    MOVE = "move"
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    DRAG = "drag"
    KEY = "key"
    HOTKEY = "hotkey"
    TYPE_TEXT = "type_text"
    SCROLL = "scroll"
    WAIT = "wait"
    FINISH = "finish"
    FAIL = "fail"


@dataclass(frozen=True)
class Action:
    """A single allowlisted action with optional typed parameters."""

    type: ActionType
    x: float | None = None
    y: float | None = None
    x2: float | None = None
    y2: float | None = None
    key: str | None = None
    keys: tuple[str, ...] = ()
    text: str | None = None
    amount: int | None = None
    seconds: float | None = None


def _require_exact_fields(data: dict[str, Any], allowed: set[str]) -> None:
    """Reject fields that are not part of the action contract."""
    unknown = set(data) - allowed
    if unknown:
        raise ModelParseError(f"Unknown action field(s): {sorted(unknown)}")


def _number(value: Any, field: str) -> float:
    """Read a finite numeric action field."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelParseError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ModelParseError(f"{field} must be finite")
    return number


def action_from_dict(data: Any) -> Action:
    """Convert a strict action dictionary into an Action."""
    if not isinstance(data, dict) or "type" not in data:
        raise ModelParseError("action must be an object containing type")

    try:
        action_type = ActionType(data["type"])
    except (ValueError, TypeError) as exc:
        raise ModelParseError("action type is unknown") from exc

    match action_type:
        case ActionType.CLICK | ActionType.DOUBLE_CLICK | ActionType.RIGHT_CLICK | ActionType.MOVE:
            _require_exact_fields(data, {"type", "x", "y"})
            return Action(
                action_type,
                x=_number(data["x"], "x"),
                y=_number(data["y"], "y"),
            )

        case ActionType.DRAG:
            _require_exact_fields(data, {"type", "x", "y", "x2", "y2"})
            return Action(
                action_type,
                x=_number(data["x"], "x"),
                y=_number(data["y"], "y"),
                x2=_number(data["x2"], "x2"),
                y2=_number(data["y2"], "y2"),
            )

        case ActionType.KEY:
            _require_exact_fields(data, {"type", "key"})
            if not isinstance(data.get("key"), str):
                raise ModelParseError("key must be a string")
            return Action(action_type, key=data["key"])

        case ActionType.HOTKEY:
            _require_exact_fields(data, {"type", "keys"})
            keys = data.get("keys")
            if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
                raise ModelParseError("keys must be a list of strings")
            return Action(action_type, keys=tuple(keys))

        case ActionType.TYPE_TEXT:
            _require_exact_fields(data, {"type", "text"})
            if not isinstance(data.get("text"), str):
                raise ModelParseError("text must be a string")
            return Action(action_type, text=data["text"])

        case ActionType.SCROLL:
            _require_exact_fields(data, {"type", "amount"})
            amount = data.get("amount")
            if isinstance(amount, bool) or not isinstance(amount, int):
                raise ModelParseError("amount must be an integer")
            return Action(action_type, amount=amount)

        case ActionType.WAIT:
            _require_exact_fields(data, {"type", "seconds"})
            return Action(action_type, seconds=_number(data["seconds"], "seconds"))

        case ActionType.FINISH | ActionType.FAIL:
            _require_exact_fields(data, {"type"})
            return Action(action_type)


def action_to_dict(action: Action) -> dict[str, Any]:
    """Convert an action into JSON-compatible data."""
    data: dict[str, Any] = {"type": action.type.value}
    for field in ("x", "y", "x2", "y2", "key", "text", "amount", "seconds"):
        value = getattr(action, field)
        if value is not None:
            data[field] = value
    if action.keys:
        data["keys"] = list(action.keys)
    return data
