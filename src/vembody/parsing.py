"""Strict parsing of model responses into typed actions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .actions import Action, action_from_dict
from .errors import ModelParseError


@dataclass(frozen=True)
class ParsedResponse:
    """Validated response envelope surrounding one action."""

    action: Action
    reasoning_summary: str = ""
    expected_result: str | None = None
    confidence: float | None = None


def _parse_confidence(value: Any) -> float | None:
    """Validate and normalize optional model confidence."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ModelParseError("confidence must be between 0 and 1")
    return float(value)


def parse_model_response(raw_text: str) -> ParsedResponse:
    """Parse strict JSON without executing model-generated code."""
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ModelParseError("Model response is not valid JSON") from exc
    if not isinstance(data, dict):
        raise ModelParseError("Model response must be an object")
    allowed = {"reasoning_summary", "action", "expected_result", "confidence"}
    unknown = set(data) - allowed
    if unknown:
        raise ModelParseError(f"Unknown response field(s): {sorted(unknown)}")
    reasoning_summary = data.get("reasoning_summary", "")
    expected_result = data.get("expected_result")
    confidence = data.get("confidence")
    if not isinstance(reasoning_summary, str) or (
        expected_result is not None and not isinstance(expected_result, str)
    ):
        raise ModelParseError("Response summaries must be strings")
    if "action" not in data:
        raise ModelParseError("Model response must contain action")

    return ParsedResponse(
        action=action_from_dict(data["action"]),
        reasoning_summary=reasoning_summary,
        expected_result=expected_result,
        confidence=_parse_confidence(confidence),
    )
