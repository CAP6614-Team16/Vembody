"""Task specifications and YAML/JSON loading."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from .actions import ActionType
from .benchmark.spec import BenchmarkSpec, board_settings, parse_benchmark
from .errors import TaskLoadError


@dataclass(frozen=True)
class TaskSpec:
    """Configuration for one reproducible agent episode."""

    id: str
    name: str
    instruction: str
    environment: str
    reset_instructions: str
    success_description: str
    max_steps: int
    timeout_seconds: float
    allowed_actions: tuple[ActionType, ...]
    screenshot_path: str | None = None
    mock_actions: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    minesweeper: dict[str, int] | None = None
    benchmark: BenchmarkSpec | None = None


def _read_data(path: Path) -> dict[str, Any]:
    """Read a YAML or JSON mapping from disk."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle) if path.suffix.lower() == ".json" else yaml.safe_load(handle)
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise TaskLoadError(f"Could not read task file: {path}") from exc
    if not isinstance(data, dict):
        raise TaskLoadError("Task file must contain an object")
    return data


def _required_string(data: dict[str, Any], field_name: str) -> str:
    """Read a required non-empty string field."""
    value = data.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise TaskLoadError(f"Task field '{field_name}' must be a non-empty string")
    return value


def _parse_allowed_actions(data: dict[str, Any]) -> tuple[ActionType, ...]:
    """Convert configured action names into action types."""
    try:
        return tuple(ActionType(value) for value in data["allowed_actions"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TaskLoadError("allowed_actions must contain known action types") from exc


def _parse_mock_actions(data: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Validate deterministic mock actions supplied by a task."""
    mock_actions = data.get("mock_actions", [])
    if not isinstance(mock_actions, list) or not all(
        isinstance(item, dict) for item in mock_actions
    ):
        raise TaskLoadError("mock_actions must be a list of objects")
    return tuple(mock_actions)


def _parse_screenshot_path(data: dict[str, Any]) -> str | None:
    """Validate the optional screenshot path."""
    screenshot_path = data.get("screenshot")
    if screenshot_path is not None and not isinstance(screenshot_path, str):
        raise TaskLoadError("screenshot must be a path string or null")
    return screenshot_path


def load_task(path: str | Path) -> TaskSpec:
    """Load and validate one task file."""
    data = _read_data(Path(path))
    allowed_actions = _parse_allowed_actions(data)

    max_steps = data.get("max_steps")
    timeout_seconds = data.get("timeout_seconds")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise TaskLoadError("max_steps must be a positive integer")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or timeout_seconds <= 0
        or not math.isfinite(timeout_seconds)
    ):
        raise TaskLoadError("timeout_seconds must be positive")

    board = data.get("minesweeper")
    benchmark = None
    if "benchmark" in data:
        if data.get("environment") != "minesweeper":
            raise TaskLoadError("Benchmark requires the minesweeper environment")
        board = board_settings(board)
        benchmark = parse_benchmark(data["benchmark"], board)
        expected = (5, 5, 3) if benchmark.template_id == "T4-01" else (9, 9, 10)
        if (board["rows"], board["cols"], board["mines"]) != expected:
            raise TaskLoadError("Board settings do not match benchmark template")
        if not allowed_actions or not set(allowed_actions).issubset(
            {ActionType.CLICK, ActionType.RIGHT_CLICK, ActionType.FINISH}
        ):
            raise TaskLoadError("Unsupported benchmark actions")

    return TaskSpec(
        id=_required_string(data, "id"),
        name=_required_string(data, "name"),
        instruction=_required_string(data, "instruction"),
        environment=_required_string(data, "environment"),
        reset_instructions=_required_string(data, "reset_instructions"),
        success_description=_required_string(data, "success_description"),
        max_steps=max_steps,
        timeout_seconds=float(timeout_seconds),
        allowed_actions=allowed_actions,
        screenshot_path=_parse_screenshot_path(data),
        mock_actions=_parse_mock_actions(data),
        minesweeper=board,
        benchmark=benchmark,
    )


def load_task_set(paths: list[str | Path]) -> list[TaskSpec]:
    """Load a collection and reject ambiguous duplicate instance IDs."""
    tasks = [load_task(path) for path in paths]
    ids = [task.id for task in tasks]
    if len(ids) != len(set(ids)):
        raise TaskLoadError("Duplicate task instance IDs")
    return tasks


def agent_task(task: TaskSpec) -> TaskSpec:
    """Expose only public task instructions to an action-proposing model."""
    if task.benchmark is None:
        return task
    return replace(
        task,
        benchmark=None,
        minesweeper=None,
        mock_actions=(),
        screenshot_path=None,
        reset_instructions="A prepared board is shown.",
        success_description="Complete the instruction using the visible board.",
    )
