"""Results and termination records produced by the agent loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from typing import Any

from .actions import Action, action_to_dict
from .observations import Observation, observation_to_dict


class TerminationReason(str, Enum):
    """Why an episode stopped."""

    FINISH_REQUESTED = "finish_requested"
    MODEL_FAILURE = "model_failure"
    PARSE_ERROR = "parse_error"
    VALIDATION_ERROR = "validation_error"
    EXECUTION_ERROR = "execution_error"
    CAPTURE_ERROR = "capture_error"
    TIMEOUT = "timeout"
    ACTION_LIMIT = "action_limit"


class EvaluationStatus(str, Enum):
    """Independent task-evaluation outcomes."""

    SUCCESS = "success"
    FAILURE = "failure"
    UNKNOWN = "unknown"
    NOT_EVALUATED = "not_evaluated"


@dataclass(frozen=True)
class ExecutionResult:
    """Result returned by an action executor."""

    succeeded: bool
    message: str
    elapsed_seconds: float = 0.0


@dataclass(frozen=True)
class EvaluationResult:
    """Result returned by an independent evaluator."""

    status: EvaluationStatus
    message: str


@dataclass
class StepResult:
    """Complete record of one observation/model/action cycle."""

    observation: Observation | None
    raw_model_response: str | None = None
    action: Action | None = None
    model_elapsed_seconds: float = 0.0
    parse_error: str | None = None
    validation_error: str | None = None
    execution: ExecutionResult | None = None
    evaluation: EvaluationResult | None = None
    error: str | None = None


@dataclass
class EpisodeResult:
    """Summary and ordered records for an episode."""

    run_id: str
    task_id: str
    started_at: str
    ended_at: str
    termination_reason: TerminationReason
    evaluation: EvaluationResult
    steps: list[StepResult]


def to_jsonable(value: Any) -> Any:
    """Convert supported domain objects into JSON-compatible values."""
    if isinstance(value, Action):
        return action_to_dict(value)
    if isinstance(value, Observation):
        return observation_to_dict(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value
