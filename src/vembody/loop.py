"""Dependency-injected agent episode loop."""

from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from .actions import ActionType
from .errors import ActionValidationError, ModelParseError
from .evaluation.base import SuccessEvaluator
from .execution.base import ActionExecutor
from .logging.jsonl import JsonlLogger
from .models.base import VisionModel
from .observations import Observation
from .parsing import parse_model_response
from .perception.base import ObservationProvider
from .results import (
    EpisodeResult,
    EvaluationResult,
    EvaluationStatus,
    StepResult,
    TerminationReason,
)
from .tasks import TaskSpec
from .validation import ActionPolicy, validate_action


class AgentLoop:
    """Run one task through observation, proposal, validation, and execution."""

    def __init__(
        self,
        model: VisionModel,
        observations: ObservationProvider,
        executor: ActionExecutor,
        evaluator: SuccessEvaluator,
        policy: ActionPolicy,
        logger: JsonlLogger,
    ) -> None:
        self.model = model
        self.observations = observations
        self.executor = executor
        self.evaluator = evaluator
        self.policy = policy
        self.logger = logger

    def run_episode(self, task: TaskSpec) -> EpisodeResult:
        """Run an episode until a bounded termination condition occurs."""
        run_id = uuid4().hex
        started_at = datetime.now(timezone.utc).isoformat()
        started_clock = perf_counter()
        history: list[StepResult] = []
        termination = TerminationReason.ACTION_LIMIT
        evaluation = EvaluationResult(EvaluationStatus.NOT_EVALUATED, "Episode did not finish.")
        last_observation: Observation | None = None

        try:
            for step in range(task.max_steps):
                if perf_counter() - started_clock >= task.timeout_seconds:
                    termination = TerminationReason.TIMEOUT
                    break
                try:
                    observation = self.observations.capture(task, run_id, step, history)
                    last_observation = observation
                except Exception as exc:
                    result = StepResult(None, error=str(exc))
                    self._record_step(history, result)
                    termination = TerminationReason.CAPTURE_ERROR
                    break

                model_started = perf_counter()
                response = None
                try:
                    response = self.model.generate_action(task, observation, history)
                    parsed = parse_model_response(response.raw_text)
                    result = StepResult(
                        observation=observation,
                        raw_model_response=response.raw_text,
                        action=parsed.action,
                        model_elapsed_seconds=max(
                            response.elapsed_seconds, perf_counter() - model_started
                        ),
                    )
                    validate_action(parsed.action, self.policy)
                except ModelParseError as exc:
                    result = StepResult(
                        observation=observation,
                        raw_model_response=response.raw_text if response is not None else None,
                        model_elapsed_seconds=perf_counter() - model_started,
                        parse_error=str(exc),
                    )
                    self._record_step(history, result)
                    termination = TerminationReason.PARSE_ERROR
                    break
                except ActionValidationError as exc:
                    result.validation_error = str(exc)
                    self._record_step(history, result)
                    termination = TerminationReason.VALIDATION_ERROR
                    break
                except Exception as exc:
                    result = StepResult(
                        observation=observation,
                        raw_model_response=response.raw_text if response is not None else None,
                        model_elapsed_seconds=perf_counter() - model_started,
                        error=str(exc),
                    )
                    self._record_step(history, result)
                    termination = TerminationReason.MODEL_FAILURE
                    break

                try:
                    result.execution = self.executor.execute(parsed.action)
                except Exception as exc:
                    result.error = str(exc)
                    self._record_step(history, result)
                    termination = TerminationReason.EXECUTION_ERROR
                    break

                if not result.execution.succeeded:
                    self._record_step(history, result)
                    termination = TerminationReason.EXECUTION_ERROR
                    break

                if parsed.action.type is ActionType.FINISH:
                    evaluation = self.evaluator.evaluate(task, last_observation)
                    result.evaluation = evaluation
                    termination = TerminationReason.FINISH_REQUESTED
                elif parsed.action.type is ActionType.FAIL:
                    evaluation = EvaluationResult(
                        EvaluationStatus.FAILURE,
                        "Model requested failure.",
                    )
                    result.evaluation = evaluation
                    termination = TerminationReason.MODEL_FAILURE

                self._record_step(history, result)
                if parsed.action.type in {ActionType.FINISH, ActionType.FAIL}:
                    break
        finally:
            self.executor.shutdown()

        ended_at = datetime.now(timezone.utc).isoformat()
        episode = EpisodeResult(
            run_id,
            task.id,
            started_at,
            ended_at,
            termination,
            evaluation,
            history,
        )
        self.logger.write_episode(episode)
        return episode

    def _record_step(self, history: list[StepResult], result: StepResult) -> None:
        """Append a step to in-memory history and the JSONL log."""
        history.append(result)
        self.logger.write_step(result)
