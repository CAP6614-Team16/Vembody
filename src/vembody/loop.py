"""Dependency-injected agent episode loop."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from .actions import ActionType
from .benchmark.labels import CertificationRequired
from .errors import ActionValidationError, EpisodeDeadlineExceeded, ModelParseError
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
from .tasks import TaskSpec, agent_task
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
                result = StepResult(observation=observation)
                try:
                    # Keep private grading channels in logs, including on legacy tasks.
                    model_history = [
                        replace(
                            item,
                            evaluation=None,
                            execution=replace(item.execution, evidence={}, details={})
                            if item.execution
                            else None,
                        )
                        for item in history
                    ]
                    response = self.model.generate_action(
                        agent_task(task), observation, model_history
                    )
                    result.raw_model_response = response.raw_text
                    result.model_metadata = dict(response.metadata)
                    result.model_elapsed_seconds = max(
                        response.elapsed_seconds, perf_counter() - model_started
                    )
                    parsed = parse_model_response(response.raw_text)
                    result.action = parsed.action
                    validate_action(parsed.action, self.policy)
                except ModelParseError as exc:
                    result.model_elapsed_seconds = max(
                        result.model_elapsed_seconds, perf_counter() - model_started
                    )
                    result.parse_error = str(exc)
                    self._record_step(history, result)
                    termination = TerminationReason.PARSE_ERROR
                    break
                except ActionValidationError as exc:
                    result.validation_error = str(exc)
                    self._record_step(history, result)
                    termination = TerminationReason.VALIDATION_ERROR
                    break
                except Exception as exc:
                    result.model_elapsed_seconds = max(
                        result.model_elapsed_seconds, perf_counter() - model_started
                    )
                    result.error = str(exc)
                    self._record_step(history, result)
                    termination = TerminationReason.MODEL_FAILURE
                    break

                try:
                    if perf_counter() - started_clock >= task.timeout_seconds:
                        self._record_step(history, result)
                        termination = TerminationReason.TIMEOUT
                        break
                    deadline_setter = getattr(self.executor, "set_deadline", None)
                    if deadline_setter is not None:
                        deadline_setter(started_clock + task.timeout_seconds)
                    result.execution = self.executor.execute(parsed.action)
                except EpisodeDeadlineExceeded as exc:
                    result.error = str(exc)
                    self._record_step(history, result)
                    termination = TerminationReason.TIMEOUT
                    break
                except CertificationRequired as exc:
                    result.error = str(exc)
                    self._record_step(history, result)
                    termination = TerminationReason.EVALUATION_ERROR
                    break
                except Exception as exc:
                    result.error = str(exc)
                    self._record_step(history, result)
                    termination = TerminationReason.EXECUTION_ERROR
                    break

                if not result.execution.succeeded:
                    self._record_step(history, result)
                    termination = TerminationReason.EXECUTION_ERROR
                    break

                try:
                    if result.execution.task_done and not result.execution.episode_done:
                        evaluation = self.evaluator.evaluate(task, last_observation)
                        result.evaluation = evaluation
                        termination = TerminationReason.TASK_COMPLETE
                    elif parsed.action.type is ActionType.FINISH:
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
                    elif result.execution.episode_done:
                        evaluation = self.evaluator.evaluate(task, last_observation)
                        result.evaluation = evaluation
                        termination = TerminationReason.GAME_OVER
                except Exception as exc:
                    result.error = f"Evaluation failed: {exc}"
                    self._record_step(history, result)
                    termination = TerminationReason.EVALUATION_ERROR
                    break

                self._record_step(history, result)
                if (
                    parsed.action.type in {ActionType.FINISH, ActionType.FAIL}
                    or result.execution.episode_done
                    or result.execution.task_done
                ):
                    break
        finally:
            try:
                self.executor.shutdown()
            except Exception as exc:
                self._record_step(history, StepResult(None, error=f"Shutdown failed: {exc}"))
                termination = TerminationReason.EXECUTION_ERROR

        if (
            task.benchmark
            and evaluation.status == EvaluationStatus.NOT_EVALUATED
            and termination
            in {
                TerminationReason.ACTION_LIMIT,
                TerminationReason.TIMEOUT,
                TerminationReason.PARSE_ERROR,
                TerminationReason.VALIDATION_ERROR,
                TerminationReason.MODEL_FAILURE,
            }
        ):
            evaluation = EvaluationResult(EvaluationStatus.FAILURE, "Task was not completed.")

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
