"""Harness that lets the agent loop play Minesweeper through screenshots and clicks.

The model side only sees saved screenshots and sends normalized mouse actions. Game
state is read here for two things the model never sees: ending the episode when the
game is over, and grading the result.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from .actions import Action, ActionType
from .benchmark.environment import prepare
from .minesweeper.screen import MinesweeperScreen
from .observations import Observation
from .results import EvaluationResult, EvaluationStatus, ExecutionResult, StepResult
from .tasks import TaskSpec


class MinesweeperSandbox:
    """Observation provider, executor, and evaluator for one Minesweeper task."""

    def __init__(self, task: TaskSpec, screenshot_dir: str | Path) -> None:
        self.screen = prepare(task) if task.benchmark else MinesweeperScreen(**task.minesweeper)
        self.screenshot_dir = Path(screenshot_dir)

    def capture(
        self,
        task: TaskSpec,
        episode_id: str,
        step: int,
        history: list[StepResult],
    ) -> Observation:
        """Save the current window as this step's screenshot."""
        path = self.screenshot_dir / episode_id / f"step_{step:03d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.screen.screenshot().save(path)
        return Observation(
            episode_id=episode_id,
            step=step,
            task_id=task.id,
            screenshot_path=str(path),
            capture_bounds=(0, 0, self.screen.width, self.screen.height),
        )

    def execute(self, action: Action) -> ExecutionResult:
        """Deliver a click as pixel input; other actions send nothing."""
        started = perf_counter()
        message = f"{action.type.value}: no input sent"
        if action.type in (ActionType.CLICK, ActionType.RIGHT_CLICK):
            # Normalized endpoints land on the first and last pixel
            px = round(action.x * (self.screen.width - 1))
            py = round(action.y * (self.screen.height - 1))
            button = "left" if action.type is ActionType.CLICK else "right"
            self.screen.click(px, py, button)
            message = f"{button} click at pixel ({px}, {py})"
        return ExecutionResult(
            True,
            message,
            perf_counter() - started,
            episode_done=self.screen.game.status != "playing",
        )

    def evaluate(self, task: TaskSpec, observation: Observation | None) -> EvaluationResult:
        """Grade from the true game state."""
        status = self.screen.game.status
        if status == "won":
            return EvaluationResult(EvaluationStatus.SUCCESS, "Board cleared.")
        return EvaluationResult(EvaluationStatus.FAILURE, f"Game {status}.")

    def shutdown(self) -> None:
        return None
