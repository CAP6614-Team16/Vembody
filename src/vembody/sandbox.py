"""Harness that lets the agent loop play Minesweeper through screenshots and clicks.

The model side only sees saved screenshots and sends normalized mouse actions. Game
state is read here for two things the model never sees: ending the episode when the
game is over, and grading the result.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from .actions import Action, ActionType
from .minesweeper.screen import MinesweeperScreen
from .observations import Observation
from .results import EvaluationResult, EvaluationStatus, ExecutionResult, StepResult
from .tasks import TaskSpec


class MinesweeperSandbox:
    """Observation provider, executor, and evaluator for one Minesweeper task."""

    def __init__(self, task: TaskSpec, screenshot_dir: str | Path) -> None:
        self.screen = MinesweeperScreen(**task.minesweeper)
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

    def board_stats(self) -> dict[str, Any]:
        """Progress counters from the true game state, for evaluation only."""
        game = self.screen.game
        return {
            "status": game.status,
            "safe_total": game.rows * game.cols - game.mine_count,
            "safe_revealed": len(game.revealed - game.mines),
            "mines_placed": bool(game.mines),
            "flags_correct": len(game.flagged & game.mines),
            "flags_wrong": len(game.flagged - game.mines),
        }

    def _snapshot(self) -> tuple:
        game = self.screen.game
        return (id(game), frozenset(game.revealed), frozenset(game.flagged), game.status)

    def execute(self, action: Action) -> ExecutionResult:
        """Deliver a click as pixel input; other actions send nothing."""
        started = perf_counter()
        message = f"{action.type.value}: no input sent"
        before = self._snapshot()
        details: dict[str, Any] = {"target": None, "cell": None, "reset": False}
        if action.type in (ActionType.CLICK, ActionType.RIGHT_CLICK):
            # Normalized endpoints land on the first and last pixel
            px = round(action.x * (self.screen.width - 1))
            py = round(action.y * (self.screen.height - 1))
            button = "left" if action.type is ActionType.CLICK else "right"
            target, cell = self.screen.locate(px, py)
            self.screen.click(px, py, button)
            message = f"{button} click at pixel ({px}, {py})"
            details.update(
                target=target,
                cell=list(cell) if cell else None,
                reset=target == "face" and button == "left",
            )
        details["board_changed"] = self._snapshot() != before
        details["board"] = self.board_stats()
        return ExecutionResult(
            True,
            message,
            perf_counter() - started,
            episode_done=self.screen.game.status != "playing",
            details=details,
        )

    def evaluate(self, task: TaskSpec, observation: Observation | None) -> EvaluationResult:
        """Grade from the true game state."""
        status = self.screen.game.status
        if status == "won":
            return EvaluationResult(EvaluationStatus.SUCCESS, "Board cleared.")
        return EvaluationResult(EvaluationStatus.FAILURE, f"Game {status}.")

    def shutdown(self) -> None:
        return None
