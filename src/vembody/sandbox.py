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
from .benchmark.environment import prepare, snapshot
from .benchmark.labels import LabelProvider, visible_board
from .benchmark.scoring import TaskScorer
from .errors import EpisodeDeadlineExceeded, OffTargetInput
from .minesweeper.screen import MinesweeperScreen
from .observations import Observation
from .results import EvaluationResult, EvaluationStatus, ExecutionResult, StepResult
from .tasks import TaskSpec


class MinesweeperSandbox:
    """Observation provider, executor, and evaluator for one Minesweeper task."""

    def __init__(
        self,
        task: TaskSpec,
        screenshot_dir: str | Path,
        label_provider: LabelProvider | None = None,
    ) -> None:
        self.screen = prepare(task) if task.benchmark else MinesweeperScreen(**task.minesweeper)
        self.screenshot_dir = Path(screenshot_dir)
        self.scorer = TaskScorer(task, self.screen, label_provider) if task.benchmark else None
        self._step_path: Path | None = None
        self._deadline: float | None = None

    def set_deadline(self, deadline: float) -> None:
        self._deadline = deadline

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
        self._step_path = path
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
        sends_input = action.type in (ActionType.CLICK, ActionType.RIGHT_CLICK)
        before = snapshot(self.screen) if self.scorer else None
        visible = visible_board(self.screen) if self.scorer else None
        # Only board-changing input can be judged against logical evidence, so an
        # action that sends nothing never pays for a provider call.
        labels = self.scorer.labels(self.screen) if self.scorer and sends_input else None
        if self._deadline is not None and perf_counter() >= self._deadline:
            raise EpisodeDeadlineExceeded("Episode deadline expired before input")
        target = None
        off_target = False
        message = f"{action.type.value}: no input sent"
        board_before = self._snapshot()
        details: dict[str, Any] = {"target": None, "cell": None, "reset": False}
        if sends_input:
            # Normalized endpoints land on the first and last pixel
            px = round(action.x * (self.screen.width - 1))
            py = round(action.y * (self.screen.height - 1))
            button = "left" if action.type is ActionType.CLICK else "right"
            target_kind, target = self.screen.locate(px, py)
            details.update(
                target=target_kind,
                cell=list(target) if target is not None else None,
                reset=target_kind == "face" and button == "left",
            )
            try:
                self.screen.click(px, py, button)
                message = f"{button} click at pixel ({px}, {py})"
            except OffTargetInput:
                # Missing the grid is an agent error, not an infrastructure error.
                # The board is unchanged, the step still spends budget, and the
                # episode continues so the miss can be graded as a failure.
                off_target = True
                message = f"{button} click at pixel ({px}, {py}) hit no cell interior"

        details["board_changed"] = self._snapshot() != board_before
        details["board"] = self.board_stats()
        evidence = {}
        if self.scorer:
            evidence = self.scorer.update(action.type, target, before, self.screen, labels)
            evidence.update(
                before=before,
                after=snapshot(self.screen),
                visible_before=visible.cells,
                resolved_cell=target,
                off_target=off_target,
            )
            if labels is not None:
                evidence["labels"] = {"safe": sorted(labels.safe), "mines": sorted(labels.mines)}
            if self._step_path is not None:
                path = self._step_path.with_name(self._step_path.stem + "_after.png")
                self.screen.screenshot().save(path)
                evidence["post_screenshot_path"] = str(path)
        return ExecutionResult(
            True,
            message,
            perf_counter() - started,
            episode_done=self.screen.game.status != "playing",
            task_done=self.scorer.done if self.scorer else False,
            evidence=evidence,
            details=details,
        )

    def evaluate(self, task: TaskSpec, observation: Observation | None) -> EvaluationResult:
        """Grade from the true game state."""
        status = self.screen.game.status
        if self.scorer:
            return self.scorer.evaluate()
        if status == "won":
            return EvaluationResult(EvaluationStatus.SUCCESS, "Board cleared.")
        return EvaluationResult(EvaluationStatus.FAILURE, f"Game {status}.")

    def shutdown(self) -> None:
        return None
