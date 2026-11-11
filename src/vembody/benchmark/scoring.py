"""Trace-aware task predicates. Hidden truth grades outcomes, never logical proof."""

from __future__ import annotations

from ..actions import ActionType
from ..minesweeper.screen import MinesweeperScreen
from ..results import EvaluationResult, EvaluationStatus
from ..tasks import TaskSpec
from .labels import CertificationRequired, LabelProvider, LogicalLabels, visible_board
from .spec import Cell

# Only these actions can change the board, so only these can need logical evidence.
BOARD_INPUT = (ActionType.CLICK, ActionType.RIGHT_CLICK)


class TaskScorer:
    def __init__(
        self, task: TaskSpec, screen: MinesweeperScreen, provider: LabelProvider | None = None
    ):
        self.spec = task.benchmark
        if self.spec is None:
            raise ValueError("Missing benchmark")
        if self.spec.split == "evaluation":
            raise CertificationRequired(
                "Evaluation release requires reviewed instance certificates"
            )
        self.predicate = self.spec.predicate
        self.provider = provider
        self.needs_labels = self.spec.template_id.startswith(("T2", "T3"))
        if self.needs_labels and provider is None:
            raise CertificationRequired("This task requires Nicholas's logical label provider")
        self.safe_reveals = 0
        self.agent_flags = set(screen.game.flagged)
        self.region_mines = {
            n
            for c in self.predicate.region
            for n in screen.game.neighbors(c)
            if n in screen.game.mines and n not in screen.game.revealed
        }
        self.success = False
        self.violation = ""

    def labels(self, screen: MinesweeperScreen) -> LogicalLabels | None:
        if self.provider is None or self.spec.template_id.startswith("T1"):
            return None
        board = visible_board(screen)
        try:
            labels = self.provider.label(board)
        except Exception as exc:
            raise CertificationRequired(f"Logical provider failed: {exc}") from exc
        if not isinstance(labels, LogicalLabels):
            raise CertificationRequired("Provider returned invalid label record")
        labels.validate(board)
        return labels

    def update(
        self,
        kind: ActionType,
        target: Cell | None,
        before: dict,
        screen: MinesweeperScreen,
        labels: LogicalLabels | None,
    ) -> dict:
        game, p = screen.game, self.predicate
        old_revealed = {tuple(c) for c in before["revealed"]}
        old_flags = {tuple(c) for c in before["flagged"]}
        revealed = game.revealed - old_revealed
        new_flag = (
            kind == ActionType.RIGHT_CLICK and target not in old_flags and target in game.flagged
        )
        unflag = (
            kind == ActionType.RIGHT_CLICK and target in old_flags and target not in game.flagged
        )
        effective = bool(revealed or new_flag or unflag or game.status == "lost")
        logical = None
        if self.needs_labels and labels is None and kind in BOARD_INPUT:
            raise CertificationRequired("Logical evidence is required")
        if labels is not None:
            if kind == ActionType.CLICK and (target in revealed or game.status == "lost"):
                logical = target in labels.safe
            elif new_flag:
                logical = target in labels.mines
        if new_flag:
            self.agent_flags.add(target)
        elif unflag:
            self.agent_flags.discard(target)
        if game.status == "lost":
            self.violation = "Mine revealed."
        elif p.type in ("exact_reveal", "exact_flag") and effective:
            if p.type == "exact_reveal":
                self.success = (
                    kind == ActionType.CLICK
                    and target == p.target
                    and revealed == {p.target}
                    and game.flagged == old_flags
                )
            else:
                self.success = (
                    new_flag
                    and target == p.target
                    and not revealed
                    and game.flagged == old_flags | {p.target}
                )
            if not self.success:
                self.violation = "A board change did not match the requested operation."
        elif p.type in ("one_provable_reveal", "one_provable_flag") and effective:
            correct_kind = kind == ActionType.CLICK if p.type == "one_provable_reveal" else new_flag
            self.success = bool(correct_kind and logical)
            if not self.success:
                self.violation = "First effective move lacks the required proof or action type."
        elif p.type in ("flag_region_mines", "safe_reveal_count"):
            if new_flag and target not in game.mines:
                self.violation = "A safe cell was flagged."
            if p.type == "flag_region_mines":
                self.success = bool(self.region_mines and self.region_mines <= self.agent_flags)
            elif kind == ActionType.CLICK and target in revealed:
                if logical:
                    self.safe_reveals += 1
                else:
                    self.violation = "Reveal was not provably safe before the action."
            if p.type == "safe_reveal_count":
                self.success = self.safe_reveals >= p.count
        elif p.type == "board_won":
            self.success = game.status == "won"
        if self.violation:
            self.success = False
        return {
            "effective": effective,
            "logical_correct": logical,
            "safe_reveals": self.safe_reveals,
            "task_success": self.success,
            "task_violation": self.violation,
            "new_flag": new_flag,
            "unflag": unflag,
            "provider": labels.provider if labels else None,
            "certification_status": "pending",
        }

    @property
    def done(self) -> bool:
        return self.success or bool(self.violation)

    def evaluate(self) -> EvaluationResult:
        return EvaluationResult(
            EvaluationStatus.SUCCESS if self.success else EvaluationStatus.FAILURE,
            "Task completed."
            if self.success
            else self.violation or "Task predicate is not satisfied.",
        )
