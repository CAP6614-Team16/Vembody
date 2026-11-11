"""Predicate tests use synthetic truth labels ONLY as test fixtures, never certificates."""

from dataclasses import replace
from pathlib import Path

import pytest

from vembody.actions import Action, ActionType
from vembody.benchmark.labels import CertificationRequired, LogicalLabels
from vembody.cli import build_loop, build_parser
from vembody.results import EvaluationStatus, TerminationReason
from vembody.sandbox import MinesweeperSandbox
from vembody.tasks import load_task

ROOT = Path("tasks/benchmark/development")


class SyntheticLabels:
    """An explicit test double; it proves nothing about a real instance."""

    def __init__(self, mines):
        self.mines = frozenset(mines)

    def label(self, board):
        assert not hasattr(board, "mines")
        return LogicalLabels(board.hidden - self.mines, board.hidden & self.mines, "TEST-ONLY")


def sandbox_for(template, tmp_path, labels=True):
    task = load_task(ROOT / f"{template.lower()}.yaml")
    from vembody.benchmark.environment import prepare

    provider = SyntheticLabels(prepare(task).game.mines) if labels else None
    return task, MinesweeperSandbox(task, tmp_path, provider)


def act(sandbox, target, kind=ActionType.CLICK):
    px, py = sandbox.screen.cell_center(target)
    return sandbox.execute(
        Action(kind, px / (sandbox.screen.width - 1), py / (sandbox.screen.height - 1))
    )


@pytest.mark.parametrize("template", ["T1-01", "T1-02", "T1-03", "T1-04"])
def test_grounding_success_and_wrong_target(template, tmp_path):
    task, sandbox = sandbox_for(template, tmp_path, labels=False)
    p = task.benchmark.predicate
    result = act(
        sandbox, p.target, ActionType.RIGHT_CLICK if p.type == "exact_flag" else ActionType.CLICK
    )
    assert result.task_done and sandbox.evaluate(task, None).status == EvaluationStatus.SUCCESS
    _, sandbox = sandbox_for(template, tmp_path, labels=False)
    wrong = next(
        (r, c)
        for r in range(9)
        for c in range(9)
        if (r, c) != p.target and (r, c) not in sandbox.screen.game.revealed
    )
    result = act(sandbox, wrong, ActionType.RIGHT_CLICK)
    assert result.task_done and sandbox.evaluate(task, None).status == EvaluationStatus.FAILURE


@pytest.mark.parametrize("template", ["T2-01", "T2-02", "T2-03", "T2-04"])
def test_single_deduction_success_or_unjustified_move(template, tmp_path):
    task, sandbox = sandbox_for(template, tmp_path)
    game = sandbox.screen.game
    flag = template == "T2-02"
    targets = (
        game.mines
        if flag
        else {
            (r, c)
            for r in range(9)
            for c in range(9)
            if (r, c) not in game.mines and (r, c) not in game.revealed
        }
    )
    result = act(sandbox, next(iter(targets)), ActionType.RIGHT_CLICK if flag else ActionType.CLICK)
    assert result.task_done and sandbox.scorer.success
    _, sandbox = sandbox_for(template, tmp_path)

    class NoProof:
        def label(self, board):
            return LogicalLabels(frozenset(), frozenset(), "TEST-NO-PROOF")

    sandbox.scorer.provider = NoProof()
    act(sandbox, next(iter(targets)), ActionType.RIGHT_CLICK if flag else ActionType.CLICK)
    assert sandbox.scorer.violation and not sandbox.scorer.success


def test_region_counts_agent_flags_not_automatic_flags_and_rejects_wrong_flag(tmp_path):
    task, sandbox = sandbox_for("T3-01", tmp_path)
    for target in sandbox.scorer.region_mines:
        result = act(sandbox, target, ActionType.RIGHT_CLICK)
    assert result.task_done and sandbox.scorer.success
    _, sandbox = sandbox_for("T3-01", tmp_path)
    game = sandbox.screen.game
    wrong = next(
        (r, c)
        for r in range(9)
        for c in range(9)
        if (r, c) not in game.mines and (r, c) not in game.revealed
    )
    act(sandbox, wrong, ActionType.RIGHT_CLICK)
    assert sandbox.scorer.violation and not sandbox.scorer.success


def test_ten_moves_count_clicks_not_flooded_cells(tmp_path):
    task, sandbox = sandbox_for("T3-02", tmp_path)
    game = sandbox.screen.game
    for _ in range(10):
        target = next(
            (r, c)
            for r in range(9)
            for c in range(9)
            if (r, c) not in game.mines
            and (r, c) not in game.revealed
            and game.adjacent_mines((r, c))
        )
        act(sandbox, target)
    assert sandbox.scorer.success and sandbox.scorer.safe_reveals == 10


@pytest.mark.parametrize("template", ["T4-01", "T4-02"])
def test_full_board_win_and_loss(template, tmp_path):
    task, sandbox = sandbox_for(template, tmp_path, labels=False)
    game = sandbox.screen.game
    for r in range(game.rows):
        for c in range(game.cols):
            if game.status == "playing" and (r, c) not in game.mines:
                act(sandbox, (r, c))
    assert sandbox.scorer.success and sandbox.scorer.agent_flags == set()
    _, sandbox = sandbox_for(template, tmp_path, labels=False)
    act(sandbox, next(iter(sandbox.screen.game.mines)))
    assert not sandbox.scorer.success and sandbox.scorer.violation


def test_logical_tasks_and_evaluation_fail_closed(tmp_path):
    for template in ("T2-01", "T2-02", "T2-03", "T2-04", "T3-01", "T3-02"):
        with pytest.raises(CertificationRequired, match="provider"):
            sandbox_for(template, tmp_path, labels=False)
    task = load_task(ROOT / "t1-01.yaml")
    task = replace(task, benchmark=replace(task.benchmark, split="evaluation"))
    with pytest.raises(CertificationRequired, match="certificates"):
        MinesweeperSandbox(task, tmp_path)


def test_grounding_through_loop_records_terminal_image_and_no_proof(tmp_path):
    args = build_parser().parse_args(
        ["run", "--task", str(ROOT / "t1-01.yaml"), "--log", str(tmp_path / "episode.jsonl")]
    )
    loop, task = build_loop(args)
    result = loop.run_episode(task)
    assert result.termination_reason == TerminationReason.TASK_COMPLETE
    assert result.evaluation.status == EvaluationStatus.SUCCESS
    evidence = result.steps[0].execution.evidence
    assert Path(evidence["post_screenshot_path"]).exists()
    assert evidence["logical_correct"] is None and evidence["certification_status"] == "pending"


def test_actions_that_send_no_input_do_not_call_the_provider(tmp_path):
    """A bounded solver call is expensive; only board input can need one."""
    task, sandbox = sandbox_for("T2-01", tmp_path)
    calls = []
    inner = sandbox.scorer.provider

    class Counting:
        def label(self, board):
            calls.append(board)
            return inner.label(board)

    sandbox.scorer.provider = Counting()
    result = sandbox.execute(Action(ActionType.FINISH, 0.0, 0.0))
    assert not calls
    assert not result.evidence["effective"] and not result.evidence["task_success"]
    act(sandbox, next(iter(sandbox.screen.game.mines)), ActionType.RIGHT_CLICK)
    assert len(calls) == 1


def test_bad_label_records_and_provider_timeouts(tmp_path):
    from vembody.benchmark.labels import VisibleBoard

    board = VisibleBoard(2, 2, 1, (("#", "#"), ("1", ".")))
    with pytest.raises(CertificationRequired):
        LogicalLabels(frozenset({(1, 1)}), frozenset(), "bad").validate(board)
    with pytest.raises(CertificationRequired):
        LogicalLabels(frozenset({(0, 0)}), frozenset({(0, 0)}), "bad").validate(board)
    for malformed in [
        LogicalLabels(None, frozenset(), "x"),
        LogicalLabels(frozenset({1}), frozenset(), "x"),
        LogicalLabels(frozenset(), frozenset(), 123),
    ]:
        with pytest.raises(CertificationRequired):
            malformed.validate(board)
    _, sandbox = sandbox_for("T2-01", tmp_path)

    class Timeout:
        def label(self, board):
            raise CertificationRequired("solver timeout")

    sandbox.scorer.provider = Timeout()
    before = set(sandbox.screen.game.revealed)
    with pytest.raises(CertificationRequired, match="timeout"):
        act(sandbox, next(iter(sandbox.screen.game.mines)))
    assert sandbox.screen.game.revealed == before


def test_label_latency_cannot_dispatch_late_click(tmp_path, monkeypatch):
    from vembody.benchmark.environment import state_hash
    from vembody.errors import EpisodeDeadlineExceeded

    _, sandbox = sandbox_for("T2-01", tmp_path)
    sandbox.set_deadline(100)
    clock = [90]
    monkeypatch.setattr("vembody.sandbox.perf_counter", lambda: clock[0])

    class SlowProvider:
        def label(self, board):
            clock[0] = 101
            return LogicalLabels(frozenset(), frozenset(), "TEST-SLOW")

    sandbox.scorer.provider = SlowProvider()
    before = state_hash(sandbox.screen)
    with pytest.raises(EpisodeDeadlineExceeded):
        act(sandbox, next(iter(sandbox.screen.game.mines)))
    assert state_hash(sandbox.screen) == before
