"""Every T2/T3 instance must start with a provably correct move, graded end to end."""

import json
import random
from pathlib import Path

import pytest

from vembody.actions import Action, ActionType
from vembody.benchmark.certify import (
    LOGICAL_TEMPLATES,
    Clue,
    certify,
    min_clicks_to_win,
    pair_deductions,
    propagate,
    subset_deductions,
)
from vembody.benchmark.environment import prepare, state_hash
from vembody.benchmark.labels import VisibleBoard, visible_board
from vembody.benchmark.solver import ExactLabelProvider
from vembody.minesweeper.game import Minesweeper
from vembody.results import EvaluationStatus
from vembody.sandbox import MinesweeperSandbox
from vembody.tasks import load_task

ROOT = Path("tasks/benchmark")
LOGICAL = sorted(
    p
    for split in ("development", "demonstration", "evaluation")
    for p in (ROOT / split).glob("t[23]-*.yaml")
)


def _id(path: Path) -> str:
    return f"{path.parent.name}/{path.stem}"


def act(sandbox, target, kind=ActionType.CLICK):
    px, py = sandbox.screen.cell_center(target)
    return sandbox.execute(
        Action(kind, px / (sandbox.screen.width - 1), py / (sandbox.screen.height - 1))
    )


def test_every_split_has_all_logical_instances():
    assert len(LOGICAL) == 6 + 6 + 30


@pytest.mark.parametrize("path", LOGICAL, ids=_id)
def test_instance_is_certified_and_manifest_matches(path):
    task = load_task(path)
    screen = prepare(task)
    spec = task.benchmark
    assert spec.template_id in LOGICAL_TEMPLATES
    assert certify(spec.template_id, screen.game, spec.predicate.region) is not None
    manifest = json.loads((path.parent / "manifest.json").read_text())
    record = next(r for r in manifest["instances"] if r["path"] == path.name)
    assert record["certification_status"] == "solver_certified"
    assert record["state_hash"] == state_hash(screen)


@pytest.mark.parametrize("path", LOGICAL, ids=_id)
def test_instance_is_solvable_through_the_sandbox(path, tmp_path):
    task = load_task(path)
    template = task.benchmark.template_id
    provider = ExactLabelProvider()
    orders = 3 if template == "T3-02" else 1
    for order in range(orders):
        sandbox = MinesweeperSandbox(task, tmp_path / str(order), ExactLabelProvider())
        game = sandbox.screen.game
        labels = provider.label(visible_board(sandbox.screen))
        if template == "T2-02":
            act(sandbox, min(labels.mines), ActionType.RIGHT_CLICK)
        elif template.startswith("T2"):
            act(sandbox, min(labels.safe))
        elif template == "T3-01":
            for mine in sorted(sandbox.scorer.region_mines):
                assert mine in labels.mines
                act(sandbox, mine, ActionType.RIGHT_CLICK)
        else:
            # Any order of provable reveals must reach ten moves within the budget
            rng = random.Random(order)
            for _ in range(task.max_steps):
                if sandbox.scorer.success:
                    break
                safe = provider.label(visible_board(sandbox.screen)).safe
                assert safe, "no provable move left before ten reveals"
                act(sandbox, rng.choice(sorted(safe)))
            assert game.status == "playing" or sandbox.scorer.safe_reveals >= 10
        assert sandbox.evaluate(task, None).status == EvaluationStatus.SUCCESS, template


def test_single_clue_and_mine_count_propagation():
    # Two hidden cells share one mine: nothing is proved
    board = VisibleBoard(2, 3, 1, (("1", "1", "#"), ("1", "1", "#")))
    safe, mines, _ = propagate(board)
    assert mines == set() and safe == set()
    # The 1 touches only (0, 1), so it is the mine; the mine count then clears (0, 2)
    board = VisibleBoard(1, 3, 1, (("1", "#", "#"),))
    safe, mines, _ = propagate(board)
    assert mines == {(0, 1)} and safe == {(0, 2)}


def test_subset_and_two_clue_rules():
    p, q, r, s = (0, 0), (0, 1), (0, 2), (0, 3)
    small = Clue((1, 0), frozenset({p, q}), 1)
    large = Clue((1, 1), frozenset({p, q, r}), 1)
    assert [cells for *_, cells in subset_deductions([small, large])] == [frozenset({r})]
    assert frozenset({r}) in [cells for *_, cells in pair_deductions([small, large])]
    # Neither clue contains the other, but together they force s to be safe
    first = Clue((1, 0), frozenset({p, q, r}), 2)
    second = Clue((1, 3), frozenset({q, r, s}), 1)
    assert subset_deductions([first, second]) == []
    assert [cells for *_, cells in pair_deductions([first, second])] == [frozenset({s})]


def test_min_clicks_counts_flood_regions_and_isolated_numbers():
    game = Minesweeper(3, 3, 1, seed=0)
    game.mines = {(0, 0)}
    game.revealed = set()
    # One click on (2, 2) floods every safe cell
    assert min_clicks_to_win(game) == 1
    game.mines = {(1, 1)}
    # Every safe cell touches the centre mine, so each needs its own click
    assert min_clicks_to_win(game) == 8


def test_rejects_wrong_templates_and_flagged_starts():
    task = load_task(ROOT / "development" / "t2-01.yaml")
    game = prepare(task).game
    with pytest.raises(ValueError):
        certify("T1-01", game)
    game.toggle_flag(next(c for c in sorted(game.mines)))
    assert certify("T2-01", game) is None
