"""The exact label provider must agree with brute-force enumeration of mine layouts."""

import json
import random
from itertools import combinations
from pathlib import Path

import pytest

from vembody.benchmark.environment import prepare
from vembody.benchmark.labels import CertificationRequired, VisibleBoard, visible_board
from vembody.benchmark.solver import PROVIDER_ID, ExactLabelProvider, build_label_provider
from vembody.cli import main
from vembody.evaluation.harness import ExperimentConfig, run_experiment
from vembody.minesweeper.game import Minesweeper
from vembody.tasks import load_task


def _board(game: Minesweeper) -> VisibleBoard:
    def symbol(cell):
        if cell in game.flagged:
            return "F"
        if cell in game.revealed:
            return str(game.adjacent_mines(cell)) if game.adjacent_mines(cell) else "."
        return "#"

    return VisibleBoard(
        game.rows,
        game.cols,
        game.mine_count,
        tuple(tuple(symbol((r, c)) for c in range(game.cols)) for r in range(game.rows)),
    )


def _brute_force(board: VisibleBoard) -> tuple[frozenset, frozenset]:
    """Enumerate every layout of the public mine count over hidden cells."""
    hidden = sorted(board.hidden)
    clues = []
    for r in range(board.rows):
        for c in range(board.cols):
            symbol = board.cells[r][c]
            if symbol in "#F":
                continue
            around = {
                (rr, cc)
                for rr in range(r - 1, r + 2)
                for cc in range(c - 1, c + 2)
                if (rr, cc) != (r, c) and 0 <= rr < board.rows and 0 <= cc < board.cols
            }
            clues.append((around, 0 if symbol == "." else int(symbol)))
    ever_mine, ever_safe = set(), set()
    for layout in combinations(hidden, board.total_mines):
        mines = set(layout)
        if all(len(around & mines) == number for around, number in clues):
            ever_mine |= mines
            ever_safe |= set(hidden) - mines
    assert ever_mine or ever_safe or not hidden, "board has no consistent layout"
    return frozenset(set(hidden) - ever_mine), frozenset(set(hidden) - ever_safe)


def _random_boards(count: int, seed: int = 0):
    rng = random.Random(seed)
    boards = []
    while len(boards) < count:
        rows, cols = rng.choice([(4, 4), (4, 5), (5, 5)])
        mines = rng.randint(2, 6)
        game = Minesweeper(rows, cols, mines, seed=rng.randrange(10**6))
        game.reveal((rng.randrange(rows), rng.randrange(cols)))
        # Open a few more safe cells so boards cover sparse and dense clue patterns
        safe = [
            (r, c)
            for r in range(rows)
            for c in range(cols)
            if (r, c) not in game.mines and (r, c) not in game.revealed
        ]
        for cell in rng.sample(safe, k=min(len(safe), rng.randint(0, 3))):
            game.reveal(cell)
        if game.status != "playing":
            continue
        # Flags, including wrong ones, must not change any label
        hidden = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in game.revealed]
        for cell in rng.sample(hidden, k=min(len(hidden), rng.randint(0, 2))):
            game.toggle_flag(cell)
        boards.append((game, _board(game)))
    return boards


@pytest.mark.parametrize("index", range(60))
def test_labels_match_brute_force(index):
    game, board = _random_boards(60)[index]
    labels = ExactLabelProvider().label(board)
    expected_safe, expected_mines = _brute_force(board)
    assert labels.safe == expected_safe
    assert labels.mines == expected_mines
    assert labels.provider == PROVIDER_ID
    # The true layout is always consistent, so labels can never contradict it
    assert not labels.safe & game.mines and labels.mines <= game.mines


def test_flags_are_ignored():
    game, board = next((g, b) for g, b in _random_boards(40, seed=3) if "F" in str(b.cells))
    unflagged = VisibleBoard(
        board.rows,
        board.cols,
        board.total_mines,
        tuple(tuple("#" if s == "F" else s for s in row) for row in board.cells),
    )
    provider = ExactLabelProvider()
    assert provider.label(board) == provider.label(unflagged)


def test_mine_count_labels_interior_cells():
    # Two hidden cells share one mine: neither is determined
    board = VisibleBoard(2, 4, 1, (("#", "1", ".", "."), ("#", "1", ".", ".")))
    labels = ExactLabelProvider().label(board)
    assert labels.mines == frozenset() and labels.safe == frozenset()
    # A 1 in the corner with one mine in total: the mine is next to the 1, so every
    # cell that touches no clue is provably safe from the mine count alone
    interior = VisibleBoard(3, 3, 1, (("1", "#", "#"), ("#", "#", "#"), ("#", "#", "#")))
    labels = ExactLabelProvider().label(interior)
    # The single mine is one of the 1's three neighbors, so every far cell is safe
    assert labels.safe == {(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)}
    assert labels.mines == frozenset()


def test_inconsistent_boards_and_budget_raise():
    provider = ExactLabelProvider()
    with pytest.raises(CertificationRequired, match="satisfied"):
        provider.label(VisibleBoard(1, 2, 1, (("3", "#"),)))
    with pytest.raises(CertificationRequired, match="mine count"):
        provider.label(VisibleBoard(1, 3, 2, (("#", "1", "."),)))
    _, board = _random_boards(1, seed=5)[0]
    with pytest.raises(CertificationRequired, match="budget"):
        ExactLabelProvider(max_nodes=1).label(board)
    with pytest.raises(ValueError):
        build_label_provider("guess")


def test_benchmark_boards_label_consistently_with_hidden_truth():
    for path in sorted(Path("tasks/benchmark/development").glob("t*.yaml")):
        screen = prepare(load_task(path))
        labels = ExactLabelProvider().label(visible_board(screen))
        assert not labels.safe & screen.game.mines, path
        assert labels.mines <= screen.game.mines, path


def test_t2_and_t3_tasks_run_with_the_provider(tmp_path):
    tasks = sorted(str(p) for p in Path("tasks/benchmark/development").glob("t[23]-*.yaml"))
    config = ExperimentConfig("solver", "solver", tasks, trials=1, label_provider="exact")
    rows = run_experiment(config, tmp_path / "exp")
    assert len(rows) == len(tasks) == 6
    assert not any(row.get("infrastructure_failure") for row in rows)
    manifest = json.loads((tmp_path / "exp" / "manifest.json").read_text())
    assert manifest["label_provider"] == PROVIDER_ID
    assert manifest["config"]["label_provider"] == "exact"


def test_cli_accepts_label_provider(tmp_path):
    code = main(
        [
            "eval",
            "--tasks",
            "tasks/benchmark",
            "--split",
            "development",
            "--model",
            "random-cell",
            "--trials",
            "1",
            "--label-provider",
            "exact",
            "--name",
            "dev",
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 0
    lines = (tmp_path / "dev" / "episodes.jsonl").read_text().splitlines()
    assert len(lines) == 12
