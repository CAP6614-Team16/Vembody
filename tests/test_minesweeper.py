from pathlib import Path

from vembody.actions import action_from_dict
from vembody.cli import build_loop, build_parser
from vembody.minesweeper.game import Minesweeper
from vembody.minesweeper.screen import CELL, HEADER, MARGIN
from vembody.results import EvaluationStatus, TerminationReason
from vembody.sandbox import MinesweeperSandbox
from vembody.tasks import load_task


def test_first_click_is_safe_and_seeded() -> None:
    boards = []
    for _ in range(2):
        game = Minesweeper(9, 9, 10, seed=3)
        game.reveal((0, 0))
        assert game.status != "lost"
        assert len(game.mines) == 10 and (0, 0) not in game.mines
        boards.append(game.mines)
    assert boards[0] == boards[1]


def test_revealing_every_safe_cell_wins() -> None:
    game = Minesweeper(5, 5, 3, seed=1)
    game.reveal((2, 2))
    for row in range(5):
        for col in range(5):
            if (row, col) not in game.mines:
                game.reveal((row, col))
    assert game.status == "won"
    assert game.flagged == game.mines


def test_normalized_click_reaches_the_intended_cell(tmp_path: Path) -> None:
    task = load_task("tasks/minesweeper-example.yaml")
    sandbox = MinesweeperSandbox(task, tmp_path)
    screen = sandbox.screen

    # Right click the center of cell (2, 5) using normalized coordinates
    px = MARGIN + 5 * CELL + CELL // 2
    py = HEADER + 2 * CELL + CELL // 2
    x = px / (screen.width - 1)
    y = py / (screen.height - 1)
    sandbox.execute(action_from_dict({"type": "right_click", "x": x, "y": y}))
    assert screen.game.flagged == {(2, 5)}


def test_hitting_a_mine_ends_the_episode(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        ["run", "--task", "tasks/minesweeper-example.yaml", "--log", str(tmp_path / "e.jsonl")]
    )
    loop, task = build_loop(args)
    # Seed 7 with a first click at (4, 4) puts a mine at (5, 2)
    loop.model._action_payloads = [
        {"type": "click", "x": 0.5, "y": 0.56},
        {"type": "click", "x": 0.3, "y": 0.65},
    ]
    result = loop.run_episode(task)
    assert result.termination_reason is TerminationReason.GAME_OVER
    assert result.evaluation.status is EvaluationStatus.FAILURE
    assert len(list(tmp_path.glob("screenshots/*/step_*.png"))) == 2
