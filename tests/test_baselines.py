import pytest

from vembody.models.baselines import RandomCellModel
from vembody.sandbox import MinesweeperSandbox
from vembody.tasks import load_task


@pytest.mark.parametrize(
    "path",
    [
        "tasks/minesweeper-example.yaml",
        "tasks/benchmark/development/t1-01.yaml",
        "tasks/benchmark/development/t4-01.yaml",
    ],
)
def test_baseline_centers_and_hidden_detection_match_rendered_board(tmp_path, path):
    task = load_task(path)
    sandbox = MinesweeperSandbox(task, tmp_path)
    model = RandomCellModel(task, seed=0)
    screen = sandbox.screen
    if not screen.game.revealed:
        screen.game.reveal((4, 4))
    flagged = next(iter(screen.game.mines))
    screen.game.toggle_flag(flagged)
    screenshot = tmp_path / "board.png"
    screen.screenshot().save(screenshot)

    expected = (
        {(row, col) for row in range(screen.rows) for col in range(screen.cols)}
        - screen.game.revealed
        - screen.game.flagged
    )
    assert set(model.hidden_cells(str(screenshot))) == expected

    for row in range(screen.rows):
        for col in range(screen.cols):
            x, y = model.geometry.cell_center((row, col))
            assert screen.cell_at(
                round(x * (screen.width - 1)), round(y * (screen.height - 1))
            ) == (row, col)
