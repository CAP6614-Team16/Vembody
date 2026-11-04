"""Reference policies that bracket VLM performance on Minesweeper.

- ``RandomClickModel``: left or right click anywhere on the screenshot (naive lower bound).
- ``RandomCellModel``: left click a random hidden cell found from screenshot pixels
  (lower bound that cannot miss the grid, but never reasons about numbers).
- ``SolverModel``: single-cell deduction with random guessing when stuck. It reads the
  visible board (revealed numbers and flags) directly instead of the screenshot, so it
  measures how far reasoning alone gets with perfect perception. It never reads mine
  positions.
"""

from __future__ import annotations

import json
import random
from time import perf_counter
from typing import Any

from PIL import Image

from ..minesweeper.screen import BACKGROUND, CELL, HEADER, MARGIN, MinesweeperScreen
from ..observations import Observation
from ..results import StepResult
from ..tasks import TaskSpec
from .base import ModelResponse

Cell = tuple[int, int]


def _response(action: dict[str, Any], reason: str, started: float, provider: str) -> ModelResponse:
    payload = {"reasoning_summary": reason, "action": action, "confidence": None}
    return ModelResponse(
        raw_text=json.dumps(payload),
        elapsed_seconds=perf_counter() - started,
        metadata={"provider": provider},
    )


class _Geometry:
    """Pixel layout of the Minesweeper window for a task's board size."""

    def __init__(self, task: TaskSpec) -> None:
        board = task.minesweeper or {}
        self.rows = board.get("rows", 9)
        self.cols = board.get("cols", 9)
        self.width = 2 * MARGIN + self.cols * CELL
        self.height = HEADER + MARGIN + self.rows * CELL

    def cell_center(self, cell: Cell) -> tuple[float, float]:
        """Normalized coordinates of a cell's center."""
        px = MARGIN + cell[1] * CELL + CELL // 2
        py = HEADER + cell[0] * CELL + CELL // 2
        return px / (self.width - 1), py / (self.height - 1)

    def click(self, cell: Cell, kind: str = "click") -> dict[str, Any]:
        x, y = self.cell_center(cell)
        return {"type": kind, "x": x, "y": y}


class RandomClickModel:
    """Uniformly random left or right click anywhere in the window."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def generate_action(
        self, task: TaskSpec, observation: Observation, history: list[StepResult]
    ) -> ModelResponse:
        started = perf_counter()
        kind = self.rng.choice(["click", "right_click"])
        action = {"type": kind, "x": self.rng.random(), "y": self.rng.random()}
        return _response(action, "Uniform random click.", started, "random")


class RandomCellModel:
    """Left click a random hidden cell, detected from the screenshot."""

    def __init__(self, task: TaskSpec, seed: int | None = None) -> None:
        self.geometry = _Geometry(task)
        self.rng = random.Random(seed)

    def hidden_cells(self, screenshot_path: str) -> list[Cell]:
        """Hidden cells keep the raised gray face; opened cells are lighter."""
        with Image.open(screenshot_path) as image:
            pixels = image.convert("RGB")
            return [
                (row, col)
                for row in range(self.geometry.rows)
                for col in range(self.geometry.cols)
                if pixels.getpixel((MARGIN + col * CELL + 5, HEADER + row * CELL + 5)) == BACKGROUND
            ]

    def generate_action(
        self, task: TaskSpec, observation: Observation, history: list[StepResult]
    ) -> ModelResponse:
        started = perf_counter()
        path = observation.screenshot_path
        hidden = self.hidden_cells(path) if path else []
        if not hidden:
            return _response({"type": "fail"}, "No hidden cell found.", started, "random-cell")
        cell = self.rng.choice(hidden)
        return _response(
            self.geometry.click(cell), f"Random hidden cell {cell}.", started, "random-cell"
        )


class SolverModel:
    """Deduction solver with perfect perception of the visible board."""

    def __init__(self, task: TaskSpec, screen: MinesweeperScreen, seed: int | None = None) -> None:
        self.geometry = _Geometry(task)
        self.screen = screen
        self.rng = random.Random(seed)

    def next_move(self) -> tuple[str, Cell, str]:
        """Return (action type, cell, reason) using only what the screen shows."""
        game = self.screen.game
        if not game.revealed:
            center = (game.rows // 2, game.cols // 2)
            return "click", center, "Opening move in the center."

        hidden = {
            (r, c)
            for r in range(game.rows)
            for c in range(game.cols)
            if (r, c) not in game.revealed and (r, c) not in game.flagged
        }
        for cell in sorted(game.revealed):
            number = game.adjacent_mines(cell)  # the digit drawn on an opened cell
            neighbors = game.neighbors(cell)
            flagged = [n for n in neighbors if n in game.flagged]
            unknown = sorted(n for n in neighbors if n in hidden)
            if not unknown:
                continue
            if number == len(flagged):
                return "click", unknown[0], f"{cell} shows {number} and has all flags placed."
            if number - len(flagged) == len(unknown):
                return "right_click", unknown[0], f"{cell} needs every hidden neighbor flagged."

        cell = self.rng.choice(sorted(hidden))
        return "click", cell, "No safe deduction; guessing."

    def generate_action(
        self, task: TaskSpec, observation: Observation, history: list[StepResult]
    ) -> ModelResponse:
        started = perf_counter()
        kind, cell, reason = self.next_move()
        return _response(self.geometry.click(cell, kind), reason, started, "solver")
