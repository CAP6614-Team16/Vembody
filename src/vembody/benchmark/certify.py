"""Solver certificates for the logical templates (T2 and T3).

The selection rules follow docs/benchmark-design.md section 4. Every rule below is
sound, and every certified target is also checked against ``ExactLabelProvider``,
which is the provider that grades the episode. A task is certified only when the
certificate exists, so certified instances always admit a provably correct move.

- T2-01: single-clue propagation (with the public mine count) proves a safe cell.
- T2-02: single-clue propagation proves a mine.
- T2-04: propagation proves no safe cell; subtracting a proper-subset clue from
  another clue proves one.
- T2-03: propagation and the subset rule prove no safe cell; combining two
  overlapping clues proves one.
- T3-01: the outlined clue touches at least one mine, and every such mine is
  provable at the start.
- T3-02: the board is solvable to a win using only provable reveals, and at least
  ten clicks are still needed to win. Provable labels never disappear when more
  cells are opened, so any sequence of provable reveals can continue until the
  board is won, which takes at least ten moves.

Certificates use the hidden layout only to describe the instance (P for T3-01,
the minimum clicks for T3-02); they are evaluator-only and never reach the agent.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from ..minesweeper.game import Minesweeper
from .labels import VisibleBoard
from .solver import PROVIDER_ID, ExactLabelProvider
from .spec import Cell

LOGICAL_TEMPLATES = ("T2-01", "T2-02", "T2-03", "T2-04", "T3-01", "T3-02")
CERTIFIER_ID = "vembody-certify/1"
T3_02_MOVES = 10


@dataclass(frozen=True)
class Clue:
    """An opened number reduced to its undetermined neighbors and remaining mines."""

    at: Cell
    cells: frozenset[Cell]
    mines: int

    def record(self) -> dict:
        return {"clue": list(self.at), "cells": _cells(self.cells), "mines": self.mines}


def _cells(cells) -> list[list[int]]:
    return [list(c) for c in sorted(cells)]


def board_of(game: Minesweeper, trust_flags: bool = False) -> VisibleBoard:
    """The visible board of a game; flags are shown as hidden unless trusted."""

    def symbol(cell: Cell) -> str:
        if cell in game.flagged and trust_flags:
            return "F"
        if cell not in game.revealed:
            return "#"
        number = game.adjacent_mines(cell)
        return str(number) if number else "."

    return VisibleBoard(
        game.rows,
        game.cols,
        game.mine_count,
        tuple(tuple(symbol((r, c)) for c in range(game.cols)) for r in range(game.rows)),
    )


def _clues(board: VisibleBoard) -> list[Clue]:
    hidden = board.hidden
    clues = []
    for r in range(board.rows):
        for c in range(board.cols):
            symbol = board.cells[r][c]
            if symbol in ("#", "F"):
                continue
            around = frozenset(
                (rr, cc)
                for rr in range(r - 1, r + 2)
                for cc in range(c - 1, c + 2)
                if (rr, cc) in hidden
            )
            if around:
                clues.append(Clue((r, c), around, 0 if symbol == "." else int(symbol)))
    return clues


def propagate(board: VisibleBoard) -> tuple[set[Cell], set[Cell], list[Clue]]:
    """Single-clue rules and the mine-count rule, repeated to a fixed point.

    Returns the safe cells, the mines, and each clue reduced by what was proved.
    """
    hidden = board.hidden
    safe: set[Cell] = set()
    mines: set[Cell] = set()
    clues = _clues(board)
    changed = True
    while changed:
        changed = False
        groups = [(clue.cells, clue.mines) for clue in clues]
        groups.append((hidden, board.total_mines))
        for cells, number in groups:
            open_cells = cells - safe - mines
            left = number - len(cells & mines)
            if open_cells and left == 0:
                safe |= open_cells
                changed = True
            elif open_cells and left == len(open_cells):
                mines |= open_cells
                changed = True
    reduced = [
        Clue(clue.at, clue.cells - safe - mines, clue.mines - len(clue.cells & mines))
        for clue in clues
    ]
    return safe, mines, [clue for clue in reduced if clue.cells]


def subset_deductions(clues: list[Clue]) -> list[tuple[Clue, Clue, frozenset[Cell]]]:
    """Proper-subset pairs (A inside B, same mine count) and the safe cells B - A."""
    found = []
    for small in clues:
        for large in clues:
            if small.cells < large.cells and small.mines == large.mines:
                found.append((small, large, large.cells - small.cells))
    return found


def pair_deductions(clues: list[Clue]) -> list[tuple[Clue, Clue, frozenset[Cell]]]:
    """Safe cells proved by two overlapping clues taken together.

    With k mines in the overlap, A needs a - k mines in A - B and B needs b - k in
    B - A. A part is safe when every feasible k leaves it no mines.
    """
    found = []
    for i, first in enumerate(clues):
        for second in clues[i + 1 :]:
            both = first.cells & second.cells
            if not both:
                continue
            only_first, only_second = first.cells - both, second.cells - both
            feasible = [
                k
                for k in range(len(both) + 1)
                if 0 <= first.mines - k <= len(only_first)
                and 0 <= second.mines - k <= len(only_second)
            ]
            safe: set[Cell] = set()
            if feasible and max(feasible) == 0:
                safe |= both
            if feasible and all(first.mines - k == 0 for k in feasible):
                safe |= only_first
            if feasible and all(second.mines - k == 0 for k in feasible):
                safe |= only_second
            if safe:
                found.append((first, second, frozenset(safe)))
    return found


def min_clicks_to_win(game: Minesweeper) -> int:
    """Fewest reveals that still win the game (3BV of the unopened part)."""
    opened = set(game.revealed)
    clicks = 0
    for r in range(game.rows):
        for c in range(game.cols):
            start = (r, c)
            if start in opened or start in game.mines or game.adjacent_mines(start):
                continue
            clicks += 1
            stack = [start]
            while stack:
                cell = stack.pop()
                if cell in opened:
                    continue
                opened.add(cell)
                if not game.adjacent_mines(cell):
                    stack.extend(n for n in game.neighbors(cell) if n not in game.mines)
    left = game.rows * game.cols - game.mine_count - len(opened)
    return clicks + left


def guess_free_trace(game: Minesweeper, provider: ExactLabelProvider) -> tuple[list[Cell], bool]:
    """Reveal the first provably safe cell until the game is won or stuck."""
    game = copy.deepcopy(game)
    game.flagged.clear()
    trace: list[Cell] = []
    while game.status == "playing":
        safe = provider.label(board_of(game)).safe
        if not safe:
            return trace, False
        target = min(safe)
        game.reveal(target)
        trace.append(target)
    return trace, game.status == "won"


def certify(template: str, game: Minesweeper, region: tuple[Cell, ...] = ()) -> dict | None:
    """Return a certificate if the start state meets the template's rule, else None.

    ``game`` must be the prepared start state (opening and setup applied, no flags).
    """
    if template not in LOGICAL_TEMPLATES:
        raise ValueError(f"No logical certificate for {template}")
    if game.status != "playing" or game.flagged:
        return None
    provider = ExactLabelProvider()
    board = board_of(game)
    labels = provider.label(board)
    record = {
        "certifier": CERTIFIER_ID,
        "label_provider": PROVIDER_ID,
        "provable_safe": _cells(labels.safe),
        "provable_mines": _cells(labels.mines),
    }
    if template.startswith("T2"):
        safe, mines, reduced = propagate(board)
        if template == "T2-01" and safe:
            proof = {"rule": "single_clue", "targets": _cells(safe)}
        elif template == "T2-02" and mines:
            proof = {"rule": "single_clue", "targets": _cells(mines)}
        elif template in ("T2-03", "T2-04") and not safe:
            subsets = subset_deductions(reduced)
            if template == "T2-04" and subsets:
                small, large, cells = subsets[0]
                proof = {
                    "rule": "subset",
                    "subset": small.record(),
                    "superset": large.record(),
                    "targets": _cells(cells),
                }
            elif template == "T2-03" and not subsets and (pairs := pair_deductions(reduced)):
                first, second, cells = pairs[0]
                proof = {
                    "rule": "two_clues",
                    "clues": [first.record(), second.record()],
                    "targets": _cells(cells),
                }
            else:
                return None
        else:
            return None
        proved = {tuple(c) for c in proof["targets"]}
        expected = labels.mines if template == "T2-02" else labels.safe
        if not proved <= expected:  # a sound rule can never disagree with the exact solver
            raise AssertionError(f"{template} proof disagrees with the exact solver")
        return record | proof
    if template == "T3-01":
        targets = {n for cell in region for n in game.neighbors(cell) if n not in game.revealed}
        mines = targets & game.mines
        if not mines or not mines <= labels.mines:
            return None
        return record | {
            "rule": "region_mines_provable",
            "region": _cells(region),
            "targets": _cells(targets),
            "mines": _cells(mines),
        }
    clicks = min_clicks_to_win(game)
    if clicks < T3_02_MOVES or not labels.safe:
        return None
    trace, won = guess_free_trace(game, provider)
    if not won:
        return None
    return record | {
        "rule": "guess_free_win",
        "min_clicks_to_win": clicks,
        "reference_trace": [list(c) for c in trace],
    }
