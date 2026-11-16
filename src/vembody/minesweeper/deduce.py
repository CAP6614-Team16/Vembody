"""Single-cell Minesweeper deductions from a visible board alone.

Cells use the transcription symbols: ``#`` hidden, ``F`` flagged, ``.`` blank, or a digit.
Flags count as mines.
"""

from __future__ import annotations

import copy

from .game import Minesweeper

Cell = tuple[int, int]


def cell_name(cell: Cell) -> str:
    """Grid label as drawn on the screenshot: row letter, then 1-based column."""
    return f"{chr(65 + cell[0])}{cell[1] + 1}"


def _neighbors(board, cell: Cell) -> list[Cell]:
    row, col = cell
    return [
        (r, c)
        for r in range(row - 1, row + 2)
        for c in range(col - 1, col + 2)
        if (r, c) != cell and 0 <= r < len(board) and 0 <= c < len(board[0])
    ]


def deductions(board) -> dict[Cell, tuple[str, str]]:
    """Map each provable hidden cell to (action type, reason naming the clue)."""
    found: dict[Cell, tuple[str, str]] = {}
    for r, row in enumerate(board):
        for c, symbol in enumerate(row):
            if not symbol.isdigit():
                continue
            number = int(symbol)
            near = _neighbors(board, (r, c))
            flags = [n for n in near if board[n[0]][n[1]] == "F"]
            hidden = [n for n in near if board[n[0]][n[1]] == "#"]
            if not hidden:
                continue
            listed = " ".join(f"{cell_name(n)} {board[n[0]][n[1]]}" for n in near)
            counts = f"{cell_name((r, c))}={number}: {listed} -> {len(flags)} flag(s), "
            counts += f"{len(hidden)} hidden"
            if number == len(flags):
                reason = f"{counts}; all {number} mine(s) flagged, so the hidden ones are safe"
                for cell in hidden:
                    found.setdefault(cell, ("click", reason))
            elif number - len(flags) == len(hidden):
                reason = (
                    f"{counts}; {number} - {len(flags)} = {len(hidden)} mine(s) left for "
                    f"{len(hidden)} hidden, so the hidden ones are mines"
                )
                for cell in hidden:
                    found.setdefault(cell, ("right_click", reason))
    return found


def visible_board(game: Minesweeper) -> list[list[str]]:
    """The board as a player sees it, in the transcription symbols."""

    def symbol(cell: Cell) -> str:
        if cell in game.flagged:
            return "F"
        if cell not in game.revealed:
            return "#"
        return str(game.adjacent_mines(cell) or ".")

    return [[symbol((r, c)) for c in range(game.cols)] for r in range(game.rows)]


def solve_by_deduction(game: Minesweeper) -> tuple[Minesweeper, int]:
    """Play a copy using only provable moves (single clues plus the mine count).

    Returns the copy, won or stuck where a guess is needed, and the number of reveals.
    """
    game = copy.deepcopy(game)
    reveals = 0
    while game.status == "playing":
        board = visible_board(game)
        moves = deductions(board)
        if not moves:
            hidden = [(r, c) for r, row in enumerate(board) for c, s in enumerate(row) if s == "#"]
            left = game.mine_count - len(game.flagged)
            if hidden and left == 0:
                moves = dict.fromkeys(hidden, ("click", ""))
            elif hidden and left == len(hidden):
                moves = dict.fromkeys(hidden, ("right_click", ""))
            else:
                break
        for cell, (kind, _) in moves.items():
            if kind == "right_click":
                game.toggle_flag(cell)
            elif cell not in game.revealed:
                game.reveal(cell)
                reveals += 1
    return game, reveals
