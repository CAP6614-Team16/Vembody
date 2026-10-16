"""Minesweeper rules: seeded mines placed after a safe first click."""

from __future__ import annotations

import random


class Minesweeper:
    """One board. Cells are (row, col) tuples; status is playing, won, or lost."""

    def __init__(self, rows: int, cols: int, mines: int, seed: int | None = None) -> None:
        self.rows = rows
        self.cols = cols
        self.mine_count = mines
        self.seed = seed
        self.mines: set[tuple[int, int]] = set()
        self.revealed: set[tuple[int, int]] = set()
        self.flagged: set[tuple[int, int]] = set()
        self.exploded: tuple[int, int] | None = None
        self.status = "playing"

    def neighbors(self, cell: tuple[int, int]) -> list[tuple[int, int]]:
        row, col = cell
        return [
            (r, c)
            for r in range(row - 1, row + 2)
            for c in range(col - 1, col + 2)
            if (r, c) != cell and 0 <= r < self.rows and 0 <= c < self.cols
        ]

    def adjacent_mines(self, cell: tuple[int, int]) -> int:
        return sum(neighbor in self.mines for neighbor in self.neighbors(cell))

    def reveal(self, cell: tuple[int, int]) -> None:
        """Open a hidden cell, or chord on a revealed number."""
        if self.status != "playing" or cell in self.flagged:
            return
        if cell in self.revealed:
            self.chord(cell)
            return

        # Place mines on the first reveal so the first click is always safe
        if not self.revealed:
            candidates = [
                (r, c) for r in range(self.rows) for c in range(self.cols) if (r, c) != cell
            ]
            self.mines = set(random.Random(self.seed).sample(candidates, self.mine_count))

        if cell in self.mines:
            self.status = "lost"
            self.exploded = cell
            return

        # Flood-fill outward from cells with no adjacent mines
        stack = [cell]
        while stack:
            current = stack.pop()
            if current in self.revealed or current in self.flagged:
                continue
            self.revealed.add(current)
            if self.adjacent_mines(current) == 0:
                stack.extend(self.neighbors(current))

        # Win once every safe cell is open; flag the remaining mines like the classic game
        if len(self.revealed) == self.rows * self.cols - self.mine_count:
            self.status = "won"
            self.flagged = set(self.mines)

    def chord(self, cell: tuple[int, int]) -> None:
        """Open all unflagged neighbors when the number's flags are all placed."""
        flags = sum(neighbor in self.flagged for neighbor in self.neighbors(cell))
        if flags != self.adjacent_mines(cell):
            return
        for neighbor in self.neighbors(cell):
            if neighbor not in self.revealed:
                self.reveal(neighbor)

    def toggle_flag(self, cell: tuple[int, int]) -> None:
        if self.status != "playing" or cell in self.revealed:
            return
        if cell in self.flagged:
            self.flagged.remove(cell)
        else:
            self.flagged.add(cell)
