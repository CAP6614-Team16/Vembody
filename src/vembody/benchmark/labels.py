"""Solver handoff: this module defines a contract, not a logical solver."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol

from ..minesweeper.screen import MinesweeperScreen
from .spec import Cell


class CertificationRequired(ValueError):
    """A task cannot be scored without trustworthy logical evidence."""


@dataclass(frozen=True)
class VisibleBoard:
    rows: int
    cols: int
    total_mines: int
    cells: tuple[tuple[str, ...], ...]

    @property
    def hidden(self) -> frozenset[Cell]:
        return frozenset(
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if self.cells[r][c] in ("#", "F")
        )


@dataclass(frozen=True)
class LogicalLabels:
    safe: frozenset[Cell]
    mines: frozenset[Cell]
    provider: str

    def validate(self, board: VisibleBoard) -> None:
        if (
            not isinstance(self.safe, frozenset)
            or not isinstance(self.mines, frozenset)
            or not isinstance(self.provider, str)
            or not self.provider.strip()
        ):
            raise CertificationRequired("Malformed logical label record")
        if any(
            not isinstance(c, tuple) or len(c) != 2 or any(type(v) is not int for v in c)
            for c in self.safe | self.mines
        ):
            raise CertificationRequired("Malformed logical label cells")
        if not self.provider or self.safe & self.mines or (self.safe | self.mines) - board.hidden:
            raise CertificationRequired("Invalid logical labels from provider")


class LabelProvider(Protocol):
    def label(self, board: VisibleBoard) -> LogicalLabels:
        """Prove labels from visible clues/total only. Flags are not trusted mines.

        A timeout or incomplete reasoning must raise CertificationRequired, never
        return an invented safe/mine label. Unlabeled hidden cells are undetermined.
        Implementations and proof/certificate review belong to the evaluation lead.
        """
        ...


def visible_board(screen: MinesweeperScreen) -> VisibleBoard:
    game = screen.game
    return VisibleBoard(
        game.rows,
        game.cols,
        game.mine_count,
        tuple(
            tuple(
                "F"
                if (r, c) in game.flagged
                else (str(game.adjacent_mines((r, c))) if game.adjacent_mines((r, c)) else ".")
                if (r, c) in game.revealed
                else "#"
                for c in range(game.cols)
            )
            for r in range(game.rows)
        ),
    )


def visible_hash(screen: MinesweeperScreen) -> str:
    """Identity of the visible input, excluding hidden layouts and task IDs."""
    board = visible_board(screen)
    return hashlib.sha256(
        json.dumps(
            {
                "rows": board.rows,
                "cols": board.cols,
                "total_mines": board.total_mines,
                "cells": board.cells,
                "highlight": sorted(screen.highlight),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
