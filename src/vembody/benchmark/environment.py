"""Deterministic preparation and private snapshots of benchmark instances."""

from __future__ import annotations

import hashlib
import json

from ..errors import TaskLoadError
from ..minesweeper.screen import MinesweeperScreen
from ..tasks import TaskSpec


def snapshot(screen: MinesweeperScreen) -> dict:
    game = screen.game
    return {
        "rules_version": 1,
        "rows": game.rows,
        "cols": game.cols,
        "mine_count": game.mine_count,
        "mines": sorted(game.mines),
        "revealed": sorted(game.revealed),
        "flagged": sorted(game.flagged),
        "exploded": game.exploded,
        "status": game.status,
    }


def state_hash(screen: MinesweeperScreen) -> str:
    return hashlib.sha256(
        json.dumps(snapshot(screen), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def prepare(task: TaskSpec) -> MinesweeperScreen:
    """Create a fresh screen; no state from a previous episode is reused."""
    spec = task.benchmark
    if spec is None:
        raise TaskLoadError("Missing benchmark configuration")
    screen = MinesweeperScreen(
        **task.minesweeper,
        benchmark=True,
        grid_labels=spec.grid_labels,
        highlight=spec.predicate.region,
    )
    screen.game.reveal(spec.opening)
    for kind, target in spec.setup:
        if screen.game.status != "playing":
            raise TaskLoadError("Setup reached a terminal state")
        px, py = screen.cell_center(target)
        screen.click(px, py, "left" if kind == "click" else "right")
    game, p = screen.game, spec.predicate
    if game.status != "playing":
        raise TaskLoadError("Benchmark must start in a playing state")
    if p.target is not None:
        if p.target in game.revealed or p.target in game.flagged:
            raise TaskLoadError("Task target must start hidden and unflagged")
        if p.type == "exact_reveal":
            if p.target in game.mines or game.adjacent_mines(p.target) == 0:
                raise TaskLoadError("Grounding reveal target must be safe and numbered")
            if len(game.revealed) + 1 == game.rows * game.cols - game.mine_count:
                raise TaskLoadError("Grounding target must not end the game")
        elif p.target not in game.mines:
            raise TaskLoadError("Grounding flag target must be a mine")
    expected = {"T1-01": (2, 4), "T1-02": (6, 1), "T1-04": (0, game.cols - 1)}
    if spec.template_id in expected and p.target != expected[spec.template_id]:
        raise TaskLoadError("Target does not match the template instruction")
    if spec.template_id == "T1-03":
        clues = [c for c in game.revealed if game.adjacent_mines(c) == 3]
        if len(clues) != 1 or p.target != (clues[0][0], clues[0][1] + 1):
            raise TaskLoadError("T1-03 requires a unique 3 immediately left of the target")
    if spec.template_id.startswith("T2") and game.flagged:
        raise TaskLoadError("Single-deduction tasks must start without flags")
    if p.region:
        if any(c not in game.revealed or game.adjacent_mines(c) == 0 for c in p.region):
            raise TaskLoadError("Region must consist of revealed numbered cells")
        reached = {p.region[0]}
        while True:
            expanded = reached | {c for c in p.region if set(game.neighbors(c)) & reached}
            if expanded == reached:
                break
            reached = expanded
        if reached != set(p.region):
            raise TaskLoadError("Region must be connected")
        targets = {n for c in p.region for n in game.neighbors(c)} & game.mines
        if not targets or targets & game.flagged:
            raise TaskLoadError("Region needs unflagged mine targets")
    if spec.expected_state_hash and state_hash(screen) != spec.expected_state_hash:
        raise TaskLoadError("Frozen starting state does not match this runtime")
    return screen
