"""Validated benchmark configuration; never serialize this into model prompts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..errors import TaskLoadError

Cell = tuple[int, int]
TEMPLATES = {
    "T1-01": "exact_reveal",
    "T1-02": "exact_flag",
    "T1-03": "exact_reveal",
    "T1-04": "exact_reveal",
    "T2-01": "one_provable_reveal",
    "T2-02": "one_provable_flag",
    "T2-03": "one_provable_reveal",
    "T2-04": "one_provable_reveal",
    "T3-01": "flag_region_mines",
    "T3-02": "safe_reveal_count",
    "T4-01": "board_won",
    "T4-02": "board_won",
}


def exact_keys(data: dict, allowed: set[str], context: str) -> None:
    if any(not isinstance(key, str) for key in data):
        raise TaskLoadError(f"{context} field names must be strings")
    if set(data) - allowed:
        raise TaskLoadError(f"Unknown {context} fields: {sorted(set(data) - allowed)}")


def cell(value: Any, rows: int, cols: int) -> Cell:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(type(v) is not int for v in value)
        or not 0 <= value[0] < rows
        or not 0 <= value[1] < cols
    ):
        raise TaskLoadError(f"Invalid cell: {value!r}")
    return tuple(value)


def board_settings(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        raise TaskLoadError("minesweeper must be an object")
    exact_keys(value, {"rows", "cols", "mines", "seed"}, "board")
    if set(value) != {"rows", "cols", "mines", "seed"}:
        raise TaskLoadError("Board requires rows, cols, mines, seed")
    if any(type(v) is not int for v in value.values()):
        raise TaskLoadError("Board values must be integers")
    if not (
        2 <= value["rows"] <= 26
        and 2 <= value["cols"] <= 26
        and 1 <= value["mines"] < value["rows"] * value["cols"]
    ):
        raise TaskLoadError("Invalid board dimensions or mine count")
    return dict(value)


@dataclass(frozen=True)
class Predicate:
    type: str
    target: Cell | None = None
    region: tuple[Cell, ...] = ()
    count: int = 0


@dataclass(frozen=True)
class BenchmarkSpec:
    version: int
    template_id: str
    split: str
    opening: Cell
    setup: tuple[tuple[str, Cell], ...]
    predicate: Predicate
    expected_state_hash: str | None = None
    grid_labels: bool = False


def parse_benchmark(value: Any, board: dict) -> BenchmarkSpec:
    if not isinstance(value, dict):
        raise TaskLoadError("benchmark must be an object")
    exact_keys(
        value,
        {
            "version",
            "template_id",
            "split",
            "opening",
            "setup",
            "predicate",
            "expected_state_hash",
            "grid_labels",
        },
        "benchmark",
    )
    template = value.get("template_id")
    if type(value.get("version")) is not int or value["version"] != 1:
        raise TaskLoadError("Only benchmark version 1 is supported")
    if not isinstance(template, str) or template not in TEMPLATES:
        raise TaskLoadError("Unknown benchmark template")
    if value.get("split") not in ("development", "demonstration", "evaluation"):
        raise TaskLoadError("Unknown benchmark split")
    rows, cols = board["rows"], board["cols"]
    opening = cell(value.get("opening"), rows, cols)
    setup = value.get("setup", [])
    if not isinstance(setup, list):
        raise TaskLoadError("setup must be an ordered list")
    moves = []
    for move in setup:
        if not isinstance(move, dict):
            raise TaskLoadError("setup moves must be objects")
        exact_keys(move, {"type", "cell"}, "setup")
        if move.get("type") not in ("click", "right_click"):
            raise TaskLoadError("Unsupported setup action")
        moves.append((move["type"], cell(move.get("cell"), rows, cols)))
    p = value.get("predicate")
    if not isinstance(p, dict) or p.get("type") != TEMPLATES[template]:
        raise TaskLoadError("Predicate must match template")
    kind = p["type"]
    fields = {"type"}
    target, region, count = None, (), 0
    if kind in ("exact_reveal", "exact_flag"):
        fields.add("target")
        target = cell(p.get("target"), rows, cols)
    elif kind == "flag_region_mines":
        fields.add("region")
        raw = p.get("region")
        if not isinstance(raw, list) or not raw:
            raise TaskLoadError("A nonempty region is required")
        region = tuple(cell(c, rows, cols) for c in raw)
        if len(set(region)) != len(region):
            raise TaskLoadError("Region cells must be unique")
    elif kind == "safe_reveal_count":
        fields.add("count")
        count = p.get("count")
        if type(count) is not int or count != 10:
            raise TaskLoadError("T3-02 requires ten safe reveals")
    exact_keys(p, fields, "predicate")
    digest = value.get("expected_state_hash")
    if digest is not None and (
        not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
    ):
        raise TaskLoadError("expected_state_hash must be a SHA256 hex digest")
    if value["split"] == "evaluation" and digest is None:
        raise TaskLoadError("Evaluation instances require a frozen state hash")
    labels = value.get("grid_labels", False)
    if type(labels) is not bool:
        raise TaskLoadError("grid_labels must be boolean")
    return BenchmarkSpec(
        1,
        template,
        value["split"],
        opening,
        tuple(moves),
        Predicate(kind, target, region, count),
        digest,
        labels,
    )
