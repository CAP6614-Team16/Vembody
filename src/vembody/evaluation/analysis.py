"""Turn experiment folders into result tables and plots.

``analyze(["runs/baseline", "runs/full"], "results/")`` writes:

- ``episodes.csv``: every episode row from every experiment
- ``summary.csv`` / ``summary.md``: one row per experiment
- ``by_task.csv``: one row per experiment x task
- ``success_rate.png``, ``progress.png``, ``latency.png`` (when matplotlib is installed)
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .metrics import aggregate

# Single-series bar color: slot 1 of the reference categorical palette
BAR_COLOR = "#2a78d6"
INK = "#52514e"
GRID = "#e4e3df"


def load_episodes(experiment_dirs: Sequence[str | Path]) -> list[dict[str, Any]]:
    """Read episodes.jsonl from each experiment folder."""
    rows: list[dict[str, Any]] = []
    for directory in experiment_dirs:
        path = Path(directory) / "episodes.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"No episodes.jsonl in {directory}")
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def _group(rows: list[dict[str, Any]], *keys: str) -> dict[tuple, list[dict[str, Any]]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in keys)].append(row)
    return dict(groups)


def _cell(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        columns.extend(k for k in row if k not in columns)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: _cell(v) for k, v in row.items()} for row in rows)


def _fmt(value: Any, percent: bool = False) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "–"
    if percent:
        return f"{100 * value:.1f}%"
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def summary_markdown(summary: list[dict[str, Any]]) -> str:
    """Compact Markdown table for the README or a status update."""
    lines = [
        "| Experiment | Model | Scored | Infra errors | Success (95% CI) | Progress (95% CI) | "
        "Mine hit | Invalid | Recovery | Median latency (s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in summary:
        success = (
            f"{_fmt(row['success_rate'], True)} "
            f"[{_fmt(row['success_ci_low'], True)}, {_fmt(row['success_ci_high'], True)}]"
        )
        progress = (
            f"{_fmt(row['progress_mean'], True)} "
            f"[{_fmt(row['progress_ci_low'], True)}, {_fmt(row['progress_ci_high'], True)}]"
        )
        lines.append(
            f"| {row['experiment']} | {row['model']} | {row['scored_episodes']} | "
            f"{row['infrastructure_failures']} | {success} | "
            f"{progress} | {_fmt(row['mine_hit_rate'], True)} | "
            f"{_fmt(row['invalid_action_rate'], True)} | {_fmt(row['recovery_rate'], True)} | "
            f"{_fmt(row['latency_median_mean'])} |"
        )
    return "\n".join(lines) + "\n"


def _missing(value: float | None) -> bool:
    return value is None or value != value


def _plot_bars(
    path: Path,
    labels: list[str],
    values: list[float | None],
    lows: list[float | None] | None,
    highs: list[float | None] | None,
    title: str,
    xlabel: str,
    percent: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clean = [v if v is not None else 0.0 for v in values]
    figure, axis = plt.subplots(figsize=(6.4, 0.5 * len(labels) + 1.4), dpi=200)
    positions = range(len(labels))
    errors = None
    label_x = list(clean)
    if lows is not None and highs is not None:
        low_ends = [v if _missing(lo) else lo for v, lo in zip(clean, lows, strict=True)]
        high_ends = [v if _missing(hi) else hi for v, hi in zip(clean, highs, strict=True)]
        label_x = [max(v, hi) for v, hi in zip(clean, high_ends, strict=True)]
        errors = [
            [max(0.0, v - lo) for v, lo in zip(clean, low_ends, strict=True)],
            [max(0.0, hi - v) for v, hi in zip(clean, high_ends, strict=True)],
        ]
    axis.barh(positions, clean, height=0.55, color=BAR_COLOR, zorder=2)
    if errors is not None:
        axis.errorbar(
            clean,
            positions,
            xerr=errors,
            fmt="none",
            ecolor=INK,
            elinewidth=1.2,
            capsize=3,
            zorder=3,
        )
    # Labels sit just past the interval's upper end so they never cross an error bar
    for position, value, raw, x in zip(positions, clean, values, label_x, strict=True):
        text = "–" if raw is None else (f"{100 * value:.0f}%" if percent else f"{value:.2f}")
        axis.annotate(
            text,
            (x, position),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK,
        )
    axis.set_yticks(list(positions), labels, fontsize=9, color=INK)
    axis.invert_yaxis()
    if percent:
        axis.set_xlim(0, 1.12)
        axis.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axis.set_xlabel(xlabel, fontsize=9, color=INK)
    axis.set_title(title, fontsize=11, loc="left", color="#0b0b0b")
    axis.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    axis.tick_params(colors=INK, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(GRID)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def plot_summary(summary: list[dict[str, Any]], out: Path) -> list[Path]:
    """Write the standard comparison plots; returns the files written."""
    labels = [str(row["experiment"]) for row in summary]
    written = []
    specs = [
        (
            "success_rate.png",
            "success_rate",
            "success_ci",
            "Task success by configuration",
            "Tasks completed (95% Wilson CI)",
            True,
        ),
        (
            "progress.png",
            "progress_mean",
            "progress_ci",
            "Board progress by configuration",
            "Safe cells revealed, mean (95% bootstrap CI)",
            True,
        ),
        (
            "latency.png",
            "latency_median_mean",
            None,
            "Model latency by configuration",
            "Median seconds per model call",
            False,
        ),
    ]
    for filename, key, ci, title, xlabel, percent in specs:
        lows = [row.get(f"{ci}_low") for row in summary] if ci else None
        highs = [row.get(f"{ci}_high") for row in summary] if ci else None
        path = out / filename
        _plot_bars(
            path, labels, [row.get(key) for row in summary], lows, highs, title, xlabel, percent
        )
        written.append(path)
    return written


def analyze(experiment_dirs: Sequence[str | Path], out_dir: str | Path) -> dict[str, Any]:
    """Aggregate experiments into CSV/Markdown tables and plots."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_episodes(experiment_dirs)

    summary = [
        {"experiment": experiment, "model": model, **aggregate(group)}
        for (experiment, model), group in _group(rows, "experiment", "model").items()
    ]
    by_task = [
        {"experiment": experiment, "task_id": task_id, **aggregate(group)}
        for (experiment, task_id), group in sorted(
            _group(rows, "experiment", "task_id").items(), key=lambda item: str(item[0])
        )
    ]

    _write_csv(out / "episodes.csv", rows)
    _write_csv(out / "summary.csv", summary)
    _write_csv(out / "by_task.csv", by_task)
    (out / "summary.md").write_text(summary_markdown(summary), encoding="utf-8")

    plots: list[Path] = []
    try:
        plots = plot_summary(summary, out)
    except ImportError:
        pass
    return {"summary": summary, "by_task": by_task, "plots": plots}
