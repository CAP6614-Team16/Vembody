"""Episode metrics and aggregate statistics.

Every function here takes plain JSON data, the same shape the JSONL logger writes for an
``episode`` record, so metrics can be recomputed from old logs without rerunning anything.
Definitions are documented in ``docs/evaluation.md``.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter
from datetime import datetime
from typing import Any

TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens", "image_count")


def _percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile; None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _is_invalid(step: dict[str, Any]) -> bool:
    return bool(step.get("parse_error") or step.get("validation_error"))


def _is_ineffective(step: dict[str, Any]) -> bool:
    """An executed action that did not change the board."""
    execution = step.get("execution") or {}
    details = execution.get("details") or {}
    return bool(execution.get("succeeded")) and details.get("board_changed") is False


def _is_effective(step: dict[str, Any]) -> bool:
    execution = step.get("execution") or {}
    details = execution.get("details") or {}
    return bool(execution.get("succeeded")) and details.get("board_changed") is True


def recovery_counts(steps: list[dict[str, Any]]) -> tuple[int, int]:
    """Count (setbacks, recoveries).

    A setback is a step whose action was invalid or executed without changing the board.
    It is recovered when the very next step changes the board. A setback on the final
    step cannot be recovered and is still counted.
    """
    setbacks = recoveries = 0
    for index, step in enumerate(steps):
        if not (_is_invalid(step) or _is_ineffective(step)):
            continue
        setbacks += 1
        if index + 1 < len(steps) and _is_effective(steps[index + 1]):
            recoveries += 1
    return setbacks, recoveries


def _final_board(steps: list[dict[str, Any]]) -> dict[str, Any] | None:
    for step in reversed(steps):
        board = ((step.get("execution") or {}).get("details") or {}).get("board")
        if board:
            return board
    return None


def _wall_seconds(episode: dict[str, Any]) -> float | None:
    try:
        started = datetime.fromisoformat(episode["started_at"])
        ended = datetime.fromisoformat(episode["ended_at"])
    except (KeyError, TypeError, ValueError):
        return None
    return (ended - started).total_seconds()


def episode_metrics(
    episode: dict[str, Any], final_board: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Compute one flat row of metrics for an episode record.

    ``final_board`` overrides the board stats from the last executed step; the harness
    passes the sandbox's state after the episode so the row is right even when no
    action executed.
    """
    steps = episode.get("steps", [])
    executed = [s for s in steps if (s.get("execution") or {}).get("succeeded")]
    details = [(s.get("execution") or {}).get("details") or {} for s in executed]
    latencies = [float(s.get("model_elapsed_seconds") or 0.0) for s in steps]
    invalid = sum(_is_invalid(s) for s in steps)
    setbacks, recoveries = recovery_counts(steps)
    status = (episode.get("evaluation") or {}).get("status")

    row: dict[str, Any] = {
        "run_id": episode.get("run_id"),
        "task_id": episode.get("task_id"),
        "termination_reason": episode.get("termination_reason"),
        "evaluation_status": status,
        "success": status == "success",
        "steps": len(steps),
        "actions_executed": len(executed),
        "invalid_actions": invalid,
        "invalid_action_rate": invalid / len(steps) if steps else None,
        "ineffective_actions": sum(_is_ineffective(s) for s in steps),
        "off_grid_clicks": sum(d.get("target") == "none" for d in details),
        "resets": sum(bool(d.get("reset")) for d in details),
        "setbacks": setbacks,
        "recoveries": recoveries,
        "recovery_rate": recoveries / setbacks if setbacks else None,
        "wall_seconds": _wall_seconds(episode),
        "model_seconds_total": sum(latencies),
        "latency_mean": statistics.fmean(latencies) if latencies else None,
        "latency_median": _percentile(latencies, 0.5),
        "latency_p95": _percentile(latencies, 0.95),
    }

    for key in TOKEN_KEYS:
        values = [s.get("model_metadata", {}).get(key) for s in steps]
        values = [v for v in values if isinstance(v, (int, float))]
        row[f"{key}_total"] = sum(values) if values else None
        row[f"{key}_per_step"] = sum(values) / len(values) if values else None

    board = final_board or _final_board(steps)
    if board:
        safe_total = board.get("safe_total") or 0
        row.update(
            final_status=board.get("status"),
            safe_revealed=board.get("safe_revealed"),
            safe_total=safe_total,
            progress=board.get("safe_revealed", 0) / safe_total if safe_total else None,
            flags_correct=board.get("flags_correct"),
            flags_wrong=board.get("flags_wrong"),
            mine_hit=board.get("status") == "lost",
        )
    return row


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion; stable for small n and 0 or n hits."""
    if total == 0:
        return (math.nan, math.nan)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def bootstrap_mean_interval(
    values: list[float], resamples: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95% percentile-bootstrap interval for a mean, with a fixed seed for reproducibility."""
    if not values:
        return (math.nan, math.nan)
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(resamples))
    return (_percentile(means, 0.025), _percentile(means, 0.975))


def _mean(values: list[Any]) -> float | None:
    numbers = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return statistics.fmean(numbers) if numbers else None


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize a group of episode rows (one configuration, or one configuration x task)."""
    n = len(rows)
    wins = sum(bool(r.get("success")) for r in rows)
    progress = [r["progress"] for r in rows if isinstance(r.get("progress"), (int, float))]
    all_latencies = [r["latency_median"] for r in rows if r.get("latency_median") is not None]
    steps_total = sum(r.get("steps", 0) for r in rows)
    setbacks = sum(r.get("setbacks", 0) for r in rows)
    success_low, success_high = wilson_interval(wins, n)
    progress_low, progress_high = bootstrap_mean_interval(progress)
    return {
        "episodes": n,
        "success_rate": wins / n if n else None,
        "success_ci_low": success_low,
        "success_ci_high": success_high,
        "progress_mean": _mean(progress),
        "progress_ci_low": progress_low,
        "progress_ci_high": progress_high,
        "mine_hit_rate": _mean([float(bool(r.get("mine_hit"))) for r in rows]),
        "steps_mean": _mean([r.get("steps") for r in rows]),
        "wall_seconds_mean": _mean([r.get("wall_seconds") for r in rows]),
        # Pooled over steps so long episodes weigh more, matching "per action" wording
        "invalid_action_rate": (
            sum(r.get("invalid_actions", 0) for r in rows) / steps_total if steps_total else None
        ),
        "ineffective_action_rate": (
            sum(r.get("ineffective_actions", 0) for r in rows) / steps_total
            if steps_total
            else None
        ),
        "recovery_rate": (
            sum(r.get("recoveries", 0) for r in rows) / setbacks if setbacks else None
        ),
        "latency_median_mean": _mean(all_latencies),
        "latency_p95_mean": _mean([r.get("latency_p95") for r in rows]),
        "prompt_tokens_per_step_mean": _mean([r.get("prompt_tokens_per_step") for r in rows]),
        "termination_reasons": dict(Counter(r.get("termination_reason") for r in rows)),
    }
