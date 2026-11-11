import math

import pytest

from vembody.evaluation.metrics import (
    aggregate,
    bootstrap_mean_interval,
    episode_metrics,
    recovery_counts,
    wilson_interval,
)


def _step(changed=None, invalid=False, latency=1.0, tokens=None, target="cell"):
    step = {
        "action": {"type": "click", "x": 0.5, "y": 0.5},
        "model_elapsed_seconds": latency,
        "model_metadata": {},
    }
    if tokens is not None:
        step["model_metadata"]["prompt_tokens"] = tokens
    if invalid:
        step["parse_error"] = "bad json"
        return step
    step["execution"] = {
        "succeeded": True,
        "details": {
            "board_changed": changed,
            "target": target,
            "reset": False,
            "board": {"status": "playing", "safe_total": 10, "safe_revealed": 4},
        },
    }
    return step


def test_wilson_interval_handles_extremes() -> None:
    low, high = wilson_interval(0, 20)
    assert low == 0.0 and 0.15 < high < 0.17
    low, high = wilson_interval(20, 20)
    assert high == 1.0 and 0.83 < low < 0.85
    assert all(math.isnan(v) for v in wilson_interval(0, 0))


def test_bootstrap_is_reproducible_and_brackets_mean() -> None:
    values = [0.1, 0.4, 0.5, 0.9]
    first = bootstrap_mean_interval(values)
    assert first == bootstrap_mean_interval(values)
    assert first[0] <= sum(values) / len(values) <= first[1]


def test_recovery_counts_setbacks_followed_by_progress() -> None:
    steps = [_step(True), _step(False), _step(True), _step(False), _step(False)]
    # Setbacks at 1, 3, 4; only step 1 is followed by a board change
    assert recovery_counts(steps) == (3, 1)


def test_episode_metrics_row() -> None:
    episode = {
        "run_id": "r",
        "task_id": "t",
        "started_at": "2026-01-01T00:00:00+00:00",
        "ended_at": "2026-01-01T00:00:03+00:00",
        "termination_reason": "action_limit",
        "evaluation": {"status": "not_evaluated"},
        "steps": [
            _step(True, latency=1.0, tokens=100),
            _step(False, latency=3.0, tokens=300, target="none"),
            _step(invalid=True, latency=2.0),
        ],
    }
    row = episode_metrics(episode)
    assert row["steps"] == 3 and row["actions_executed"] == 2
    assert row["invalid_actions"] == 1 and math.isclose(row["invalid_action_rate"], 1 / 3)
    assert row["off_grid_clicks"] == 1 and row["ineffective_actions"] == 1
    assert row["latency_median"] == 2.0 and row["wall_seconds"] == 3.0
    assert row["prompt_tokens_total"] == 400 and row["prompt_tokens_per_step"] == 200
    assert row["progress"] == 0.4 and row["success"] is False

    final = {"status": "won", "safe_total": 10, "safe_revealed": 10}
    assert episode_metrics(episode, final)["progress"] == 1.0


def test_aggregate_pools_rates_over_steps() -> None:
    rows = [
        {"success": True, "progress": 1.0, "steps": 10, "invalid_actions": 1},
        {"success": False, "progress": 0.5, "steps": 30, "invalid_actions": 1},
    ]
    summary = aggregate(rows)
    assert summary["success_rate"] == 0.5
    assert summary["progress_mean"] == 0.75
    assert summary["invalid_action_rate"] == 2 / 40


@pytest.mark.parametrize("outcome", ["lost", "reset", "violation", "incorrect"])
def test_recovery_excludes_terminal_and_incorrect_moves(outcome):
    step = _step(True)
    if outcome == "lost":
        step["execution"]["details"]["board"]["status"] = "lost"
    elif outcome == "reset":
        step["execution"]["details"]["reset"] = True
    elif outcome == "violation":
        step["execution"]["evidence"] = {"task_violation": "Wrong target."}
    else:
        step["execution"]["evidence"] = {"logical_correct": False}
    assert recovery_counts([_step(False), step]) == (1, 0)


@pytest.mark.parametrize("kind", ["finish", "fail"])
def test_terminal_requests_are_not_executed_or_ineffective_actions(kind):
    step = _step(False)
    step["action"] = {"type": kind}
    row = episode_metrics({"steps": [step]})
    assert row["actions_executed"] == 0
    assert row["ineffective_actions"] == 0
    assert row["setbacks"] == 0


def test_infrastructure_errors_are_reported_outside_success_denominator():
    rows = [
        {"success": True, "termination_reason": "task_complete", "progress": 0.3},
        {"success": False, "termination_reason": "capture_error", "progress": 0.0},
    ]
    summary = aggregate(rows)
    assert summary["episodes"] == 2
    assert summary["scored_episodes"] == 1
    assert summary["infrastructure_failures"] == 1
    assert summary["success_rate"] == 1
    assert summary["progress_mean"] == 0.3
    assert summary["termination_reasons"]["capture_error"] == 1
