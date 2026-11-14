"""Bounded autonomous Minesweeper play from one high-level goal and screenshots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vembody.evaluation.harness import ModelFactory

AUTONOMOUS_PROFILES = ("original", "budget-only", "brief")


def response_diagnostics(output: Path) -> dict[str, int]:
    """Count truncations and parse failures once per model step, not duplicated episode records."""
    counts = {"truncated_responses": 0, "parse_errors": 0}
    for path in (output / "logs").rglob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("record_type") != "step":
                continue

            step = record["step"]
            counts["truncated_responses"] += (
                step.get("model_metadata", {}).get("finish_reason") == "length"
            )
            counts["parse_errors"] += bool(step.get("parse_error"))

    return counts


def run_autonomous(
    repo: Path,
    output: Path,
    trials: int = 1,
    adapter: str = "ui-mate",
    model_factory: ModelFactory | None = None,
    *,
    profile: str = "brief",
) -> dict:
    """Evaluate the existing fixed 5x5 development game with its 40-action/900-second limits."""
    from vembody.evaluation.harness import ExperimentConfig, run_experiment

    if adapter not in {"ui-mate", "ui-mate-legacy"}:
        raise ValueError("Choose a UI-Mate adapter")

    if profile not in AUTONOMOUS_PROFILES:
        raise ValueError("Choose an autonomous response profile")

    if profile == "brief" and adapter != "ui-mate":
        raise ValueError("The brief profile requires the v2 ui-mate adapter")

    model = "ui-mate-brief" if profile == "brief" else adapter
    endpoint = "kaggle.yaml" if profile == "original" else "kaggle-autonomous.yaml"
    config = ExperimentConfig(
        name="ui-mate-autonomous-development",
        model=model,
        task_paths=[str(repo / "tasks/benchmark/development/t4-01.yaml")],
        trials=trials,
        model_config=str(repo / "configs/models" / endpoint),
        labels={
            "purpose": "autonomous-development",
            "history": "off",
            "training": "none",
            "response_profile": profile,
        },
    )
    rows = run_experiment(config, output, model_factory=model_factory)
    count = len(rows)
    calls = sum(row["steps"] for row in rows)
    summary = {
        "purpose": "autonomous_play_development_not_final_benchmark",
        "task": "Reveal every safe cell without revealing a mine.",
        "episodes": count,
        "response_profile": profile,
        "actions_executed": sum(row["actions_executed"] for row in rows),
        **response_diagnostics(output),
        "success_rate": sum(row["success"] for row in rows) / count,
        "mine_losses": sum(row["mine_hit"] for row in rows),
        "off_grid_clicks": sum(row["off_grid_clicks"] for row in rows),
        "ineffective_actions": sum(row["ineffective_actions"] for row in rows),
        "invalid_actions": sum(row["invalid_actions"] for row in rows),
        "infrastructure_failures": sum(row["infrastructure_failure"] for row in rows),
        "false_completion_claims": sum(
            row["termination_reason"] == "finish_requested" and not row["success"] for row in rows
        ),
        "model_calls": calls,
        "mean_model_latency_seconds": sum(row["model_seconds_total"] for row in rows) / calls
        if calls
        else None,
        "note": (
            "Only screenshots and a high-level goal reach the agent. Fixed development opening, "
            "no history, no training, no independently certified strategy labels; a mine hit "
            "does not by itself distinguish a reasoning error from an unavoidable guess."
        ),
    }
    (output / "autonomous-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary
