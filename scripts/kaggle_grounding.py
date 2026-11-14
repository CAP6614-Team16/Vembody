"""Run and summarize a small development grounding check without training the model."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vembody.evaluation.harness import ModelFactory


def summarize_grounding(output: Path, rows: list[dict]) -> dict:
    """Measure first-action grounding separately from episode success and response validity."""
    from vembody.sandbox import MinesweeperSandbox
    from vembody.tasks import load_task

    trials = []
    latencies = []
    for row in rows:
        task = load_task(row["task_path"])
        sandbox = MinesweeperSandbox(task, output / "unused")
        try:
            target = task.benchmark.predicate.target
            center = sandbox.screen.cell_center(target)
        finally:
            sandbox.shutdown()

        log = output / "logs" / task.id / f"trial_{row['trial']:03d}.jsonl"
        steps = [
            record["step"]
            for line in log.read_text(encoding="utf-8").splitlines()
            if (record := json.loads(line))["record_type"] == "step"
        ]
        latencies.extend(
            step["model_elapsed_seconds"] for step in steps if step.get("observation") is not None
        )
        first = steps[0] if steps else {}
        action = first.get("action") or {}
        execution = first.get("execution") or {}
        resolved = (execution.get("details") or {}).get("cell")
        valid = bool(action) and not any(
            first.get(key) for key in ("parse_error", "validation_error")
        )
        expected_kind = "right_click" if task.benchmark.predicate.type == "exact_flag" else "click"
        on_target = resolved == list(target)
        coordinate_error = None
        if action.get("type") in {"click", "right_click"} and valid:
            bounds = first["observation"]["capture_bounds"]
            pixel = (round(action["x"] * (bounds[2] - 1)), round(action["y"] * (bounds[3] - 1)))
            coordinate_error = math.dist(pixel, center)

        trials.append(
            {
                "task_id": task.id,
                "trial": row["trial"],
                "target_cell": list(target),
                "resolved_cell": resolved,
                "first_response_valid": valid,
                "first_action_on_target": on_target,
                "first_action_correct": valid and on_target and action.get("type") == expected_kind,
                "first_click_center_error_pixels": coordinate_error,
                "episode_success": row["success"],
                "infrastructure_failure": row["infrastructure_failure"],
                "steps": row["steps"],
            }
        )

    errors = [
        r["first_click_center_error_pixels"]
        for r in trials
        if r["first_click_center_error_pixels"] is not None
    ]
    count = len(trials)
    summary = {
        "purpose": "development_grounding_check_not_final_benchmark",
        "note": (
            "Fixed openings and greedy decoding; repeated trials check repeatability, "
            "not independent diversity."
        ),
        "episodes": count,
        "episode_success_rate": sum(r["episode_success"] for r in trials) / count
        if count
        else None,
        "first_response_valid_rate": sum(r["first_response_valid"] for r in trials) / count
        if count
        else None,
        "first_action_accuracy": sum(r["first_action_correct"] for r in trials) / count
        if count
        else None,
        "infrastructure_failures": sum(r["infrastructure_failure"] for r in trials),
        "model_calls": len(latencies),
        "mean_model_latency_seconds": statistics.mean(latencies) if latencies else None,
        "median_model_latency_seconds": statistics.median(latencies) if latencies else None,
        "mean_first_click_center_error_pixels": statistics.mean(errors) if errors else None,
        "coordinate_error_samples": len(errors),
        "trials": trials,
    }
    (output / "grounding-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def run_grounding(
    repo: Path,
    output: Path,
    trials: int = 3,
    model_factory: ModelFactory | None = None,
    adapter: str = "ui-mate",
) -> dict:
    """Run only the four existing T1 development instances, never solver-dependent tasks."""
    from vembody.evaluation.harness import ExperimentConfig, run_experiment

    if adapter not in {"ui-mate", "ui-mate-legacy"}:
        raise ValueError("Choose a UI-Mate adapter")

    config = ExperimentConfig(
        name="ui-mate-q6-grounding",
        model=adapter,
        task_paths=[
            str(repo / f"tasks/benchmark/development/t1-{number:02d}.yaml")
            for number in range(1, 5)
        ],
        trials=trials,
        timeout_seconds=300,
        model_config=str(repo / "configs/models/kaggle.yaml"),
        labels={"purpose": "development-grounding", "memory": "off", "quantization": "Q6_K"},
    )
    rows = run_experiment(
        config,
        output,
        model_factory=model_factory,
        progress=lambda row: print(
            f"{row['task_id']} trial {row['trial']}: success={row['success']}",
            flush=True,
        ),
    )
    summary = summarize_grounding(output, rows)
    print(json.dumps({k: v for k, v in summary.items() if k != "trials"}, indent=2), flush=True)
    return summary
