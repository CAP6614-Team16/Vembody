"""Evaluation harness: run tasks x trials under one configuration and record everything.

Output layout for ``run_experiment(..., out_dir="runs/<name>")``::

    runs/<name>/
        manifest.json          configuration, code version, environment, task hashes
        episodes.jsonl         one metrics row per episode (input to analysis)
        logs/<task>/trial_NNN.jsonl   full step log per episode
        screenshots/<run_id>/step_NNN.png
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import __version__
from ..logging.jsonl import JsonlLogger
from ..loop import AgentLoop
from ..models.base import VisionModel
from ..models.registry import build_model
from ..results import to_jsonable
from ..sandbox import MinesweeperSandbox
from ..tasks import TaskSpec, load_task
from ..validation import ActionPolicy
from .metrics import episode_metrics

ModelFactory = Callable[[TaskSpec, MinesweeperSandbox, int], VisionModel]


@dataclass
class ExperimentConfig:
    """Everything that defines one experimental condition."""

    name: str
    model: str
    task_paths: Sequence[str]
    trials: int = 1
    vary_seeds: bool = False
    base_seed: int = 0
    max_steps: int | None = None
    timeout_seconds: float | None = None
    labels: dict[str, str] = field(default_factory=dict)


def _git_info() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, check=True, timeout=10
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = git("status", "--porcelain")
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def episode_task(task: TaskSpec, config: ExperimentConfig, trial: int) -> TaskSpec:
    """Apply per-trial seed and limit overrides to a loaded task."""
    changes: dict[str, Any] = {}
    if config.max_steps is not None:
        changes["max_steps"] = config.max_steps
    if config.timeout_seconds is not None:
        changes["timeout_seconds"] = config.timeout_seconds
    if config.vary_seeds and task.minesweeper is not None:
        base = task.minesweeper.get("seed") or 0
        changes["minesweeper"] = {**task.minesweeper, "seed": base + trial}
    return replace(task, **changes) if changes else task


def run_experiment(
    config: ExperimentConfig,
    out_dir: str | Path,
    model_factory: ModelFactory | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run every task for ``config.trials`` trials and return the metrics rows."""
    out = Path(out_dir)
    if (out / "episodes.jsonl").exists():
        raise FileExistsError(f"{out} already holds results; choose a new experiment name")
    tasks = [(path, load_task(path)) for path in config.task_paths]
    for path, task in tasks:
        if task.environment != "minesweeper":
            raise ValueError(f"{path}: the harness currently supports minesweeper tasks only")
    out.mkdir(parents=True, exist_ok=True)

    def default_factory(task: TaskSpec, sandbox: MinesweeperSandbox, seed: int) -> VisionModel:
        return build_model(config.model, task, sandbox, seed)

    factory = model_factory or default_factory

    manifest = {
        "experiment": config.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": to_jsonable(config),
        "vembody_version": __version__,
        "git": _git_info(),
        "python": sys.version,
        "platform": platform.platform(),
        "argv": sys.argv,
        "tasks": [{"path": str(p), "id": t.id, "sha256": _file_sha256(p)} for p, t in tasks],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    rows = []
    for path, base_task in tasks:
        for trial in range(config.trials):
            task = episode_task(base_task, config, trial)
            sandbox = MinesweeperSandbox(task, out / "screenshots")
            model_seed = config.base_seed + trial
            loop = AgentLoop(
                model=factory(task, sandbox, model_seed),
                observations=sandbox,
                executor=sandbox,
                evaluator=sandbox,
                policy=ActionPolicy.from_task(task),
                logger=JsonlLogger(out / "logs" / task.id / f"trial_{trial:03d}.jsonl"),
            )
            episode = loop.run_episode(task)
            row = {
                "experiment": config.name,
                "model": config.model,
                **{f"label_{k}": v for k, v in config.labels.items()},
                "task_path": str(path),
                "trial": trial,
                "board_seed": (task.minesweeper or {}).get("seed"),
                "model_seed": model_seed,
                "max_steps": task.max_steps,
                **episode_metrics(to_jsonable(episode), sandbox.board_stats()),
            }
            with (out / "episodes.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            rows.append(row)
            if progress:
                progress(row)
    return rows
