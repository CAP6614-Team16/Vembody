"""Command-line entry point for the Milestone 1 runtime."""

from __future__ import annotations

import argparse
import json

from .evaluation.base import NullEvaluator
from .execution.desktop import DesktopExecutor
from .execution.dry_run import DryRunExecutor
from .logging.jsonl import JsonlLogger
from .loop import AgentLoop
from .models.mock import MockModel
from .perception.file import FileObservationProvider
from .tasks import TaskSpec, load_task
from .validation import ActionPolicy


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(prog="vembody")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run one bounded episode")
    run_parser.add_argument("--task", required=True, help="path to a YAML or JSON task")
    run_parser.add_argument("--model", choices=("mock",), default="mock")
    run_parser.add_argument("--executor", choices=("dry-run", "desktop"), default="dry-run")
    run_parser.add_argument("--log", default="artifacts/episode.jsonl")
    run_parser.add_argument("--allow-live-actions", action="store_true")
    return parser


def build_loop(args: argparse.Namespace) -> tuple[AgentLoop, TaskSpec]:
    """Construct the concrete Milestone 1 runtime."""
    task = load_task(args.task)
    if args.model != "mock":
        raise ValueError("Only the mock model is implemented in Milestone 1")
    model = MockModel.from_task(task)
    observations = FileObservationProvider(task.screenshot_path)
    if args.executor == "dry-run":
        executor = DryRunExecutor()
    else:
        executor = DesktopExecutor(allow_live_actions=args.allow_live_actions)
    loop = AgentLoop(
        model=model,
        observations=observations,
        executor=executor,
        evaluator=NullEvaluator(),
        policy=ActionPolicy.from_task(task),
        logger=JsonlLogger(args.log),
    )
    return loop, task


def run_command(args: argparse.Namespace) -> int:
    """Run one configured episode and print its summary."""
    loop, task = build_loop(args)
    result = loop.run_episode(task)
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "task_id": result.task_id,
                "termination_reason": result.termination_reason.value,
                "evaluation_status": result.evaluation.status.value,
                "steps": len(result.steps),
            },
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch the selected command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return run_command(args)
    parser.error("Unknown command")
    return 2
