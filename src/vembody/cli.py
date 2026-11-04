"""Command-line entry point: run episodes, run and analyze experiments, play minesweeper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluation.analysis import analyze, summary_markdown
from .evaluation.base import NullEvaluator
from .evaluation.harness import ExperimentConfig, run_experiment
from .evaluation.metrics import aggregate
from .execution.desktop import DesktopExecutor
from .execution.dry_run import DryRunExecutor
from .logging.jsonl import JsonlLogger
from .loop import AgentLoop
from .minesweeper.screen import MinesweeperScreen
from .minesweeper.webui import make_server
from .models.registry import MODEL_NAMES, build_model
from .perception.file import FileObservationProvider
from .sandbox import MinesweeperSandbox
from .tasks import TaskSpec, load_task
from .validation import ActionPolicy


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(prog="vembody")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run one bounded episode")
    run_parser.add_argument("--task", required=True, help="path to a YAML or JSON task")
    run_parser.add_argument("--model", choices=MODEL_NAMES, default="mock")
    run_parser.add_argument("--executor", choices=("dry-run", "desktop"), default="dry-run")
    run_parser.add_argument("--log", default="artifacts/episode.jsonl")
    run_parser.add_argument("--allow-live-actions", action="store_true")

    eval_parser = subparsers.add_parser("eval", help="run tasks x trials as one experiment")
    eval_parser.add_argument("--tasks", nargs="+", required=True, help="task files or folders")
    eval_parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    eval_parser.add_argument("--trials", type=int, default=5)
    eval_parser.add_argument("--name", required=True, help="experiment name (output folder)")
    eval_parser.add_argument("--out", default="runs", help="parent folder for experiments")
    eval_parser.add_argument(
        "--vary-seeds", action="store_true", help="trial k plays board seed task_seed + k"
    )
    eval_parser.add_argument("--base-seed", type=int, default=0, help="first model seed")
    eval_parser.add_argument("--max-steps", type=int, help="override every task's max_steps")
    eval_parser.add_argument("--timeout", type=float, help="override every task's timeout")
    eval_parser.add_argument(
        "--label",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="tag the condition, e.g. --label memory=off (repeatable)",
    )

    analyze_parser = subparsers.add_parser("analyze", help="summarize experiment folders")
    analyze_parser.add_argument("experiments", nargs="+", help="experiment folders under runs/")
    analyze_parser.add_argument("--out", default="results", help="folder for tables and plots")

    play_parser = subparsers.add_parser("play", help="play minesweeper in the browser")
    play_parser.add_argument("--seed", type=int, default=None)
    play_parser.add_argument("--port", type=int, default=8000)
    return parser


def build_loop(args: argparse.Namespace) -> tuple[AgentLoop, TaskSpec]:
    """Construct the runtime for one task."""
    task = load_task(args.task)
    sandbox = None

    # Minesweeper tasks run in the sandbox, which captures, executes, and grades
    if task.environment == "minesweeper":
        sandbox = MinesweeperSandbox(task, Path(args.log).parent / "screenshots")
        observations = executor = evaluator = sandbox
    else:
        observations = FileObservationProvider(task.screenshot_path)
        evaluator = NullEvaluator()
        if args.executor == "dry-run":
            executor = DryRunExecutor()
        else:
            executor = DesktopExecutor(allow_live_actions=args.allow_live_actions)

    loop = AgentLoop(
        model=build_model(args.model, task, sandbox),
        observations=observations,
        executor=executor,
        evaluator=evaluator,
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


def play_command(args: argparse.Namespace) -> int:
    """Serve a playable minesweeper game until interrupted."""
    server = make_server(MinesweeperScreen(seed=args.seed), args.port)
    print(f"Play at http://127.0.0.1:{args.port} (Ctrl+C to stop)")
    server.serve_forever()
    return 0


def _task_paths(entries: list[str]) -> list[str]:
    """Expand folders into their task files, sorted for a stable order."""
    paths: list[str] = []
    for entry in entries:
        path = Path(entry)
        if path.is_dir():
            paths.extend(
                str(p) for p in sorted(path.iterdir()) if p.suffix in {".yaml", ".yml", ".json"}
            )
        else:
            paths.append(entry)
    return paths


def eval_command(args: argparse.Namespace) -> int:
    """Run one experiment and print a short summary."""
    labels = {}
    for item in args.label:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"--label must look like KEY=VALUE, got {item!r}")
        labels[key] = value
    config = ExperimentConfig(
        name=args.name,
        model=args.model,
        task_paths=_task_paths(args.tasks),
        trials=args.trials,
        vary_seeds=args.vary_seeds,
        base_seed=args.base_seed,
        max_steps=args.max_steps,
        timeout_seconds=args.timeout,
        labels=labels,
    )
    out = Path(args.out) / args.name

    def report(row: dict) -> None:
        progress = row.get("progress")
        shown = f"{100 * progress:5.1f}%" if progress is not None else "  n/a"
        print(
            f"{row['task_id']} trial {row['trial']}: {row['evaluation_status']:<13} "
            f"progress {shown}  steps {row['steps']:>3}  ({row['termination_reason']})"
        )

    rows = run_experiment(config, out, progress=report)
    print(summary_markdown([{"experiment": args.name, "model": args.model, **aggregate(rows)}]))
    print(f"Results written to {out}")
    return 0


def analyze_command(args: argparse.Namespace) -> int:
    """Write tables and plots comparing experiments."""
    result = analyze(args.experiments, args.out)
    print(summary_markdown(result["summary"]))
    for plot in result["plots"]:
        print(f"Wrote {plot}")
    print(f"Tables written to {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch the selected command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return run_command(args)
    if args.command == "play":
        return play_command(args)
    if args.command == "eval":
        return eval_command(args)
    if args.command == "analyze":
        return analyze_command(args)
    parser.error("Unknown command")
    return 2
