"""Generate an explicitly uncertified development pilot for all twelve templates.

Run from the repository root: python scripts/prepare_benchmark.py
This selects physical starting states, NOT solver-certified difficulty/solutions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import yaml

from vembody.actions import ActionType
from vembody.benchmark.environment import prepare, state_hash
from vembody.benchmark.labels import visible_hash
from vembody.benchmark.spec import TEMPLATES, BenchmarkSpec, Predicate
from vembody.errors import TaskLoadError
from vembody.minesweeper.deduce import solve_by_deduction, visible_board
from vembody.tasks import TaskSpec, load_task_set

INSTRUCTIONS = [
    "Reveal the cell in row C, column 5.",
    "Flag the cell in row G, column 2.",
    "Reveal the hidden cell directly to the right of the only revealed 3.",
    "Reveal the top-right corner cell.",
    "Reveal one hidden cell that is guaranteed safe.",
    "Flag one hidden cell that is guaranteed to contain a mine.",
    "Reveal one hidden cell that is guaranteed safe.",
    "Reveal one hidden cell that is guaranteed safe.",
    "Flag every mine among the hidden cells touching the yellow outlined region.",
    "Make 10 logically safe reveal moves without losing. A flood fill counts as one move.",
    "Reveal every safe cell without revealing a mine.",
    "Reveal every safe cell without revealing a mine.",
]


LIMITS = {  # (max steps, timeout seconds) for the multi-move templates
    "T3-01": (20, 600),
    "T3-02": (25, 600),
    "T4-01": (40, 900),
    "T4-02": (120, 1800),
}


def generate(
    root: Path,
    split: str = "development",
    seed_start: int = 0,
    exclude: list[Path] = (),
    instances: int = 1,
) -> list[dict]:
    root.mkdir(parents=True, exist_ok=True)
    records = []
    seed = seed_start
    seen_visible = set()
    trained = set()  # boards in the fine-tuning data
    for folder in exclude:
        if (folder / "train.jsonl").exists():
            for part in ("train", "val"):
                for line in (folder / f"{part}.jsonl").open(encoding="utf-8"):
                    trained.add(tuple(json.loads(json.loads(line)["response"])["board"]))
            continue
        seen_visible.update(
            visible_hash(prepare(t)) for t in load_task_set(list(folder.glob("*.yaml")))
        )
    for i, (template, kind) in enumerate(TEMPLATES.items()):
        for k in range(1, instances + 1):
            size, mines = (5, 3) if template == "T4-01" else (9, 10)
            target = {"T1-01": (2, 4), "T1-02": (6, 1), "T1-04": (0, 8)}.get(template)
            steps, seconds = LIMITS.get(template, (3, 120))
            for _ in range(10000):
                board = {"rows": size, "cols": size, "mines": mines, "seed": seed}
                seed += 1
                opening = (board["seed"] // size % size, board["seed"] % size)
                if template == "T1-03":
                    target = (opening[0], opening[1] + 1)
                predicate = Predicate(kind, target, count=10 if template == "T3-02" else 0)
                spec = BenchmarkSpec(1, template, split, opening, (), predicate)
                task = TaskSpec(
                    f"{split}-{template}-{k:03d}",
                    f"{template} {split} candidate",
                    INSTRUCTIONS[i],
                    "minesweeper",
                    "Reconstruct fixed opening.",
                    "See versioned benchmark predicate.",
                    steps,
                    seconds,
                    (ActionType.CLICK, ActionType.RIGHT_CLICK, ActionType.FINISH),
                    minesweeper=board,
                    benchmark=spec,
                )
                if template == "T3-01":
                    from vembody.minesweeper.game import Minesweeper

                    game = Minesweeper(size, size, mines, board["seed"])
                    game.reveal(opening)
                    region = next(
                        (
                            c
                            for c in sorted(game.revealed)
                            if game.adjacent_mines(c)
                            and any(n in game.mines for n in game.neighbors(c))
                        ),
                        None,
                    )
                    if region is None:
                        continue
                    task = replace(
                        task, benchmark=replace(spec, predicate=Predicate(kind, region=(region,)))
                    )
                try:
                    screen = prepare(task)
                except TaskLoadError:
                    continue
                if visible_hash(screen) in seen_visible:
                    continue
                if tuple(" ".join(row) for row in visible_board(screen.game)) in trained:
                    continue
                # T4 boards must be winnable without guessing; T3-02 needs 10 provable reveals
                if template in ("T3-02", "T4-01", "T4-02"):
                    solved, reveals = solve_by_deduction(screen.game)
                    if template == "T3-02" and reveals < 10:
                        continue
                    if template != "T3-02" and solved.status != "won":
                        continue
                if (
                    template == "T3-02"
                    and sum(
                        (r, c) not in screen.game.revealed
                        and (r, c) not in screen.game.mines
                        and bool(screen.game.adjacent_mines((r, c)))
                        for r in range(size)
                        for c in range(size)
                    )
                    < 10
                ):
                    continue
                seen_visible.add(visible_hash(screen))
                break
            else:
                raise ValueError(f"Could not prepare candidate for {template}")
            p = task.benchmark.predicate
            data = {
                "id": task.id,
                "name": task.name,
                "instruction": task.instruction,
                "environment": task.environment,
                "reset_instructions": task.reset_instructions,
                "success_description": task.success_description,
                "max_steps": steps,
                "timeout_seconds": seconds,
                "allowed_actions": [a.value for a in task.allowed_actions],
                "minesweeper": board,
                "benchmark": {
                    "version": 1,
                    "template_id": template,
                    "split": split,
                    "opening": list(opening),
                    "setup": [],
                    "predicate": {"type": kind},
                    "expected_state_hash": state_hash(screen),
                },
            }
            if p.target is not None:
                data["benchmark"]["predicate"]["target"] = list(p.target)
                px, py = screen.cell_center(p.target)
                data["mock_actions"] = [
                    {
                        "type": "right_click" if kind == "exact_flag" else "click",
                        "x": px / (screen.width - 1),
                        "y": py / (screen.height - 1),
                    }
                ]
            if p.region:
                data["benchmark"]["predicate"]["region"] = [list(c) for c in p.region]
            if p.count:
                data["benchmark"]["predicate"]["count"] = p.count
            name = template.lower() if instances == 1 else f"{template.lower()}-{k:03d}"
            path = root / f"{name}.yaml"
            path.write_text(
                "# Candidate only: solver certification is pending.\n"
                + yaml.safe_dump(data, sort_keys=False),
                encoding="utf-8",
            )
            records.append(
                {
                    "instance_id": task.id,
                    "template_id": template,
                    "path": path.name,
                    "state_hash": state_hash(screen),
                    "certification_status": "pending",
                    "pending_checks": [
                        "whole-board guess-free proof",
                        "template difficulty",
                        "solver reference trace",
                    ],
                }
            )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "split": split,
                "release_status": "uncertified_candidates",
                "instances": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("tasks/benchmark/development"))
    parser.add_argument(
        "--split", choices=("development", "demonstration", "evaluation"), default="development"
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--exclude",
        type=Path,
        nargs="*",
        default=[],
        help="Skip boards in these task or SFT data dirs",
    )
    parser.add_argument("--instances", type=int, default=1, help="boards per template")
    args = parser.parse_args()
    count = len(generate(args.output, args.split, args.seed_start, args.exclude, args.instances))
    print(f"Prepared {count} uncertified candidates.")
