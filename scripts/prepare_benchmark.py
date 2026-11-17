"""Generate benchmark candidates for the twelve templates.

Run from the repository root: python scripts/prepare_benchmark.py
T2/T3 starts are selected with solver certificates (vembody.benchmark.certify), so
each has a provably correct move under its template's rule. T1/T4 candidates remain
uncertified physical starting states.

Replace only the logical templates of one split, keeping its T1/T4 files:
python scripts/prepare_benchmark.py --templates T2-01 T2-02 T2-03 T2-04 T3-01 T3-02
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import yaml

from vembody.actions import ActionType
from vembody.benchmark.certify import LOGICAL_TEMPLATES, certify
from vembody.benchmark.environment import prepare, state_hash
from vembody.benchmark.labels import visible_hash
from vembody.benchmark.spec import TEMPLATES, BenchmarkSpec, Predicate
from vembody.errors import TaskLoadError
from vembody.minesweeper.deduce import solve_by_deduction, visible_board
from vembody.minesweeper.game import Minesweeper
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
    templates: list[str] | None = None,
) -> list[dict]:
    """Write candidates for ``templates`` (all twelve by default) into ``root``.

    With a template list, other templates' files in ``root`` and their manifest
    records are kept, and their seeds and visible starts are not reused.
    """
    root.mkdir(parents=True, exist_ok=True)
    selected = list(TEMPLATES) if templates is None else templates
    records = []
    seed = seed_start
    seen_visible = set()
    used_seeds = set()
    trained = set()  # boards in the fine-tuning data
    kept = []
    if templates is not None:
        manifest = root / "manifest.json"
        old = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
        kept = [r for r in old.get("instances", []) if r["template_id"] not in selected]
        exclude = [*exclude, root]
    for folder in exclude:
        if (folder / "train.jsonl").exists():
            for part in ("train", "val"):
                for line in (folder / f"{part}.jsonl").open(encoding="utf-8"):
                    trained.add(tuple(json.loads(json.loads(line)["response"])["board"]))
            continue
        for t in load_task_set(list(folder.glob("*.yaml"))):
            if folder == root and t.benchmark.template_id in selected:
                continue  # being replaced
            seen_visible.add(visible_hash(prepare(t)))
            used_seeds.add(t.minesweeper["seed"])
    for i, (template, kind) in enumerate(TEMPLATES.items()):
        if template not in selected:
            continue
        for k in range(1, instances + 1):
            size, mines = (5, 3) if template == "T4-01" else (9, 10)
            target = {"T1-01": (2, 4), "T1-02": (6, 1), "T1-04": (0, 8)}.get(template)
            steps, seconds = LIMITS.get(template, (3, 120))
            certificate = None
            for _ in range(20000):
                board = {"rows": size, "cols": size, "mines": mines, "seed": seed}
                seed += 1
                if board["seed"] in used_seeds:
                    continue
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
                    # Outline the first clue whose touching mines are all provable
                    game = Minesweeper(size, size, mines, board["seed"])
                    game.reveal(opening)
                    region = next(
                        (
                            c
                            for c in sorted(game.revealed)
                            if game.adjacent_mines(c) and certify(template, game, (c,))
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
                # T4 boards must be winnable without guessing
                if template in ("T4-01", "T4-02"):
                    solved, _ = solve_by_deduction(screen.game)
                    if solved.status != "won":
                        continue
                # T2/T3 starts must meet their template's solver certificate
                if template in LOGICAL_TEMPLATES:
                    region = task.benchmark.predicate.region
                    certificate = certify(template, screen.game, region)
                    if certificate is None:
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
                used_seeds.add(board["seed"])
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
            header = (
                "# Solver-certified start: a provably correct move exists (see manifest).\n"
                if certificate
                else "# Candidate only: solver certification is pending.\n"
            )
            path.write_text(header + yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
            record = {
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
            if certificate:
                record["certification_status"] = "solver_certified"
                record["pending_checks"] = ["certificate review"]
                record["certificate"] = certificate
            records.append(record)
    order = list(TEMPLATES)
    records = sorted(kept + records, key=lambda r: (order.index(r["template_id"]), r["path"]))
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
    parser.add_argument(
        "--templates",
        nargs="+",
        choices=tuple(TEMPLATES),
        help="regenerate only these templates and keep the other files in --output",
    )
    args = parser.parse_args()
    records = generate(
        args.output, args.split, args.seed_start, args.exclude, args.instances, args.templates
    )
    certified = sum(r["certification_status"] == "solver_certified" for r in records)
    print(f"Prepared {len(records)} candidates ({certified} solver-certified).")
