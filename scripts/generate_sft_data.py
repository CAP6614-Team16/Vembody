"""Generate supervised fine-tuning examples from solver play on fresh boards.

Each example is a screenshot with the same prompt used at inference, and a target
reply with the board copy, the clue that proves the move, and the action. Only
provable moves are used as targets. Benchmark boards are skipped.

    python scripts/generate_sft_data.py --episodes 800 --out data/sft
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path

from vembody.actions import action_from_dict
from vembody.benchmark.environment import prepare
from vembody.benchmark.labels import visible_board
from vembody.minesweeper.deduce import cell_name, deductions
from vembody.minesweeper.screen import MinesweeperScreen
from vembody.models.prompts import PROMPT_VERSION, SYSTEM_PROMPT, task_prompt
from vembody.results import StepResult
from vembody.tasks import load_task

SIZES = [(5, 5, 3), (6, 6, 5), (7, 7, 7), (8, 8, 9), (9, 9, 10), (9, 9, 12)]
# small boards give fewer examples per game
SIZE_WEIGHTS = [4, 3, 2, 1.5, 1, 1]
PLAY = "Reveal every safe cell without revealing a mine."
GROUNDING = 0.25  # share of examples that ask for one named cell instead of play
RECOVERY = 0.2  # share whose history ends with a repeated action that changed nothing
SEED_OFFSET = 100_000  # keeps seeds away from the benchmark generator's
BENCHMARK = Path("tasks/benchmark")
TEMPLATE = BENCHMARK / "development/t4-01.yaml"


def benchmark_boards() -> set:
    """Visible starting boards of every benchmark task, to keep them out of training."""
    return {
        visible_board(prepare(load_task(path))).cells for path in sorted(BENCHMARK.glob("*/*.yaml"))
    }


def click(screen: MinesweeperScreen, cell, kind: str) -> dict:
    px, py = screen.cell_center(cell)
    return {
        "type": kind,
        "x": round(px / (screen.width - 1), 3),
        "y": round(py / (screen.height - 1), 3),
    }


def wasted_action(rng, screen: MinesweeperScreen, board) -> dict:
    """A click that changes nothing: on an opened cell or off the grid."""
    opened = [
        (r, c) for r, row in enumerate(board) for c, cell in enumerate(row) if cell not in "#F"
    ]
    if opened and rng.random() < 0.6:
        return click(screen, rng.choice(opened), "click")
    return {"type": rng.choice(["click", "right_click"]), "x": round(rng.random(), 3), "y": 0.01}


def example(rng, screen, board, cell, kind, reason, history, template):
    """Build one (instruction, history, target) triple for a provable move."""
    name = cell_name(cell)
    if rng.random() < GROUNDING:
        verb = "Reveal" if kind == "click" else "Flag"
        instruction = f"{verb} the cell in row {name[0]}, column {name[1:]}."
        summary = f"The instruction names {name}: row {name[0]}, column {name[1:]}."
    else:
        instruction = PLAY
        verb = "reveal" if kind == "click" else "flag"
        summary = f"{reason}; {verb} {name}."
    history = list(history)
    if rng.random() < RECOVERY:
        miss = wasted_action(rng, screen, board)
        repeats = rng.choice([1, 2, 3, 5, 10, 20, 40])  # shown as one entry with "times"
        history += [StepResult(observation=None, action=action_from_dict(miss))] * repeats
    target = {
        "board": [" ".join(row) for row in board],
        "reasoning_summary": summary,
        "action": click(screen, cell, kind),
    }
    task = replace(template, instruction=instruction)
    return {
        "system": SYSTEM_PROMPT,
        "prompt": task_prompt(task, history),
        "response": json.dumps(target),
    }


def episode(seed: int, template, excluded: set, images: Path) -> list[dict]:
    rng = random.Random(seed)
    rows, cols, mines = rng.choices(SIZES, weights=SIZE_WEIGHTS)[0]
    screen = MinesweeperScreen(rows, cols, mines, seed=seed, benchmark=True, grid_labels=True)
    game = screen.game
    game.reveal((rng.randrange(rows), rng.randrange(cols)))
    history, examples = [], []
    while game.status == "playing":
        board = visible_board(screen).cells
        moves = deductions(board)
        if moves:
            cell = rng.choice(sorted(moves))
            kind, reason = moves[cell]
            if board not in excluded:
                record = example(rng, screen, board, cell, kind, reason, history, template)
                path = images / f"{seed}-{len(history):03d}.png"
                screen.screenshot().save(path)
                examples.append({"id": path.stem, "seed": seed, "image": path.name, **record})
        else:
            safe = [
                (r, c)
                for r in range(rows)
                for c in range(cols)
                if board[r][c] == "#" and (r, c) not in game.mines
            ]
            cell, kind = rng.choice(safe), "click"
        action = click(screen, cell, kind)
        history.append(StepResult(observation=None, action=action_from_dict(action)))
        game.reveal(cell) if kind == "click" else game.toggle_flag(cell)
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--val-share", type=float, default=0.1)
    parser.add_argument("--out", type=Path, default=Path("data/sft"))
    args = parser.parse_args()

    template = load_task(TEMPLATE)
    excluded = benchmark_boards()
    images = args.out / "images"
    images.mkdir(parents=True, exist_ok=True)
    val_episodes = max(1, round(args.episodes * args.val_share))
    counts = {}
    for split, seeds in (
        ("val", range(SEED_OFFSET, SEED_OFFSET + val_episodes)),
        ("train", range(SEED_OFFSET + val_episodes, SEED_OFFSET + args.episodes)),
    ):
        with (args.out / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            counts[split] = 0
            for seed in seeds:
                for record in episode(seed, template, excluded, images):
                    handle.write(json.dumps(record) + "\n")
                    counts[split] += 1
    meta = {"prompt_version": PROMPT_VERSION, "episodes": args.episodes, "examples": counts}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {counts['train']} train and {counts['val']} val examples to {args.out}")


if __name__ == "__main__":
    main()
