"""Export reproducible screenshots and evaluator-only visible transcriptions.

No logical example is invented without a reviewed proof. Only the four grounding
demonstrations have response examples. Run this after prepare_benchmark.py.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import replace
from pathlib import Path

import PIL
from PIL import Image, ImageDraw

from vembody.actions import action_from_dict
from vembody.benchmark.environment import prepare, state_hash
from vembody.benchmark.labels import visible_board, visible_hash
from vembody.sandbox import MinesweeperSandbox
from vembody.tasks import load_task_set


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def export(sources: list[Path], output: Path) -> dict:
    tasks = load_task_set([p for source in sources for p in sorted(source.glob("*.yaml"))])
    seeds, layouts, starts, visible_starts = set(), set(), set(), set()
    records, examples = [], []
    contacts = {}
    for task in tasks:
        spec = task.benchmark
        if spec is None:
            raise ValueError("Only benchmark instances may be exported")
        screen = prepare(task)
        seed = task.minesweeper["seed"]
        layout = (screen.rows, screen.cols, tuple(sorted(screen.game.mines)))
        start = state_hash(screen)
        visible_start = visible_hash(screen)
        if seed in seeds or layout in layouts or start in starts or visible_start in visible_starts:
            raise ValueError(f"Duplicate seed, layout, or starting state: {task.id}")
        seeds.add(seed)
        layouts.add(layout)
        starts.add(start)
        visible_starts.add(visible_start)
        raw = screen.screenshot()
        grid = prepare(replace(task, benchmark=replace(spec, grid_labels=True))).screenshot()
        if raw.size != grid.size:
            raise ValueError("Raw/grid screenshot geometry mismatch")
        subdir = output / spec.split
        subdir.mkdir(parents=True, exist_ok=True)
        images = {}
        for suffix, picture in (("raw", raw), ("grid", grid)):
            path = subdir / f"{task.id.lower()}-{suffix}.png"
            picture.save(path)
            images[suffix] = {
                "path": path.relative_to(output).as_posix(),
                "file_sha256": digest(path.read_bytes()),
                "pixel_sha256": digest(picture.tobytes()),
            }
        record = {
            "instance_id": task.id,
            "template_id": spec.template_id,
            "split": spec.split,
            "seed": seed,
            "state_hash": start,
            "visible_hash": visible_start,
            "layout_hash": digest(json.dumps(layout).encode()),
            "width": raw.width,
            "height": raw.height,
            "grid_left": screen.grid_left,
            "grid_top": screen.grid_top,
            "cell_size": 32,
            "images": images,
            "visible_board": visible_board(screen).cells,
            "certification_status": "pending",
        }
        records.append(record)
        contact = contacts.setdefault(spec.split, Image.new("RGB", (3 * 688, 4 * 410), "white"))
        index = len([r for r in records if r["split"] == spec.split]) - 1
        if index < 12:
            left, top = (index % 3) * 688, (index // 3) * 410
            ImageDraw.Draw(contact).text(
                (left + 8, top + 2), f"{spec.template_id} raw / grid - UNCERTIFIED", fill="black"
            )
            contact.paste(raw, (left + 4, top + 24))
            contact.paste(grid, (left + 348, top + 24))
        if spec.split == "demonstration":
            example = {
                "instance_id": task.id,
                "template_id": spec.template_id,
                "images": images,
                "instruction": task.instruction,
                "status": "pending_solver_example",
                "response": None,
            }
            if spec.template_id.startswith("T1"):
                check = MinesweeperSandbox(task, output / "unused")
                check.execute(action_from_dict(task.mock_actions[0]))
                if not check.scorer.success:
                    raise ValueError("Grounding example failed its predicate")
                example.update(
                    status="verified_grounding_only",
                    response={
                        "reasoning_summary": "Select the cell requested by the instruction.",
                        "action": task.mock_actions[0],
                        "expected_result": "The requested cell changes.",
                    },
                )
            examples.append(example)
    manifest = {
        "version": 1,
        "release_status": "uncertified_candidates",
        "runtime": {
            "python": platform.python_version(),
            "pillow": PIL.__version__,
            "renderer": "pillow-benchmark-v1",
            "font": "Pillow default size20",
        },
        "instances": records,
        "examples": examples,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for split, contact in contacts.items():
        contact.save(output / f"{split}-contact-sheet.png")
    return manifest


if __name__ == "__main__":
    root = Path("tasks/benchmark")
    manifest = export([root / "development", root / "demonstration"], Path("data/benchmark/pilot"))
    print(f"Exported {len(manifest['instances'])} candidate image pairs; certification pending.")
