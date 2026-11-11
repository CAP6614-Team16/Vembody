"""Validate candidate coverage and assets, with an explicit certification gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

from PIL import Image

from vembody.benchmark.environment import prepare, state_hash
from vembody.benchmark.labels import CertificationRequired, visible_board, visible_hash
from vembody.benchmark.replay import replay_grounding
from vembody.benchmark.spec import TEMPLATES
from vembody.cli import build_loop, build_parser
from vembody.tasks import load_task_set


def check(root: Path, assets: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    paths = [
        p
        for split in ("development", "demonstration")
        for p in sorted((root / split).glob("*.yaml"))
    ]
    tasks = load_task_set(paths)
    manifest = json.loads((assets / "manifest.json").read_text(encoding="utf-8"))
    images = {r["instance_id"]: r for r in manifest["instances"]}
    if len(images) != len(manifest["instances"]) or set(images) != {t.id for t in tasks}:
        raise ValueError("Asset/task instance coverage mismatch")
    for split in ("development", "demonstration"):
        group = [t for t in tasks if t.benchmark.split == split]
        if len(group) != 12 or {t.benchmark.template_id for t in group} != set(TEMPLATES):
            raise ValueError(f"Incomplete template coverage: {split}")
        source_manifest = json.loads((root / split / "manifest.json").read_text(encoding="utf-8"))
        if {r["instance_id"] for r in source_manifest["instances"]} != {t.id for t in group}:
            raise ValueError("Source manifest disagrees with task files")
    seen_seeds, seen_layouts, seen_visible = set(), set(), set()
    rows = []
    for path, task in zip(paths, tasks, strict=True):
        screen = prepare(task)
        repeated = prepare(task)
        if state_hash(screen) != state_hash(repeated) or (
            screen.screenshot().tobytes() != repeated.screenshot().tobytes()
        ):
            raise ValueError("Non-repeatable reset")
        image_record = images[task.id]
        if set(image_record["images"]) != {"raw", "grid"}:
            raise ValueError("Each instance requires exactly raw and grid images")
        expected_geometry = {
            "width": screen.width,
            "height": screen.height,
            "grid_left": screen.grid_left,
            "grid_top": screen.grid_top,
            "cell_size": 32,
        }
        if any(image_record[key] != value for key, value in expected_geometry.items()):
            raise ValueError("Manifest geometry mismatch")
        if image_record["visible_board"] != json.loads(json.dumps(visible_board(screen).cells)):
            raise ValueError("Manifest visible transcription mismatch")
        if state_hash(screen) != image_record["state_hash"] or (
            visible_hash(screen) != image_record["visible_hash"]
        ):
            raise ValueError("Manifest state identity mismatch")
        layout = (screen.rows, screen.cols, tuple(sorted(screen.game.mines)))
        seed, visible = task.minesweeper["seed"], visible_hash(screen)
        if seed in seen_seeds or layout in seen_layouts or visible in seen_visible:
            raise ValueError("Duplicate seed/layout/visible state")
        seen_seeds.add(seed)
        seen_layouts.add(layout)
        seen_visible.add(visible)
        for variant, record in image_record["images"].items():
            picture = prepare(
                replace(task, benchmark=replace(task.benchmark, grid_labels=variant == "grid"))
            )
            image_path = assets / record["path"]
            if hashlib.sha256(image_path.read_bytes()).hexdigest() != record["file_sha256"]:
                raise ValueError("Image file checksum mismatch")
            with Image.open(image_path) as saved:
                if (
                    saved.mode != "RGB"
                    or saved.size != picture.screenshot().size
                    or (saved.tobytes() != picture.screenshot().tobytes())
                ):
                    raise ValueError("Reference pixels do not match this renderer")
                if hashlib.sha256(saved.tobytes()).hexdigest() != record["pixel_sha256"]:
                    raise ValueError("Image pixel checksum mismatch")
        row = {
            "instance_id": task.id,
            "template_id": task.benchmark.template_id,
            "split": task.benchmark.split,
            "reset_and_images": "passed",
            "certification": "pending",
            "integration": "not_run",
        }
        log_path = output / f"{task.id}.jsonl"
        if log_path.exists():
            # JsonlLogger appends, so keep reruns in separate directories.
            raise ValueError(f"Use a fresh output directory; log already exists: {log_path}")
        args = build_parser().parse_args(["run", "--task", str(path), "--log", str(log_path)])
        try:
            loop, loaded = build_loop(args)
        except CertificationRequired:
            row["integration"] = "blocked_missing_logical_provider"
        else:
            episode = loop.run_episode(loaded)
            if task.benchmark.template_id.startswith("T1"):
                if episode.evaluation.status.value != "success" or len(episode.steps) != 1:
                    raise ValueError("Grounding reference trace failed")
                row["replay"] = replay_grounding(task, log_path)
                row["integration"] = "grounding_reference_passed"
                post = Path(episode.steps[-1].execution.evidence["post_screenshot_path"])
                if not post.exists():
                    raise ValueError("Missing terminal evidence")
            else:
                if episode.evaluation.status.value != "failure":
                    raise ValueError("Premature-finish negative control must fail")
                row["integration"] = "premature_finish_negative_control_passed"
        rows.append(row)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout.strip()
    report = {
        "status": "infrastructure_passed_certification_pending",
        "evaluation_ready": False,
        "code_revision": revision,
        "working_tree_dirty": bool(dirty),
        "instances": len(rows),
        "grounding_replays": sum(r["integration"] == "grounding_reference_passed" for r in rows),
        "logical_provider_gates": sum(
            r["integration"] == "blocked_missing_logical_provider" for r in rows
        ),
        "full_game_negative_controls": sum(
            r["integration"] == "premature_finish_negative_control_passed" for r in rows
        ),
        "rows": rows,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Fresh directory for evidence")
    parser.add_argument(
        "--require-certified",
        action="store_true",
        help="Exit 2 while the evaluation release remains uncertified",
    )
    args = parser.parse_args()
    report = check(Path("tasks/benchmark"), Path("data/benchmark/pilot"), args.output)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    raise SystemExit(2 if args.require_certified and not report["evaluation_ready"] else 0)
