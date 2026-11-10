"""Recompute development grounding outcomes without invoking a model."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from ..actions import action_from_dict
from ..tasks import TaskSpec


def replay_grounding(task: TaskSpec, log_path: Path) -> dict:
    from ..sandbox import MinesweeperSandbox

    if task.benchmark is None or not task.benchmark.template_id.startswith("T1"):
        raise ValueError("Replay currently supports grounding traces only")
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    episodes = [r["episode"] for r in records if r.get("record_type") == "episode"]
    if len(episodes) != 1 or episodes[0]["task_id"] != task.id:
        raise ValueError("Replay requires exactly one matching episode")
    episode = episodes[0]
    steps = [r["step"] for r in records if r.get("record_type") == "step"]
    if steps != episode["steps"]:
        raise ValueError("Step records and episode summary disagree")
    sandbox = MinesweeperSandbox(task, log_path.parent / "unused")
    for step in steps:
        if sandbox.scorer.done:
            raise ValueError("Trace continues after task termination")
        if not step.get("execution"):
            raise ValueError("Replay requires executed grounding steps")
        with Image.open(step["observation"]["screenshot_path"]) as before_image:
            if (
                before_image.mode != "RGB"
                or before_image.size != sandbox.screen.screenshot().size
                or before_image.tobytes() != sandbox.screen.screenshot().tobytes()
            ):
                raise ValueError("Pre-action image disagrees with replay")
        actual = sandbox.execute(action_from_dict(step["action"]))
        # Round-trip tuples so comparison matches the JSON representation.
        evidence = json.loads(json.dumps(actual.evidence))
        saved = step["execution"]["evidence"]
        for field in (
            "before",
            "after",
            "resolved_cell",
            "effective",
            "task_success",
            "task_violation",
            "logical_correct",
        ):
            if saved.get(field) != evidence[field]:
                raise ValueError(f"Trace mismatch: {field}")
        with Image.open(saved["post_screenshot_path"]) as after_image:
            if (
                after_image.mode != "RGB"
                or after_image.size != sandbox.screen.screenshot().size
                or after_image.tobytes() != sandbox.screen.screenshot().tobytes()
            ):
                raise ValueError("Post-action image disagrees with replay")
    outcome = sandbox.evaluate(task, None).status.value
    if outcome != episode["evaluation"]["status"]:
        raise ValueError("Replayed outcome disagrees with saved evaluation")
    return {"task_id": task.id, "replayed_status": outcome, "steps": len(steps)}
