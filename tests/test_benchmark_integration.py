import importlib.util
import json
from pathlib import Path

import pytest

from vembody.benchmark.replay import replay_grounding
from vembody.tasks import load_task


def test_full_candidate_check_and_tamper_detection(tmp_path):
    spec = importlib.util.spec_from_file_location("checker", "scripts/check_benchmark.py")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    report = checker.check(Path("tasks/benchmark"), Path("data/benchmark/pilot"), tmp_path)
    assert report["instances"] == 24 and report["grounding_replays"] == 8
    assert not report["evaluation_ready"]
    gates = [r for r in report["rows"] if r["integration"] == "blocked_missing_logical_provider"]
    assert len(gates) == 12
    task = load_task("tasks/benchmark/development/t1-01.yaml")
    path = tmp_path / f"{task.id}.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["step"]["execution"]["evidence"]["task_success"] = False
    # Tamper both copies to exercise recomputation rather than copy disagreement.
    records[-1]["episode"]["steps"][0]["execution"]["evidence"]["task_success"] = False
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Trace mismatch"):
        replay_grounding(task, path)


def test_certified_release_gate_returns_nonzero(tmp_path):
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_benchmark.py",
            "--output",
            str(tmp_path),
            "--require-certified",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert json.loads(result.stdout)["evaluation_ready"] is False


@pytest.mark.parametrize("bad_keys", [[], ["raw"], ["raw", "grid", "other"]])
def test_checker_rejects_missing_or_unknown_image_variants(tmp_path, bad_keys):
    import shutil

    assets = tmp_path / "assets"
    shutil.copytree("data/benchmark/pilot", assets)
    path = assets / "manifest.json"
    data = json.loads(path.read_text())
    original = data["instances"][0]["images"]
    data["instances"][0]["images"] = {key: original.get(key, {}) for key in bad_keys}
    path.write_text(json.dumps(data), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("checker", "scripts/check_benchmark.py")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    with pytest.raises(ValueError, match="exactly raw and grid"):
        checker.check(Path("tasks/benchmark"), assets, tmp_path / "logs")


def test_replay_rejects_image_with_identical_bytes_but_wrong_dimensions(tmp_path):
    from PIL import Image

    from vembody.cli import build_loop, build_parser

    task_path = Path("tasks/benchmark/development/t1-01.yaml")
    log = tmp_path / "run.jsonl"
    loop, task = build_loop(
        build_parser().parse_args(["run", "--task", str(task_path), "--log", str(log)])
    )
    episode = loop.run_episode(task)
    path = Path(episode.steps[0].execution.evidence["post_screenshot_path"])
    with Image.open(path) as picture:
        reshaped = Image.frombytes(
            "RGB", (picture.width * 2, picture.height // 2), picture.tobytes()
        )
    reshaped.save(path)
    with pytest.raises(ValueError, match="Post-action image"):
        replay_grounding(task, log)
