import importlib.util
from pathlib import Path

import pytest
from PIL import Image

from vembody.errors import TaskLoadError


def load_exporter():
    spec = importlib.util.spec_from_file_location("exporter", "scripts/export_benchmark_assets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_assets_cover_all_templates_and_keep_examples_separate(tmp_path):
    exporter = load_exporter()
    root = Path("tasks/benchmark")
    manifest = exporter.export([root / "development", root / "demonstration"], tmp_path)
    assert len(manifest["instances"]) == 24
    assert {r["split"] for r in manifest["instances"]} == {"development", "demonstration"}
    for split in ("development", "demonstration"):
        instances = [r for r in manifest["instances"] if r["split"] == split]
        assert len({r["template_id"] for r in instances}) == 12
    assert len({r["layout_hash"] for r in manifest["instances"]}) == 24
    assert len({r["visible_hash"] for r in manifest["instances"]}) == 24
    for record in manifest["instances"]:
        for image in record["images"].values():
            path = tmp_path / image["path"]
            assert exporter.digest(path.read_bytes()) == image["file_sha256"]
            with Image.open(path) as picture:
                assert picture.mode == "RGB"
                assert picture.size == (record["width"], record["height"])
                assert exporter.digest(picture.tobytes()) == image["pixel_sha256"]
    assert len([e for e in manifest["examples"] if e["response"]]) == 4
    assert all(e["instance_id"].startswith("demonstration-") for e in manifest["examples"])
    assert all(
        e["status"] == "pending_solver_example"
        for e in manifest["examples"]
        if not e["template_id"].startswith("T1")
    )


def test_duplicate_split_inputs_are_rejected(tmp_path):
    exporter = load_exporter()
    source = Path("tasks/benchmark/development")
    with pytest.raises(TaskLoadError, match="Duplicate"):
        exporter.export([source, source], tmp_path)


def test_distinct_hidden_layouts_with_same_visible_start_are_rejected(tmp_path):
    import yaml

    from vembody.benchmark.environment import prepare
    from vembody.benchmark.labels import visible_hash
    from vembody.tasks import load_task

    data = yaml.safe_load(Path("tasks/benchmark/development/t4-02.yaml").read_text())
    data["benchmark"].pop("expected_state_hash")
    data["benchmark"]["opening"] = [4, 4]
    source = tmp_path / "source"
    source.mkdir()
    seen = {}
    for seed in range(100):
        data["id"] = f"different-{seed}"
        data["minesweeper"]["seed"] = seed
        path = tmp_path / "candidate.yaml"
        text = yaml.safe_dump(data)
        path.write_text(text, encoding="utf-8")
        key = visible_hash(prepare(load_task(path)))
        if key in seen:
            (source / "first.yaml").write_text(seen[key], encoding="utf-8")
            (source / "second.yaml").write_text(text, encoding="utf-8")
            break
        seen[key] = text
    else:
        pytest.fail("Could not find duplicate visible fixture")
    with pytest.raises(ValueError, match="Duplicate"):
        load_exporter().export([source], tmp_path / "out")


def test_instance_id_cannot_escape_asset_directory(tmp_path):
    import yaml

    from vembody.tasks import load_task

    data = yaml.safe_load(Path("tasks/benchmark/development/t1-01.yaml").read_text())
    data["id"] = "../outside"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(TaskLoadError, match="safe filename"):
        load_task(path)
