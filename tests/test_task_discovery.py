from pathlib import Path

import pytest

from vembody.cli import main
from vembody.errors import TaskLoadError
from vembody.tasks import discover_task_paths, load_task_set


def test_benchmark_root_selects_development_manifest():
    paths = discover_task_paths(["tasks/benchmark"])
    tasks = load_task_set(paths)
    assert len(tasks) == 12
    assert {task.benchmark.split for task in tasks} == {"development"}
    assert all(Path(path).suffix == ".yaml" for path in paths)


def test_split_directory_and_manifest_resolve_the_same_tasks():
    directory = discover_task_paths(["tasks/benchmark/development"])
    manifest = discover_task_paths(["tasks/benchmark/development/manifest.json"])
    assert directory == manifest
    assert len(directory) == 12


def test_explicit_demonstration_split_does_not_mix_development():
    paths = discover_task_paths(["tasks/benchmark"], split="demonstration")
    assert {task.benchmark.split for task in load_task_set(paths)} == {"demonstration"}


def test_empty_folder_is_an_error_and_creates_no_experiment(tmp_path, capsys):
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    assert (
        main(
            [
                "eval",
                "--tasks",
                str(tasks),
                "--model",
                "mock",
                "--name",
                "empty",
                "--out",
                str(tmp_path / "runs"),
            ]
        )
        == 1
    )
    assert "No task files found" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def test_missing_split_is_not_replaced_with_another_split():
    with pytest.raises(TaskLoadError, match="split does not exist"):
        discover_task_paths(["tasks/benchmark"], split="evaluation")
