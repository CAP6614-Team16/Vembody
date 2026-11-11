import json
from pathlib import Path

import pytest

from vembody.cli import main
from vembody.errors import TaskLoadError
from vembody.evaluation.analysis import analyze
from vembody.evaluation.harness import ExperimentConfig, run_experiment
from vembody.models.baselines import RandomCellModel
from vembody.models.mock import MockModel
from vembody.sandbox import MinesweeperSandbox
from vembody.tasks import load_task

TASK = "tasks/minesweeper-example.yaml"


def test_experiment_writes_manifest_logs_and_rows(tmp_path: Path) -> None:
    config = ExperimentConfig(
        "solver-smoke",
        "solver",
        [TASK],
        trials=3,
        vary_seeds=True,
        max_steps=100,
        labels={"memory": "off"},
    )
    rows = run_experiment(config, tmp_path / "exp")

    assert [row["board_seed"] for row in rows] == [7, 8, 9]
    assert all(row["label_memory"] == "off" and row["max_steps"] == 100 for row in rows)
    assert all(0 <= row["progress"] <= 1 for row in rows)
    manifest = json.loads((tmp_path / "exp" / "manifest.json").read_text())
    assert manifest["config"]["trials"] == 3 and len(manifest["tasks"][0]["sha256"]) == 64
    assert len((tmp_path / "exp" / "episodes.jsonl").read_text().splitlines()) == 3
    assert len(list((tmp_path / "exp" / "logs").rglob("trial_*.jsonl"))) == 3

    # Refuse to mix new results into an existing experiment
    with pytest.raises(FileExistsError):
        run_experiment(config, tmp_path / "exp")


def test_same_seeds_reproduce_results(tmp_path: Path) -> None:
    config = ExperimentConfig("a", "random-cell", [TASK], trials=3, vary_seeds=True)
    first = run_experiment(config, tmp_path / "a")
    second = run_experiment(config, tmp_path / "b")
    keys = ("progress", "steps", "evaluation_status")
    assert [[r[k] for k in keys] for r in first] == [[r[k] for k in keys] for r in second]


def test_random_cell_reads_hidden_cells_from_pixels(tmp_path: Path) -> None:
    task = load_task(TASK)
    sandbox = MinesweeperSandbox(task, tmp_path)
    model = RandomCellModel(task, seed=0)
    path = tmp_path / "before.png"
    sandbox.screen.screenshot().save(path)
    assert len(model.hidden_cells(str(path))) == 81

    sandbox.screen.game.reveal((4, 4))
    sandbox.screen.screenshot().save(path)
    hidden = set(model.hidden_cells(str(path)))
    game = sandbox.screen.game
    expected = {(r, c) for r in range(9) for c in range(9)} - game.revealed
    assert hidden == expected


def test_cli_eval_and_analyze(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "beginner.yaml").write_text(Path(TASK).read_text())
    for model in ("random-cell", "solver"):
        assert (
            main(
                [
                    "eval",
                    "--tasks",
                    str(suite),
                    "--model",
                    model,
                    "--trials",
                    "2",
                    "--name",
                    model,
                    "--out",
                    str(runs),
                ]
            )
            == 0
        )
    result = analyze([runs / "random-cell", runs / "solver"], tmp_path / "results")
    assert {row["experiment"] for row in result["summary"]} == {"random-cell", "solver"}
    for name in ("episodes.csv", "summary.csv", "by_task.csv", "summary.md"):
        assert (tmp_path / "results" / name).exists()


def test_eval_rejects_non_minesweeper_tasks(tmp_path: Path) -> None:
    config = ExperimentConfig("x", "mock", ["tasks/example.yaml"])
    with pytest.raises(ValueError, match="minesweeper"):
        run_experiment(config, tmp_path / "x")


def test_plots_render(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    run_experiment(ExperimentConfig("s", "solver", [TASK], trials=2), tmp_path / "s")
    result = analyze([tmp_path / "s"], tmp_path / "out")
    assert {p.name for p in result["plots"]} == {"success_rate.png", "progress.png", "latency.png"}


def test_benchmark_trials_preserve_frozen_board_and_vary_model_seed(tmp_path):
    path = "tasks/benchmark/development/t1-01.yaml"
    original = load_task(path)
    config = ExperimentConfig("frozen", "mock", [path], trials=3, vary_seeds=True, base_seed=10)
    rows = run_experiment(config, tmp_path / "frozen")
    assert all(row["success"] for row in rows)
    assert [row["board_seed"] for row in rows] == [original.minesweeper["seed"]] * 3
    assert [row["model_seed"] for row in rows] == [10, 11, 12]
    assert all(row["final_status"] == "playing" for row in rows)
    manifest = json.loads((tmp_path / "frozen" / "manifest.json").read_text())
    assert manifest["seed_policy"]["benchmark_board"] == "frozen_instance"


def test_interrupted_experiment_cannot_reuse_partial_output(tmp_path):
    class Interrupted(MockModel):
        def generate_action(self, task, observation, history):
            if history:
                raise KeyboardInterrupt
            return super().generate_action(task, observation, history)

    out = tmp_path / "interrupted"
    config = ExperimentConfig("interrupted", "mock", [TASK])
    with pytest.raises(KeyboardInterrupt):
        run_experiment(
            config, out, model_factory=lambda task, *_: Interrupted(list(task.mock_actions))
        )

    assert not (out / "episodes.jsonl").exists()
    old_files = {
        str(path.relative_to(out)): path.read_bytes() for path in out.rglob("*") if path.is_file()
    }
    assert any(name.endswith("trial_000.jsonl") for name in old_files)
    with pytest.raises(FileExistsError):
        run_experiment(config, out)

    assert old_files == {
        str(path.relative_to(out)): path.read_bytes() for path in out.rglob("*") if path.is_file()
    }


@pytest.mark.parametrize("paths, error", [([], ValueError), ([TASK, TASK], TaskLoadError)])
def test_invalid_task_collection_does_not_create_output(tmp_path, paths, error):
    with pytest.raises(error):
        run_experiment(ExperimentConfig("invalid", "mock", paths), tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("trials", [0, -1])
def test_nonpositive_trial_count_does_not_create_output(tmp_path, trials):
    with pytest.raises(ValueError, match="trials"):
        run_experiment(ExperimentConfig("invalid", "mock", [TASK], trials=trials), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_mine_after_ineffective_action_is_not_recovery(tmp_path):
    actions = [
        {"type": "click", "x": 156 / 311, "y": 196 / 351},
        {"type": "click", "x": 0, "y": 0},
        {"type": "click", "x": 92 / 311, "y": 228 / 351},
    ]
    rows = run_experiment(
        ExperimentConfig("fatal", "mock", [TASK]),
        tmp_path / "fatal",
        model_factory=lambda *_: MockModel(actions),
    )
    assert rows[0]["final_status"] == "lost"
    assert rows[0]["setbacks"] == 1
    assert rows[0]["recoveries"] == 0
    assert rows[0]["recovery_rate"] == 0
