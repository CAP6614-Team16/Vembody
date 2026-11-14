import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from scripts import kaggle_run as runner
from scripts.kaggle_autonomous import run_autonomous
from vembody.models.base import ModelResponse
from vembody.models.mock import MockModel


@pytest.mark.parametrize("mode", ["oracle", "false-finish", "miss", "truncated"])
def test_autonomous_play_is_goal_driven_bounded_and_independently_graded(
    tmp_path, monkeypatch, mode
):
    monkeypatch.setenv("VEMBODY_KAGGLE_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("VEMBODY_KAGGLE_MODEL", "offline")
    monkeypatch.setenv("VEMBODY_KAGGLE_API_KEY", "offline-test")

    class TruncatedModel:
        def generate_action(self, *args):
            """Reproduce the pilot's exhausted budget without issuing an action."""
            return ModelResponse(
                '{"ui_mate_parse_error": "truncated"}', 0.01, {"finish_reason": "length"}
            )

    def factory(task, sandbox, seed):
        """Use hidden-state oracle actions solely as an offline evaluator test fixture."""
        if mode == "false-finish":
            return MockModel([{"type": "finish"}])
        elif mode == "miss":
            return MockModel([{"type": "click", "x": 0, "y": 0}])
        elif mode == "truncated":
            return TruncatedModel()

        actions = []
        screen = sandbox.screen
        game = screen.game
        for row in range(game.rows):
            for col in range(game.cols):
                cell = (row, col)
                if cell in game.mines or cell in game.revealed:
                    continue
                x, y = screen.cell_center(cell)
                actions.append(
                    {"type": "click", "x": x / (screen.width - 1), "y": y / (screen.height - 1)}
                )
        return MockModel(actions)

    out = tmp_path / "autonomous"
    summary = run_autonomous(Path.cwd(), out, model_factory=factory)
    assert summary["episodes"] == 1
    assert summary["success_rate"] == (1 if mode == "oracle" else 0)
    assert summary["infrastructure_failures"] == 0
    assert summary["false_completion_claims"] == (1 if mode == "false-finish" else 0)
    if mode == "miss":
        assert summary["model_calls"] == summary["off_grid_clicks"] == 40
    elif mode == "truncated":
        assert summary["model_calls"] == summary["truncated_responses"] == 1
        assert summary["parse_errors"] == 1
        assert summary["actions_executed"] == 0

    assert summary["response_profile"] == "brief"

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["config"]["timeout_seconds"] is None
    assert manifest["config"]["max_steps"] is None
    assert manifest["model_endpoint"]["max_tokens"] == 4096
    assert manifest["model_endpoint"]["prompt_version"] == "ui-mate-minesweeper-v3-brief"
    row = json.loads((out / "episodes.jsonl").read_text())
    assert row["max_steps"] == 40
    log = next((out / "logs").rglob("*.jsonl"))
    episode = json.loads(log.read_text().splitlines()[-1])["episode"]
    assert len(episode["steps"]) <= 40


def test_runner_defaults_to_one_autonomous_game_without_a_targeting_smoke(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    (repo / "configs/kaggle-ui-mate.json").write_text("{}")
    (repo / "bundle-manifest.json").write_text("{}")
    output = tmp_path / "output"
    calls = []

    @contextmanager
    def serve(runtime, path):
        """Simulate a server context without a GPU or model process."""
        path.mkdir()
        yield

    monkeypatch.setattr(runner, "load_runtime", lambda *_: None)
    monkeypatch.setattr(runner, "serve_model", serve)
    monkeypatch.setattr(runner, "check_contract", lambda: {"status": "fixture"})
    monkeypatch.setattr(runner, "run_autonomous", lambda *args, **kw: calls.append((args, kw)))
    monkeypatch.setattr(
        runner, "smoke_model", lambda *_: pytest.fail("No targeting smoke in gameplay")
    )
    assert (
        runner.main(
            [
                "autonomous",
                "--repo",
                str(repo),
                "--work",
                str(tmp_path / "work"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert calls == [((repo, output / "autonomous", 1, "ui-mate"), {"profile": "brief"})]
    compatibility = json.loads((output / "compatibility.json").read_text())
    assert compatibility["selected_adapter"] == "ui-mate-brief"
    assert compatibility["selected_prompt_version"] == "ui-mate-minesweeper-v3-brief"


@pytest.mark.parametrize(
    "profile,model,tokens,timeout,prompt",
    [
        ("original", "ui-mate", 2048, 180, "ui-mate-minesweeper-v2"),
        ("budget-only", "ui-mate", 4096, 240, "ui-mate-minesweeper-v2"),
        ("brief", "ui-mate-brief", 4096, 240, "ui-mate-minesweeper-v3-brief"),
    ],
)
def test_autonomous_profiles_record_distinct_reproducible_settings(
    tmp_path, monkeypatch, profile, model, tokens, timeout, prompt
):
    monkeypatch.setenv("VEMBODY_KAGGLE_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("VEMBODY_KAGGLE_MODEL", "offline")
    monkeypatch.setenv("VEMBODY_KAGGLE_API_KEY", "offline-test")
    output = tmp_path / profile
    run_autonomous(
        Path.cwd(),
        output,
        profile=profile,
        model_factory=lambda *_: MockModel([{"type": "finish"}]),
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["config"]["model"] == model
    assert manifest["config"]["labels"]["response_profile"] == profile
    endpoint = manifest["model_endpoint"]
    assert endpoint["max_tokens"] == tokens
    assert endpoint["timeout_seconds"] == timeout
    assert endpoint["prompt_version"] == prompt
    assert endpoint["temperature"] == 0


def test_unsupported_profile_combinations_fail_before_starting_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "load_runtime", lambda *_: pytest.fail("Must validate first"))
    with pytest.raises(SystemExit):
        runner.main(
            [
                "autonomous",
                "--repo",
                str(tmp_path),
                "--work",
                str(tmp_path),
                "--output",
                str(tmp_path / "out"),
                "--adapter",
                "ui-mate-legacy",
            ]
        )
    with pytest.raises(ValueError, match="profile"):
        run_autonomous(tmp_path, tmp_path / "out", profile="unknown")
    with pytest.raises(ValueError, match="requires"):
        run_autonomous(tmp_path, tmp_path / "out", adapter="ui-mate-legacy", profile="brief")
