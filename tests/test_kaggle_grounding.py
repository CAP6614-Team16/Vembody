import json
from pathlib import Path

import pytest

from scripts.kaggle_grounding import run_grounding
from vembody.models.base import ModelResponse
from vembody.models.mock import MockModel


@pytest.mark.parametrize(
    "mode,trials", [("correct", 3), ("wrong", 1), ("invalid", 1), ("finish", 1)]
)
def test_grounding_suite_measures_outcomes_without_a_gpu(tmp_path, monkeypatch, mode, trials):
    """Exercise all four real task definitions and the existing loop/scorer/logger offline."""
    monkeypatch.setenv("VEMBODY_KAGGLE_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("VEMBODY_KAGGLE_MODEL", "offline-fixture")
    monkeypatch.setenv("VEMBODY_KAGGLE_API_KEY", "offline-test-key")

    class InvalidModel:
        def generate_action(self, *args):
            """Return an intentionally unparseable response for reliability accounting."""
            return ModelResponse("not JSON", 0.01, {"provider": "mock"})

    def factory(task, sandbox, seed):
        """Use task oracle actions only in this test factory, never in the actual VLM prompt."""
        match mode:
            case "correct":
                return MockModel.from_task(task)
            case "wrong":
                return MockModel([{"type": "click", "x": 411 / 999, "y": 264 / 999}])
            case "invalid":
                return InvalidModel()
            case "finish":
                return MockModel([{"type": "finish"}])

    output = tmp_path / "grounding"
    summary = run_grounding(Path.cwd(), output, trials=trials, model_factory=factory)
    assert summary["episodes"] == 4 * trials
    assert summary["infrastructure_failures"] == 0
    assert len(summary["trials"]) == 4 * trials
    assert summary["first_response_valid_rate"] == (0 if mode == "invalid" else 1)
    assert summary["first_action_accuracy"] == (1 if mode == "correct" else 0)
    assert summary["episode_success_rate"] == (1 if mode == "correct" else 0)
    assert summary["coordinate_error_samples"] == (
        4 * trials if mode in {"correct", "wrong"} else 0
    )
    assert json.loads((output / "grounding-summary.json").read_text()) == summary
    if mode == "correct":
        assert summary["mean_first_click_center_error_pixels"] == 0
    elif mode == "wrong":
        assert summary["mean_first_click_center_error_pixels"] > 0

    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["tasks"]) == 4
    assert not manifest["config"]["vary_seeds"]
