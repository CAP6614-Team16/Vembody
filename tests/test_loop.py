import json
from pathlib import Path

import pytest

from vembody.cli import build_loop, build_parser
from vembody.evaluation.metrics import episode_metrics
from vembody.models.base import ModelResponse
from vembody.models.mock import MockModel
from vembody.results import EvaluationStatus, TerminationReason


def test_mock_dry_run_episode(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "--task",
            "tasks/example.yaml",
            "--model",
            "mock",
            "--executor",
            "dry-run",
            "--log",
            str(tmp_path / "episode.jsonl"),
        ]
    )
    loop, task = build_loop(args)
    result = loop.run_episode(task)
    assert result.termination_reason is TerminationReason.FINISH_REQUESTED
    assert result.evaluation.status is EvaluationStatus.UNKNOWN
    records = (tmp_path / "episode.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(records) == 3
    assert json.loads(records[-1])["record_type"] == "episode"


@pytest.mark.parametrize(
    "task_path", ["tasks/minesweeper-example.yaml", "tasks/benchmark/development/t4-02.yaml"]
)
def test_private_execution_channels_stay_in_logs_only(tmp_path, task_path):
    """Both legacy and benchmark models receive history without evaluator counters."""
    log = tmp_path / "episode.jsonl"
    loop, task = build_loop(
        build_parser().parse_args(["run", "--task", task_path, "--log", str(log)])
    )
    checked = []

    class Spy(MockModel):
        def generate_action(self, public, observation, history):
            if history:
                checked.append(True)
                assert history[-1].execution.details == {}
                assert history[-1].execution.evidence == {}
                assert history[-1].evaluation is None
            return super().generate_action(public, observation, history)

    loop.model = Spy([{"type": "click", "x": 0, "y": 0}, {"type": "finish"}])
    result = loop.run_episode(task)
    assert checked
    assert "flags_correct" in result.steps[0].execution.details["board"]
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert records[0]["step"]["execution"]["details"]["board"]


@pytest.mark.parametrize(
    ("raw_text", "termination"),
    [
        ("not JSON", TerminationReason.PARSE_ERROR),
        ('{"action":{"type":"click","x":2,"y":0.5}}', TerminationReason.VALIDATION_ERROR),
    ],
)
def test_invalid_responses_keep_usage_metadata(tmp_path, raw_text, termination):
    log = tmp_path / "episode.jsonl"
    loop, task = build_loop(
        build_parser().parse_args(
            [
                "run",
                "--task",
                "tasks/minesweeper-example.yaml",
                "--log",
                str(log),
            ]
        )
    )
    usage = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120, "image_count": 1}
    loop.model.generate_action = lambda *args: ModelResponse(raw_text, 0.5, usage)
    result = loop.run_episode(task)
    assert result.termination_reason == termination
    assert result.steps[0].execution is None
    assert result.steps[0].model_metadata == usage
    records = [json.loads(line) for line in log.read_text().splitlines()]
    metrics = episode_metrics(records[-1]["episode"])
    assert metrics["prompt_tokens_total"] == 100
    assert metrics["completion_tokens_total"] == 20
    assert metrics["total_tokens_total"] == 120
    assert metrics["image_count_total"] == 1
    assert metrics["model_seconds_total"] >= 0.5
