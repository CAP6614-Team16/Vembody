import json
from pathlib import Path

from vembody.cli import build_loop, build_parser
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
