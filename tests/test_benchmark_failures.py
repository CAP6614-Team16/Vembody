from dataclasses import replace
from pathlib import Path

from vembody.benchmark.environment import state_hash
from vembody.cli import build_loop, build_parser
from vembody.models.base import ModelResponse
from vembody.results import EvaluationStatus, TerminationReason

PROVIDER_MESSAGE = "This task requires Nicholas's logical label provider"
ROOT = Path("tasks/benchmark/development")


def loop_for(tmp_path, template="t1-01"):
    return build_loop(
        build_parser().parse_args(
            [
                "run",
                "--task",
                f"tasks/benchmark/development/{template}.yaml",
                "--log",
                str(tmp_path / "episode.jsonl"),
            ]
        )
    )


def test_action_limit_and_ineffective_input(tmp_path):
    loop, task = loop_for(tmp_path)
    px, py = loop.executor.screen.cell_center(task.benchmark.opening)
    loop.model._action_payloads = [
        {
            "type": "click",
            "x": px / (loop.executor.screen.width - 1),
            "y": py / (loop.executor.screen.height - 1),
        }
    ]
    result = loop.run_episode(replace(task, max_steps=1))
    assert result.termination_reason == TerminationReason.ACTION_LIMIT
    assert result.evaluation.status == EvaluationStatus.FAILURE
    assert not result.steps[0].execution.evidence["effective"]


def test_off_target_click_is_a_task_failure_not_an_infrastructure_error(tmp_path):
    """A miss must be graded, not excluded from results as a harness error."""
    loop, task = loop_for(tmp_path)
    screen = loop.executor.screen
    gutter = (5, 5)
    border = (screen.grid_left + 3 * 32, screen.grid_top + 3 * 32)
    for index, (px, py) in enumerate((gutter, border)):
        loop, task = loop_for(tmp_path / f"miss{index}")
        screen = loop.executor.screen
        assert screen.cell_at(px, py) is None
        before = state_hash(screen)
        loop.model._action_payloads = [
            {"type": "click", "x": px / (screen.width - 1), "y": py / (screen.height - 1)}
        ]
        result = loop.run_episode(replace(task, max_steps=1))
        assert result.termination_reason == TerminationReason.ACTION_LIMIT
        assert result.evaluation.status == EvaluationStatus.FAILURE
        evidence = result.steps[0].execution.evidence
        assert evidence["off_target"] and not evidence["effective"]
        assert evidence["resolved_cell"] is None
        assert state_hash(screen) == before


def test_episode_continues_after_a_miss_and_still_spends_budget(tmp_path):
    loop, task = loop_for(tmp_path)
    screen = loop.executor.screen
    loop.model._action_payloads = [
        {"type": "click", "x": 5 / (screen.width - 1), "y": 5 / (screen.height - 1)},
        dict(task.mock_actions[0]),
    ]
    result = loop.run_episode(task)
    assert len(result.steps) == 2
    assert result.steps[0].execution.evidence["off_target"]
    assert not result.steps[1].execution.evidence["off_target"]
    assert result.termination_reason == TerminationReason.TASK_COMPLETE
    assert result.evaluation.status == EvaluationStatus.SUCCESS


def test_blocked_benchmark_run_reports_one_line_without_a_traceback(tmp_path, capsys):
    from vembody.cli import main

    code = main(
        [
            "run",
            "--task",
            "tasks/benchmark/development/t2-01.yaml",
            "--log",
            str(tmp_path / "episode.jsonl"),
        ]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.splitlines() == ["error: CertificationRequired: " + PROVIDER_MESSAGE]
    assert not captured.out

    code = main(["run", "--task", str(tmp_path / "missing.yaml")])
    assert code == 1
    assert capsys.readouterr().err.startswith("error: TaskLoadError:")


def test_module_entry_point_propagates_the_exit_code(tmp_path):
    """A refused run must not look successful to a script or CI step."""
    import os
    import subprocess
    import sys

    env = dict(os.environ, PYTHONPATH=str(Path("src").resolve()))
    for task, expected in ((ROOT / "t1-01.yaml", 0), (ROOT / "t2-01.yaml", 1)):
        done = subprocess.run(
            [
                sys.executable,
                "-m",
                "vembody",
                "run",
                "--task",
                str(task),
                "--log",
                str(tmp_path / "e.jsonl"),
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert done.returncode == expected, done.stderr
        assert "Traceback" not in done.stderr


def test_model_latency_cannot_dispatch_late_action(tmp_path, monkeypatch):
    loop, task = loop_for(tmp_path)
    clock = [100]
    monkeypatch.setattr("vembody.loop.perf_counter", lambda: clock[0])
    original = loop.model.generate_action

    def slow(*args):
        clock[0] += task.timeout_seconds + 1
        return original(*args)

    loop.model.generate_action = slow
    before = set(loop.executor.screen.game.revealed)
    result = loop.run_episode(task)
    assert result.termination_reason == TerminationReason.TIMEOUT
    assert result.evaluation.status == EvaluationStatus.FAILURE
    assert result.steps[0].execution is None
    assert loop.executor.screen.game.revealed == before


def test_parse_failure_never_executes_input(tmp_path):
    loop, task = loop_for(tmp_path)
    loop.model.generate_action = lambda *args: ModelResponse("not json")
    result = loop.run_episode(task)
    assert result.termination_reason == TerminationReason.PARSE_ERROR
    assert result.steps[0].execution is None
    assert result.evaluation.status == EvaluationStatus.FAILURE


def test_evaluator_and_cleanup_errors_keep_episode_summary(tmp_path):
    loop, task = loop_for(tmp_path)

    def broken_evaluate(*args):
        raise RuntimeError("test evaluator failure")

    loop.evaluator.evaluate = broken_evaluate
    result = loop.run_episode(task)
    assert result.termination_reason == TerminationReason.EVALUATION_ERROR
    assert result.evaluation.status == EvaluationStatus.NOT_EVALUATED
    assert '"record_type": "episode"' in (tmp_path / "episode.jsonl").read_text()
    other = tmp_path / "cleanup"
    loop, task = loop_for(other)

    def broken_shutdown():
        raise RuntimeError("test cleanup failure")

    loop.executor.shutdown = broken_shutdown
    result = loop.run_episode(task)
    assert result.termination_reason == TerminationReason.EXECUTION_ERROR
    assert result.steps[-1].error == "Shutdown failed: test cleanup failure"
    assert (other / "episode.jsonl").exists()


def test_premature_finish_and_wrong_action_type_fail(tmp_path):
    loop, task = loop_for(tmp_path)
    loop.model._action_payloads = [{"type": "finish"}]
    result = loop.run_episode(task)
    assert result.evaluation.status == EvaluationStatus.FAILURE
    assert result.termination_reason == TerminationReason.FINISH_REQUESTED
    loop, task = loop_for(tmp_path / "wrong")
    action = dict(task.mock_actions[0], type="right_click")
    loop.model._action_payloads = [action]
    result = loop.run_episode(task)
    assert result.evaluation.status == EvaluationStatus.FAILURE
    assert result.steps[0].execution.evidence["task_violation"]


def test_nested_action_log_uses_strict_action_schema(tmp_path):
    import json

    from vembody.actions import action_from_dict

    loop, task = loop_for(tmp_path)
    loop.run_episode(task)
    record = json.loads((tmp_path / "episode.jsonl").read_text().splitlines()[0])
    action = record["step"]["action"]
    assert set(action) == {"type", "x", "y"}
    assert action_from_dict(action).type.value == "click"


def test_capture_failure_remains_infrastructure_error(tmp_path):
    loop, task = loop_for(tmp_path)

    def broken_capture(*args):
        raise RuntimeError("test capture failure")

    loop.observations.capture = broken_capture
    result = loop.run_episode(task)
    assert result.termination_reason == TerminationReason.CAPTURE_ERROR
    assert result.evaluation.status == EvaluationStatus.NOT_EVALUATED
    assert result.steps[0].execution is None
