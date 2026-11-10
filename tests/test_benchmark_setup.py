import copy
from dataclasses import replace

import pytest
import yaml

from vembody.benchmark.environment import prepare, snapshot, state_hash
from vembody.errors import TaskLoadError
from vembody.tasks import agent_task, load_task, load_task_set


def task_data():
    return {
        "id": "development-T4-02-001",
        "name": "Full board",
        "instruction": "Clear the board.",
        "environment": "minesweeper",
        "reset_instructions": "private setup",
        "success_description": "private predicate",
        "max_steps": 120,
        "timeout_seconds": 1800,
        "allowed_actions": ["click", "right_click", "finish"],
        "minesweeper": {"rows": 9, "cols": 9, "mines": 10, "seed": 7},
        "benchmark": {
            "version": 1,
            "template_id": "T4-02",
            "split": "development",
            "opening": [4, 4],
            "predicate": {"type": "board_won"},
        },
    }


def write_task(tmp_path, data=None):
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(data or task_data()), encoding="utf-8")
    return load_task(path)


def test_reset_state_and_pixels_are_independent_of_previous_episode(tmp_path):
    task = write_task(tmp_path)
    first = prepare(task)
    digest = state_hash(first)
    pixels = first.screenshot().tobytes()
    target = next(iter(first.game.mines))
    first.game.toggle_flag(target)
    again = prepare(task)
    assert state_hash(again) == digest
    assert again.screenshot().tobytes() == pixels
    changed_opening = replace(task, benchmark=replace(task.benchmark, opening=(0, 0)))
    assert prepare(changed_opening).game.mines != again.game.mines


def test_ordered_setup_and_hash_mismatch(tmp_path):
    task = write_task(tmp_path)
    target = next(iter(prepare(task).game.mines))
    data = task_data()
    data["benchmark"]["setup"] = [{"type": "right_click", "cell": list(target)}]
    task = write_task(tmp_path, data)
    screen = prepare(task)
    assert screen.game.flagged == {target}
    pinned = replace(
        task, benchmark=replace(task.benchmark, expected_state_hash=state_hash(screen))
    )
    assert snapshot(prepare(pinned)) == snapshot(screen)
    with pytest.raises(TaskLoadError, match="Frozen"):
        prepare(replace(pinned, benchmark=replace(pinned.benchmark, expected_state_hash="0" * 64)))
    data["benchmark"]["setup"] = [{"type": "click", "cell": list(target)}]
    with pytest.raises(TaskLoadError, match="playing"):
        prepare(write_task(tmp_path, data))


def test_geometry_labels_boundaries_and_disabled_chord(tmp_path):
    task = write_task(tmp_path)
    raw = prepare(task)
    labeled = prepare(replace(task, benchmark=replace(task.benchmark, grid_labels=True)))
    assert raw.screenshot().size == labeled.screenshot().size == (336, 376)
    box = (raw.grid_left, raw.grid_top, raw.width, raw.height)
    assert raw.screenshot().crop(box).tobytes() == labeled.screenshot().crop(box).tobytes()
    for row in range(9):
        for col in range(9):
            assert raw.cell_at(*raw.cell_center((row, col))) == (row, col)
    assert raw.cell_at(raw.grid_left, raw.grid_top) is None
    before = state_hash(raw)
    raw.click(*raw.cell_center((4, 4)), "left")
    assert state_hash(raw) == before
    with pytest.raises(ValueError, match="interior"):
        raw.click(raw.face[0] + 5, raw.face[1] + 5, "left")
    assert state_hash(raw) == before


@pytest.mark.parametrize(
    "key,value",
    [
        ("version", True),
        ("opening", [9, 1]),
        ("opening", [True, 0]),
        ("opening", "C5"),
        ("split", "other"),
        ("grid_labels", 1),
        ("setup", [{"type": "finish", "cell": [1, 1]}]),
        ("predicate", {"type": "arbitrary_code"}),
        ("expected_state_hash", "bad"),
    ],
)
def test_rejects_malformed_benchmark(tmp_path, key, value):
    data = task_data()
    data["benchmark"][key] = value
    with pytest.raises(TaskLoadError):
        write_task(tmp_path, data)


@pytest.mark.parametrize(
    "update",
    [
        {"rows": 0},
        {"mines": 81},
        {"seed": False},
        {"cols": 5},
    ],
)
def test_rejects_invalid_board(tmp_path, update):
    data = task_data()
    data["minesweeper"].update(update)
    with pytest.raises(TaskLoadError):
        write_task(tmp_path, data)


def test_invalid_budget_duplicate_ids_and_public_projection(tmp_path):
    data = task_data()
    data["timeout_seconds"] = float("nan")
    with pytest.raises(TaskLoadError):
        write_task(tmp_path, data)
    task = write_task(tmp_path)
    second = tmp_path / "second.yaml"
    second.write_text((tmp_path / "task.yaml").read_text(), encoding="utf-8")
    with pytest.raises(TaskLoadError, match="Duplicate"):
        load_task_set([tmp_path / "task.yaml", second])
    public = agent_task(task)
    assert public.benchmark is None and public.minesweeper is None
    assert not public.mock_actions and "private" not in repr(public)
    assert public.instruction == task.instruction


def test_evaluation_requires_frozen_state(tmp_path):
    data = copy.deepcopy(task_data())
    data["benchmark"]["split"] = "evaluation"
    with pytest.raises(TaskLoadError, match="frozen"):
        write_task(tmp_path, data)


def test_mixed_key_types_are_task_errors(tmp_path):
    data = task_data()
    data["benchmark"].update({1: "bad", "unknown": "bad"})
    with pytest.raises(TaskLoadError, match="strings"):
        write_task(tmp_path, data)


def test_cli_rejects_uncertified_evaluation_benchmark(tmp_path):
    from vembody.benchmark.labels import CertificationRequired
    from vembody.cli import build_loop, build_parser

    task = write_task(tmp_path)
    data = task_data()
    data["benchmark"].update(split="evaluation", expected_state_hash=state_hash(prepare(task)))
    write_task(tmp_path, data)
    args = build_parser().parse_args(["run", "--task", str(tmp_path / "task.yaml")])
    with pytest.raises(CertificationRequired, match="certificates"):
        build_loop(args)


def test_loop_does_not_pass_private_task_or_evaluation_to_model(tmp_path):
    from vembody.evaluation.base import NullEvaluator
    from vembody.logging.jsonl import JsonlLogger
    from vembody.loop import AgentLoop
    from vembody.models.mock import MockModel
    from vembody.results import EvaluationResult, EvaluationStatus
    from vembody.sandbox import MinesweeperSandbox
    from vembody.validation import ActionPolicy

    task = write_task(tmp_path)
    screen = prepare(task)
    px, py = screen.cell_center((4, 4))

    class SpyModel(MockModel):
        def generate_action(self, public, observation, history):
            assert public.benchmark is None and public.minesweeper is None
            assert not public.mock_actions
            assert all(item.evaluation is None for item in history)
            assert all(not item.execution.evidence for item in history if item.execution)
            return super().generate_action(public, observation, history)

    class PrivateEvaluationLogger(JsonlLogger):
        def write_step(self, result):
            super().write_step(result)
            result.evaluation = EvaluationResult(EvaluationStatus.UNKNOWN, "PRIVATE_SENTINEL")
            if result.execution:
                result.execution = replace(result.execution, evidence={"mines": "PRIVATE_SENTINEL"})

    model = SpyModel(
        [
            {"type": "click", "x": px / (screen.width - 1), "y": py / (screen.height - 1)},
            {"type": "finish"},
        ]
    )
    sandbox = MinesweeperSandbox(task, tmp_path / "images")
    loop = AgentLoop(
        model,
        sandbox,
        sandbox,
        NullEvaluator(),
        ActionPolicy.from_task(task),
        PrivateEvaluationLogger(tmp_path / "episode.jsonl"),
    )
    assert len(loop.run_episode(task).steps) == 2
