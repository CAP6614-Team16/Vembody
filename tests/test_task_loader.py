from pathlib import Path

from vembody.tasks import load_task


def test_example_task_loads() -> None:
    task = load_task(Path("tasks/example.yaml"))
    assert task.id == "example-task-001"
    assert task.max_steps == 3
    assert len(task.mock_actions) == 2
