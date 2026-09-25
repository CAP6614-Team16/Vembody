import pytest

from vembody.actions import action_from_dict
from vembody.errors import ExecutionError
from vembody.execution.desktop import DesktopExecutor
from vembody.execution.dry_run import DryRunExecutor


def test_dry_run_records_action_without_live_input() -> None:
    executor = DryRunExecutor()
    result = executor.execute(action_from_dict({"type": "finish"}))
    assert result.succeeded
    assert executor.events == [{"type": "finish"}]


def test_desktop_executor_is_disabled() -> None:
    executor = DesktopExecutor()
    with pytest.raises(ExecutionError):
        executor.execute(action_from_dict({"type": "finish"}))
