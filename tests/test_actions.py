import pytest

from vembody.actions import ActionType, action_from_dict, action_to_dict
from vembody.errors import ModelParseError


def test_action_round_trip() -> None:
    action = action_from_dict({"type": "click", "x": 0.25, "y": 1.0})
    assert action.type is ActionType.CLICK
    assert action_to_dict(action) == {"type": "click", "x": 0.25, "y": 1.0}


def test_unknown_action_field_is_rejected() -> None:
    with pytest.raises(ModelParseError):
        action_from_dict({"type": "click", "x": 0.2, "y": 0.3, "code": "bad"})
