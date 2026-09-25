import pytest

from vembody.actions import action_from_dict
from vembody.errors import ActionValidationError
from vembody.validation import ActionPolicy, validate_action


def test_boundary_coordinates_are_allowed() -> None:
    policy = ActionPolicy(frozenset({"click"}))
    validate_action(action_from_dict({"type": "click", "x": 0.0, "y": 1.0}), policy)


def test_out_of_range_coordinates_are_rejected() -> None:
    policy = ActionPolicy(frozenset({"click"}))
    action = action_from_dict({"type": "click", "x": 1.01, "y": 0.5})
    with pytest.raises(ActionValidationError):
        validate_action(action, policy)


def test_disallowed_action_is_rejected() -> None:
    policy = ActionPolicy(frozenset({"finish"}))
    action = action_from_dict({"type": "click", "x": 0.5, "y": 0.5})
    with pytest.raises(ActionValidationError):
        validate_action(action, policy)
