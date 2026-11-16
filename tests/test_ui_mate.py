import json

import pytest

from vembody.errors import ModelParseError
from vembody.models.base import ModelResponse
from vembody.models.http import HttpVisionModel
from vembody.models.registry import build_model
from vembody.models.settings import EndpointConfig
from vembody.models.ui_mate import (
    BriefUiMateModel,
    LegacyUiMateModel,
    UiMateModel,
    parse_ui_mate_action,
)
from vembody.observations import Observation
from vembody.tasks import load_task


def native_call(action="left_click", coordinate="[500, 250]", extra=""):
    """Construct a native tool response without relying on any model or GPU."""
    return (
        "<tool_call><function=computer_use><parameter=action>"
        + action
        + "</parameter>"
        + ("<parameter=coordinate>" + coordinate + "</parameter>" if coordinate else "")
        + extra
        + "</function></tool_call>"
    )


@pytest.mark.parametrize(
    "native,expected",
    [("left_click", "click"), ("right_click", "right_click"), ("double_click", "double_click")],
)
def test_native_clicks_use_999_coordinate_grid(native, expected):
    action = parse_ui_mate_action(native_call(native))
    assert action == {"type": expected, "x": 500 / 999, "y": 250 / 999}


@pytest.mark.parametrize("status,expected", [("success", "finish"), ("failure", "fail")])
def test_native_finished_requires_status(status, expected):
    assert parse_ui_mate_action(
        native_call("finished", "", f"<parameter=status>{status}</parameter>")
    ) == {"type": expected}


@pytest.mark.parametrize(
    "value",
    [
        "[-1, 0]",
        "[1000, 0]",
        "[true, 0]",
        "[NaN, 0]",
        "[Infinity, 0]",
        "[0]",
        '["500", 0]',
        "{}",
        "__import__('os').system('echo unsafe')",
    ],
)
def test_invalid_coordinates_are_rejected(value):
    with pytest.raises(ModelParseError):
        parse_ui_mate_action(native_call(coordinate=value))


@pytest.mark.parametrize(
    "text",
    [
        "pyautogui.click(500, 250)",
        native_call() + native_call(),
        native_call().replace("computer_use", "shell"),
        native_call(extra="<parameter=action>right_click</parameter>"),
        native_call(extra="<parameter=script>print(1)</parameter>"),
        native_call("type", ""),
        native_call("finished", ""),
        native_call("finished", "", "<parameter=status>maybe</parameter>"),
        native_call().replace("</function>", ""),
        "<think>" + native_call(),
        native_call() + "do something else",
    ],
)
def test_unsupported_ambiguous_and_code_responses_are_rejected(text):
    with pytest.raises(ModelParseError):
        parse_ui_mate_action(text)


def test_reasoning_examples_are_not_executed():
    response = "<think>I could try " + native_call("right_click") + "</think>"
    response += "<action>Reveal the target cell.</action>" + native_call()
    assert parse_ui_mate_action(response)["type"] == "click"


def test_prompt_contains_no_benchmark_answers():
    task = load_task("tasks/benchmark/development/t1-01.yaml")
    model = UiMateModel(EndpointConfig("http://localhost/v1", "test", "TEST_KEY"))
    messages = model.build_messages(task, "data:image/png;base64,TEST_IMAGE")
    prompt = json.dumps(messages)
    assert "expected_state_hash" not in prompt and "mock_actions" not in prompt
    assert "1000-unit" in prompt
    text = messages[1]["content"][1]["text"]
    assert "Instruction: " + task.instruction in text
    assert "Previous actions:\nNone" in text
    system = messages[0]["content"][0]["text"]
    schema = json.loads(system.split("<tools>\n", 1)[1].split("\n</tools>", 1)[0])
    properties = schema["function"]["parameters"]["properties"]
    assert properties["action"]["enum"] == ["left_click", "right_click", "finished"]
    assert properties["status"]["enum"] == ["success"]


def test_legacy_prompt_is_still_available_for_controlled_comparisons():
    task = load_task("tasks/benchmark/development/t1-01.yaml")
    model = LegacyUiMateModel(EndpointConfig("http://localhost/v1", "test", "TEST_KEY"))
    messages = model.build_messages(task, "data:image/png;base64,TEST_IMAGE")
    assert model.prompt_version == "ui-mate-minesweeper-v1"
    text = json.loads(messages[1]["content"][1]["text"])
    assert text["permitted_clicks"] == ["left_click", "right_click"]


def test_adapter_refuses_other_environments():
    model = UiMateModel(EndpointConfig("http://localhost/v1", "test", "TEST_KEY"))
    with pytest.raises(ValueError, match="Minesweeper"):
        model.build_messages(load_task("tasks/example.yaml"), "data:image/png;base64,TEST_IMAGE")


def test_brief_prompt_changes_only_response_instruction_not_grounding_or_private_inputs():
    task = load_task("tasks/benchmark/development/t4-01.yaml")
    config = EndpointConfig("http://localhost/v1", "test", "TEST_KEY")
    model = build_model("ui-mate-brief", task, model_config=config)
    assert isinstance(model, BriefUiMateModel)
    brief = model.build_messages(task, "data:image/png;base64,TEST_IMAGE")
    original = UiMateModel(config).build_messages(task, "data:image/png;base64,TEST_IMAGE")
    assert brief[1] == original[1]
    before = original[0]["content"][0]["text"]
    after = brief[0]["content"][0]["text"]
    assert before.split("# Response format")[0] == after.split("# Response format")[0]
    assert "one short sentence" in after and "complete XML tool call" in after
    assert "optional reasoning" in before and "optional reasoning" not in after
    assert "expected_state_hash" not in json.dumps(brief)
    assert "mock_actions" not in json.dumps(brief)
    assert brief[1]["content"][1]["text"].endswith("Previous actions:\nNone")
    assert model.prompt_version == "ui-mate-minesweeper-v3-brief"


@pytest.mark.parametrize("raw", ["Incomplete decision", native_call()])
def test_truncated_response_is_rejected_even_if_a_complete_call_precedes_cutoff(monkeypatch, raw):
    monkeypatch.setattr(
        HttpVisionModel,
        "generate_action",
        lambda *_: ModelResponse(raw, 0.01, {"finish_reason": "length", "image_size": [208, 248]}),
    )
    model = BriefUiMateModel(EndpointConfig("http://localhost/v1", "test", "TEST_KEY"))
    response = model.generate_action(
        load_task("tasks/benchmark/development/t4-01.yaml"), Observation("e", 0, "t"), []
    )
    assert "truncated" in response.metadata["adapter_error"]
    assert response.metadata["native_response"] == raw
    assert response.metadata["finish_reason"] == "length"
    assert "action" not in json.loads(response.raw_text)
