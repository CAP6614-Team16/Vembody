import pytest

from vembody.errors import ModelParseError
from vembody.parsing import parse_model_response


def test_parser_reads_one_action() -> None:
    parsed = parse_model_response('{"action":{"type":"finish"},"confidence":0.9}')
    assert parsed.action.type.value == "finish"
    assert parsed.confidence == 0.9


def test_parser_rejects_non_json() -> None:
    with pytest.raises(ModelParseError):
        parse_model_response("click(0.5, 0.5)")


def test_parser_rejects_unknown_top_level_field() -> None:
    with pytest.raises(ModelParseError):
        parse_model_response('{"action":{"type":"finish"},"execute":"yes"}')
