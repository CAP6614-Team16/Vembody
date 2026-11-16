import json
import sys
from pathlib import Path

from vembody.evaluation.metrics import deduced_move
from vembody.minesweeper.deduce import cell_name, deductions
from vembody.models.http import HttpVisionModel
from vembody.models.prompts import SYSTEM_PROMPT
from vembody.parsing import parse_model_response

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import generate_sft_data  # noqa: E402


def test_deductions_find_safe_cells_and_mines():
    board = [["1", "#"], ["F", "#"]]
    assert {cell: kind for cell, (kind, _) in deductions(board).items()} == {
        (0, 1): "click",
        (1, 1): "click",
    }
    mines = deductions([["1", "#"], [".", "1"]])
    assert mines[(0, 1)][0] == "right_click"
    assert cell_name((2, 4)) == "C5"


def test_deduced_move_scores_only_moves_on_hidden_cells():
    board = [["1", "#"], [".", "1"]]

    def step(kind, cell):
        return {
            "action": {"type": kind},
            "execution": {"evidence": {"visible_before": board, "resolved_cell": cell}},
        }

    assert deduced_move(step("right_click", [0, 1])) is True
    assert deduced_move(step("click", [0, 1])) is False
    assert deduced_move(step("click", [0, 0])) is None


def test_examples_parse_and_match_the_inference_prompt(tmp_path):
    template = generate_sft_data.load_task(generate_sft_data.TEMPLATE)
    records = generate_sft_data.episode(123_456, template, set(), tmp_path)
    assert records
    model = HttpVisionModel.__new__(HttpVisionModel)
    for record in records:
        target = json.loads(record["response"])
        parse_model_response(record["response"])
        assert record["system"] == SYSTEM_PROMPT
        assert (tmp_path / record["image"]).exists()
        # Board copy uses the transcription format the metric reads
        rows = len(target["board"])
        assert all(len(row.split()) == len(target["board"][0].split()) for row in target["board"])
        assert rows >= 5
    # Prompt text equals what build_messages sends for the same task
    first = records[0]
    messages = model.build_messages(
        generate_sft_data.replace(template, instruction=json.loads(first["prompt"])["instruction"]),
        "x",
    )
    assert messages[0]["content"] == first["system"]
    if "previous_actions" not in first["prompt"]:
        assert messages[1]["content"][0]["text"] == first["prompt"]
