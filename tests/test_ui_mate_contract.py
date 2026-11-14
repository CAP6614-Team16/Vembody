import json
from zipfile import ZipFile

import pytest
from PIL import Image

from scripts.check_ui_mate_compatibility import archive_coordinate_audit, check_contract
from vembody.models.ui_mate import parse_ui_mate_action
from vembody.models.ui_mate_contract import coordinate_pixel, resize_screenshot


@pytest.mark.parametrize("size", [(336, 376), (208, 248), (1920, 1080)])
def test_native_coordinates_roundtrip_through_executor_exactly(size):
    for x, y in ((0, 0), (500, 622), (489, 360), (268, 750), (999, 999)):
        native = (
            "<tool_call><function=computer_use><parameter=action>left_click</parameter>"
            f"<parameter=coordinate>[{x},{y}]</parameter></function></tool_call>"
        )
        action = parse_ui_mate_action(native, size)
        actual = [round(action[key] * (length - 1)) for key, length in zip("xy", size, strict=True)]
        expected = [
            min(int(value * length / 999), length - 1)
            for value, length in zip((x, y), size, strict=True)
        ]
        assert actual == expected


@pytest.mark.parametrize("value,size", [(-1, 336), (1000, 336), (float("nan"), 336), (500, 1)])
def test_invalid_pixel_inputs_fail_closed(value, size):
    with pytest.raises(ValueError):
        coordinate_pixel(value, size)


@pytest.mark.parametrize(
    "original,expected",
    [
        ((336, 376), (320, 384)),
        ((208, 248), (192, 256)),
        ((1920, 1080), (1920, 1088)),
        ((4, 4), (64, 64)),
    ],
)
def test_client_resize_matches_pinned_factor32_contract(original, expected):
    image = Image.new("RGB", original, (123, 45, 67))
    prepared = resize_screenshot(image)
    assert prepared.size == expected
    assert prepared.mode == "RGB"
    assert prepared.getpixel((0, 0)) == (123, 45, 67)
    assert prepared.getpixel((prepared.width - 1, prepared.height - 1)) == (123, 45, 67)


@pytest.mark.parametrize("size", [(1, 100), (100, 1), (2, 500)])
def test_unsupported_screenshot_dimensions_are_rejected(size):
    with pytest.raises(ValueError):
        resize_screenshot(Image.new("RGB", size))


def test_contract_report_and_archived_rounding_audit_are_offline(tmp_path):
    native = (
        "<tool_call><function=computer_use><parameter=action>left_click</parameter>"
        "<parameter=coordinate>[500,622]</parameter></function></tool_call>"
    )
    record = {
        "record_type": "step",
        "step": {
            "model_metadata": {"native_response": native},
            "observation": {"capture_bounds": [0, 0, 336, 376]},
        },
    }
    path = tmp_path / "results.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("grounding/logs/task/trial_000.jsonl", json.dumps(record) + "\n")

    audit = archive_coordinate_audit(path)
    assert audit["clicks_checked"] == 1
    assert audit["maximum_per_axis_difference_pixels"] == 1
    assert check_contract()["status"] == "offline_contract_checks_passed"
