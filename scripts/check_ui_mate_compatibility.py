"""Check the pinned UI-Mate contract offline, optionally auditing archived coordinate mappings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PIL import Image

from vembody.errors import ModelParseError
from vembody.models.ui_mate import PROMPT_VERSION, parse_ui_mate_action
from vembody.models.ui_mate_contract import UPSTREAM_SOURCE, coordinate_pixel, resize_screenshot


def archive_coordinate_audit(path: Path) -> dict:
    """Compare only decoded coordinates from old logs, without executing archived instructions."""
    differences = []
    with ZipFile(path) as archive:
        for name in archive.namelist():
            if "/logs/" not in name or not name.endswith(".jsonl"):
                continue
            for line in archive.read(name).decode("utf-8").splitlines():
                record = json.loads(line)
                if record.get("record_type") != "step":
                    continue
                step = record["step"]
                raw = step.get("model_metadata", {}).get("native_response")
                observation = step.get("observation") or {}
                bounds = observation.get("capture_bounds")
                if not raw or not bounds:
                    continue
                old = parse_ui_mate_action(raw)
                new = parse_ui_mate_action(raw, (bounds[2], bounds[3]))
                if old["type"] not in {"click", "right_click", "double_click"}:
                    continue
                before = [
                    round(old[key] * (size - 1)) for key, size in zip("xy", bounds[2:], strict=True)
                ]
                after = [
                    round(new[key] * (size - 1)) for key, size in zip("xy", bounds[2:], strict=True)
                ]
                differences.append(max(abs(a - b) for a, b in zip(before, after, strict=True)))

    return {
        "source_archive": path.name,
        "clicks_checked": len(differences),
        "clicks_with_rounding_change": sum(value != 0 for value in differences),
        "maximum_per_axis_difference_pixels": max(differences, default=0),
        "note": "This audits coordinate conversion only, not model decisions or image encoding.",
    }


def check_contract(results: Path | None = None) -> dict:
    """Verify pixel scaling, client resizing, and strict parser behavior without a model call."""
    checked = 0
    for width, height in ((336, 376), (208, 248), (1920, 1080)):
        for x, y in ((0, 0), (500, 250), (489, 360), (999, 999)):
            raw = (
                "<tool_call><function=computer_use><parameter=action>left_click</parameter>"
                f"<parameter=coordinate>[{x},{y}]</parameter></function></tool_call>"
            )
            action = parse_ui_mate_action(raw, (width, height))
            actual = (round(action["x"] * (width - 1)), round(action["y"] * (height - 1)))
            expected = (coordinate_pixel(x, width), coordinate_pixel(y, height))
            if actual != expected:
                raise RuntimeError("UI-Mate pixel round-trip failed")
            checked += 1

    encoded = resize_screenshot(Image.new("RGB", (336, 376)))
    if encoded.size != (320, 384):
        raise RuntimeError("UI-Mate screenshot resize differs from the pinned contract")
    for unsafe in ("pyautogui.click(10, 20)", raw + raw):
        try:
            parse_ui_mate_action(unsafe)
        except ModelParseError:
            continue
        raise RuntimeError("Unsafe response was accepted")

    result = {
        "status": "offline_contract_checks_passed",
        "upstream_source": UPSTREAM_SOURCE,
        "prompt_version": PROMPT_VERSION,
        "coordinate_roundtrip_cases": checked,
        "original_image_size": [336, 376],
        "encoded_image_size": list(encoded.size),
        "intentional_deviations": [
            "Restricted schema/prompt, not the verbatim general-desktop system prompt.",
            "One allowed call only; no generated code or implicit-success fallback.",
            "Far-edge coordinates clipped to the last image pixel.",
            "Current screenshot only; no upstream multi-turn image/reasoning history.",
            "Greedy decoding and smaller context/output budgets; not official benchmark parity.",
        ],
        "scope": "Offline adapter checks only; GPU preprocessing and accuracy require a real run.",
    }
    if results is not None:
        result["archive_coordinate_audit"] = archive_coordinate_audit(results)
    return result


def main() -> None:
    """Print a small audit report, optionally saving it beside local experiment artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(check_contract(args.results), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
