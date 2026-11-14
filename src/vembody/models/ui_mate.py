"""Safe UI-Mate XML-tool adaptation for the screenshot-only Minesweeper baseline."""

from __future__ import annotations

import json
import math
import re

from PIL import Image

from ..errors import ModelParseError
from ..observations import Observation
from ..results import StepResult
from ..tasks import TaskSpec
from .base import ModelResponse
from .http import HttpVisionModel
from .ui_mate_contract import (
    UPSTREAM_REVISION,
    coordinate_pixel,
    instruction_prompt,
    resize_screenshot,
    system_prompt,
)

PROMPT_VERSION = "ui-mate-minesweeper-v2"
BRIEF_PROMPT_VERSION = "ui-mate-minesweeper-v3-brief"
LEGACY_PROMPT_VERSION = "ui-mate-minesweeper-v1"
# Original restricted prompt; native format/coordinate contract documented by Tencent:
# https://github.com/Tencent/UI-Mate/blob/1cb9e1e44ce856e23b593992b02efbd489943fcb/agents/ui_mate_agent.py
SYSTEM_PROMPT = """You are a GUI agent controlling only a Minesweeper screenshot.
Use visible pixels, not code or external tools. Coordinates use a 1000-unit image grid:
top-left [0, 0], bottom-right [999, 999]. Click the center of a cell.
Choose exactly one action per response. Do not send multiple calls or Python code.
Available function: computer_use(action, coordinate?, status?).
Actions: left_click, right_click, double_click (coordinate=[x,y]);
finished (status=success or failure). Only use actions permitted by the user's task.
Completion requests are independently checked; never assume they establish success.
Return a brief <action>description</action>, followed by exactly one native tool call:
<tool_call>
<function=computer_use>
<parameter=action>left_click</parameter>
<parameter=coordinate>[500, 500]</parameter>
</function>
</tool_call>
Reasoning, if present, must precede the action and tool call, never follow them.
"""
CLICK_TYPES = {"left_click": "click", "right_click": "right_click", "double_click": "double_click"}


def parse_ui_mate_action(text: str, image_size: tuple[int, int] | None = None) -> dict:
    """Translate exactly one native call into an allowlisted action without executing code."""
    if (
        text.count("<think>") > 1
        or text.count("</think>") > 1
        or ("<think>" in text and "</think>" not in text)
    ):
        raise ModelParseError("UI-Mate reasoning block is incomplete or repeated")

    # Reasoning can mention hypothetical calls; only parse the final action section.
    visible = text.rsplit("</think>", 1)[-1]
    calls = re.findall(r"<tool_call>(.*?)</tool_call>", visible, re.DOTALL)
    if len(calls) != 1 or visible.count("<tool_call>") != 1 or visible.count("</tool_call>") != 1:
        raise ModelParseError("UI-Mate must return exactly one complete tool call")

    if visible.split("</tool_call>", 1)[1].strip():
        raise ModelParseError("Unexpected output after UI-Mate tool call")

    call = calls[0].strip()
    function = re.fullmatch(r"<function=computer_use>\s*(.*?)\s*</function>", call, re.DOTALL)
    if function is None:
        raise ModelParseError("UI-Mate may only call computer_use")

    params = {}
    remaining = function[1]
    pattern = re.compile(r"\s*<parameter=([a-z_]+)>(.*?)</parameter>", re.DOTALL)
    while remaining.strip():
        match = pattern.match(remaining)
        if match is None or match[1] in params:
            raise ModelParseError("Malformed or repeated UI-Mate parameter")

        params[match[1]] = match[2].strip()
        remaining = remaining[match.end() :]

    kind = params.get("action")
    if kind in CLICK_TYPES:
        if set(params) != {"action", "coordinate"}:
            raise ModelParseError("A UI-Mate click requires only action and coordinate")

        try:
            coordinate = json.loads(params["coordinate"])
        except ValueError as exc:
            raise ModelParseError("UI-Mate coordinates must be a JSON pair") from exc

        if (
            not isinstance(coordinate, list)
            or len(coordinate) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 999
                for value in coordinate
            )
        ):
            raise ModelParseError("UI-Mate coordinates must be finite numbers from 0 to 999")

        if image_size is not None:
            x, y = [
                coordinate_pixel(value, size) / (size - 1)
                for value, size in zip(coordinate, image_size, strict=True)
            ]
        else:
            x, y = coordinate[0] / 999, coordinate[1] / 999

        return {"type": CLICK_TYPES[kind], "x": x, "y": y}

    elif kind == "finished":
        if set(params) != {"action", "status"} or params["status"] not in {"success", "failure"}:
            raise ModelParseError("UI-Mate finished requires an explicit success/failure status")

        return {"type": "finish" if params["status"] == "success" else "fail"}

    raise ModelParseError("Unsupported UI-Mate action for the Minesweeper baseline")


class LegacyUiMateModel(HttpVisionModel):
    """Preserve the v1 pilot adapter for explicit before/after comparisons."""

    prompt_version = LEGACY_PROMPT_VERSION
    upstream_coordinate_mapping = False

    def build_messages(self, task: TaskSpec, image_url: str) -> list[dict]:
        """Send a restricted native prompt and current screenshot without private state/history."""
        if task.environment != "minesweeper":
            raise ValueError("The UI-Mate adapter currently supports Minesweeper only")

        permitted = [
            native
            for native, action in CLICK_TYPES.items()
            if action in {item.value for item in task.allowed_actions}
        ]
        statuses = [
            status
            for action, status in (("finish", "success"), ("fail", "failure"))
            if action in {item.value for item in task.allowed_actions}
        ]
        text = json.dumps(
            {
                "instruction": task.instruction,
                "permitted_clicks": permitted,
                "permitted_finished_statuses": statuses,
            }
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": text},
                ],
            },
        ]

    def generate_action(
        self, task: TaskSpec, observation: Observation, history: list[StepResult]
    ) -> ModelResponse:
        """Normalize a native action while retaining original output and usage on parse failures."""
        response = super().generate_action(task, observation, history)
        metadata = {**response.metadata, "native_response": response.raw_text}
        try:
            if metadata.get("finish_reason") == "length":
                raise ModelParseError("UI-Mate output was truncated at the token/context limit")

            size = tuple(metadata["image_size"]) if self.upstream_coordinate_mapping else None
            raw = json.dumps({"action": parse_ui_mate_action(response.raw_text, size)})
        except ModelParseError as exc:
            metadata["adapter_error"] = str(exc)
            # Preserve one failed model step for the loop's existing strict JSON parser.
            raw = json.dumps({"ui_mate_parse_error": str(exc)})

        return ModelResponse(raw, response.elapsed_seconds, metadata)


class UiMateModel(LegacyUiMateModel):
    """Align sizing, first-turn layout, and pixel mapping with the pinned upstream contract."""

    prompt_version = PROMPT_VERSION
    upstream_coordinate_mapping = True
    brief_response = False

    def prepare_image(self, image: Image.Image) -> Image.Image:
        """Apply upstream's client-side factor-32 resize before server-side image processing."""
        return resize_screenshot(image)

    def build_messages(self, task: TaskSpec, image_url: str) -> list[dict]:
        """Send a restricted schema and current screenshot, without benchmark state/history."""
        if task.environment != "minesweeper":
            raise ValueError("The UI-Mate adapter currently supports Minesweeper only")

        return [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt(task, brief=self.brief_response)}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": instruction_prompt(task)},
                ],
            },
        ]

    def generate_action(
        self,
        task: TaskSpec,
        observation: Observation,
        history: list[StepResult],
    ) -> ModelResponse:
        """Record the contract revision and transforms so comparisons remain reproducible."""
        response = super().generate_action(task, observation, history)
        response.metadata.update(
            upstream_contract_revision=UPSTREAM_REVISION,
            image_transform="factor32-no-padding",
            coordinate_transform="upstream-truncate-clamp-then-normalize",
            history_policy="current-screenshot-only",
        )
        return response


class BriefUiMateModel(UiMateModel):
    """Request one concise decision while preserving v2 image, coordinate, and safety behavior."""

    prompt_version = BRIEF_PROMPT_VERSION
    brief_response = True
