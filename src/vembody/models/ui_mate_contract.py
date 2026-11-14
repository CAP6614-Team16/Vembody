"""Independent, safety-restricted implementation of UI-Mate's published I/O contract."""

from __future__ import annotations

import json
import math

from PIL import Image

from ..tasks import TaskSpec

UPSTREAM_REVISION = "1cb9e1e44ce856e23b593992b02efbd489943fcb"
UPSTREAM_SOURCE = (
    f"https://github.com/Tencent/UI-Mate/blob/{UPSTREAM_REVISION}/agents/ui_mate_agent.py"
)
COORDINATE_GRID_MAX = 999


def coordinate_pixel(value: float, size: int) -> int:
    """Match upstream truncation, clipping only the far edge to keep input inside the image."""
    if size < 2 or not math.isfinite(value) or not 0 <= value <= COORDINATE_GRID_MAX:
        raise ValueError("Invalid UI-Mate coordinate or image dimension")

    return min(int(value * size / COORDINATE_GRID_MAX), size - 1)


def resize_screenshot(image: Image.Image) -> Image.Image:
    """Match upstream factor-32 sizing and pixel limits without adding borders or cropping."""
    width, height = image.size
    if min(width, height) < 2 or max(width, height) / min(width, height) > 200:
        raise ValueError("Unsupported UI-Mate screenshot dimensions")

    if max(width, height) > 8192:
        scale = max(width, height) / 8192
        width, height = int(width / scale), int(height / scale)

    factor = 32
    minimum, maximum = 56 * 56, 16 * 16 * 4 * 12800
    aligned = [round(length / factor) * factor for length in (width, height)]
    if math.prod(aligned) > maximum:
        scale = math.sqrt(width * height / maximum)
        aligned = [math.floor(length / scale / factor) * factor for length in (width, height)]
    elif math.prod(aligned) < minimum:
        scale = math.sqrt(minimum / (width * height))
        aligned = [math.ceil(length * scale / factor) * factor for length in (width, height)]

    return image.convert("RGB").resize(tuple(aligned), resample=Image.Resampling.BICUBIC)


def system_prompt(task: TaskSpec, *, brief: bool = False) -> str:
    """Use upstream's schema/XML organization while advertising only task-permitted controls."""
    allowed = {action.value for action in task.allowed_actions}
    names = {"click": "left_click", "right_click": "right_click", "double_click": "double_click"}
    actions = [native for action, native in names.items() if action in allowed]
    statuses = [
        status
        for action, status in (("finish", "success"), ("fail", "failure"))
        if action in allowed
    ]
    if statuses:
        actions.append("finished")

    definition = {
        "type": "function",
        "function": {
            "name": "computer_use",
            "description": (
                "Interact only with the visible Minesweeper screenshot. Locate cell centers "
                "visually on a 1000-unit image grid, with each axis ranging from 0 to 999. "
                "No terminal, code execution, hidden game state, or other applications exist."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": actions},
                    "coordinate": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "[x,y] in the 0-999 image grid; required for a click.",
                    },
                    "status": {"type": "string", "enum": statuses},
                },
                "required": ["action"],
            },
        },
    }
    response_instruction = (
        "Choose only the next action, not a complete game plan. "
        "Give at most one short sentence summarizing the decision in <action>...</action>; "
        "do not include extended analysis or a <think> block. "
        "Immediately follow it with exactly one complete XML tool call, with no text afterwards:\n"
        if brief
        else "Return optional reasoning before <action>a short action description</action>, "
        "then exactly one XML tool call, with no text afterwards:\n"
    )
    return (
        "You are a GUI agent.\n\n# Tools\n<tools>\n"
        + json.dumps(definition)
        + "\n</tools>\n\n# Response format\n"
        + response_instruction
        + "<tool_call>\n<function=computer_use>\n<parameter=action>left_click</parameter>\n"
        "<parameter=coordinate>[500,500]</parameter>\n</function>\n</tool_call>\n"
        "Use only permitted actions. For finished, supply status instead of coordinate. "
        "Task success is independently checked; a completion claim does not establish success. "
        "Never output or execute Python, shell commands, macros, or multiple calls."
    )


def instruction_prompt(task: TaskSpec) -> str:
    """Follow upstream's first-turn instruction layout without exposing private task fields."""
    return (
        "Choose the next action from the screenshot and instruction.\n\n"
        f"Instruction: {task.instruction}\n\nPrevious actions:\nNone"
    )
