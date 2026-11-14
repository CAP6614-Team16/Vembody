"""Build action-proposing models by name."""

from __future__ import annotations

from ..tasks import TaskSpec
from .base import VisionModel
from .baselines import RandomCellModel, RandomClickModel, SolverModel
from .http import HttpVisionModel
from .mock import MockModel
from .prompts import PROMPT_VERSION
from .settings import EndpointConfig
from .ui_mate import (
    BRIEF_PROMPT_VERSION,
    LEGACY_PROMPT_VERSION,
    BriefUiMateModel,
    LegacyUiMateModel,
    UiMateModel,
)
from .ui_mate import PROMPT_VERSION as UI_MATE_PROMPT_VERSION

REMOTE_MODEL_NAMES = ("http", "ui-mate", "ui-mate-legacy", "ui-mate-brief")
MODEL_NAMES = ("mock", "random", "random-cell", "solver", *REMOTE_MODEL_NAMES)
MODEL_PROMPT_VERSIONS = {
    "http": PROMPT_VERSION,
    "ui-mate": UI_MATE_PROMPT_VERSION,
    "ui-mate-legacy": LEGACY_PROMPT_VERSION,
    "ui-mate-brief": BRIEF_PROMPT_VERSION,
}


def build_model(
    name: str,
    task: TaskSpec,
    sandbox: object | None = None,
    seed: int | None = None,
    model_config: EndpointConfig | None = None,
) -> VisionModel:
    """Create a fresh model for one episode; ``seed`` drives any random choices."""
    match name:
        case "mock":
            return MockModel.from_task(task)

        case "random":
            return RandomClickModel(seed)

        case "random-cell":
            return RandomCellModel(task, seed)

        case "solver":
            if sandbox is None or not hasattr(sandbox, "screen"):
                raise ValueError("The solver baseline needs a Minesweeper sandbox")

            return SolverModel(task, sandbox.screen, seed)

        case "http" | "ui-mate" | "ui-mate-legacy" | "ui-mate-brief":
            if model_config is None:
                raise ValueError("The HTTP model requires endpoint configuration")

            adapters = {
                "http": HttpVisionModel,
                "ui-mate": UiMateModel,
                "ui-mate-legacy": LegacyUiMateModel,
                "ui-mate-brief": BriefUiMateModel,
            }
            return adapters[name](model_config)

        case _:
            raise ValueError(f"Unknown model: {name}")
