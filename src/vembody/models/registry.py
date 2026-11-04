"""Build action-proposing models by name."""

from __future__ import annotations

from ..tasks import TaskSpec
from .base import VisionModel
from .baselines import RandomCellModel, RandomClickModel, SolverModel
from .mock import MockModel

MODEL_NAMES = ("mock", "random", "random-cell", "solver")


def build_model(
    name: str,
    task: TaskSpec,
    sandbox: object | None = None,
    seed: int | None = None,
) -> VisionModel:
    """Create a fresh model for one episode; ``seed`` drives any random choices."""
    if name == "mock":
        return MockModel.from_task(task)
    if name == "random":
        return RandomClickModel(seed)
    if name == "random-cell":
        return RandomCellModel(task, seed)
    if name == "solver":
        if sandbox is None or not hasattr(sandbox, "screen"):
            raise ValueError("The solver baseline needs a Minesweeper sandbox")
        return SolverModel(task, sandbox.screen, seed)
    raise ValueError(f"Unknown model: {name}")
