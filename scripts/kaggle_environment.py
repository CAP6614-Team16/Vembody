"""Isolate Vembody dependencies from the Kaggle notebook kernel's shared packages."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def create_environment(repo: Path, directory: Path) -> Path:
    """Create a private environment, install pinned requirements, and check its dependencies."""
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(directory)], check=True)
    config = (directory / "pyvenv.cfg").read_text(encoding="utf-8").lower()
    if "include-system-site-packages = false" not in config:
        raise RuntimeError("Expected an isolated virtual environment, not Kaggle's shared packages")

    python = directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    # Host pip targets only this environment, avoiding Kaggle's failing ensurepip bootstrap.
    pip = [sys.executable, "-I", "-m", "pip", "--python", str(python)]
    subprocess.run(
        [
            *pip,
            "install",
            "-r",
            str(repo / "requirements/kaggle-runtime.txt"),
            "-e",
            str(repo),
        ],
        check=True,
    )
    subprocess.run([*pip, "check"], check=True)
    return python
