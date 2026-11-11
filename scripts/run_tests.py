"""Run pytest with a fresh project-local temporary directory and no shared cache.

Windows test directories created in another execution context can be inaccessible.
Each invocation gets its own directory directly under the repository root. No
shared system temp/cache folders are deleted or have their permissions changed.
Additional arguments are forwarded to pytest, for example tests/test_loop.py -v.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from uuid import uuid4


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    run_temp = root / f".pytest-tmp-{uuid4().hex}"
    if run_temp.parent != root or run_temp.exists():
        raise RuntimeError("Expected a fresh test directory inside the repository")
    print(f"Test temporary directory: {run_temp}", flush=True)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "--basetemp",
        str(run_temp),
        *sys.argv[1:],
    ]
    return subprocess.run(command, cwd=root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
