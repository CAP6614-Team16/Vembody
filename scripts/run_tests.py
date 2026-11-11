"""Run pytest with a fresh project-local temporary directory and no shared cache.

Windows test directories created in another execution context can be inaccessible.
Each invocation gets its own directory directly under the repository root. No
shared system temp/cache folders are deleted or have their permissions changed.
Directories from earlier runs are pruned first so they stop accumulating; pass
--keep-temp to retain them. The current run's directory always survives for
inspection. Pruning skips any directory this process cannot access and says how
many stayed. Additional arguments are forwarded to pytest, for example
tests/test_loop.py -v.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

RUN_TEMP = re.compile(r"\.pytest-tmp-[0-9a-f]{32}\Z")


def prune(root: Path) -> tuple[int, int]:
    """Remove this script's own earlier run directories, best effort.

    A directory created under a different execution context can itself be
    inaccessible, so report what stayed instead of implying it was removed.
    """
    removed = stuck = 0
    for path in sorted(root.glob(".pytest-tmp-*")):
        if path.parent != root or not path.is_dir() or not RUN_TEMP.match(path.name):
            continue
        shutil.rmtree(path, ignore_errors=True)
        if path.exists():
            stuck += 1
        else:
            removed += 1
    return removed, stuck


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    forwarded = [arg for arg in sys.argv[1:] if arg != "--keep-temp"]
    if "--keep-temp" not in sys.argv[1:]:
        removed, stuck = prune(root)
        if removed or stuck:
            print(f"Pruned {removed} earlier test directories, {stuck} inaccessible", flush=True)
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
        *forwarded,
    ]
    return subprocess.run(command, cwd=root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
