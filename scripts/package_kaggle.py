"""Bundle current project code for Kaggle without credentials, weights, or run artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {"pyproject.toml", "README.md", "LICENSE"}
FOLDERS = {"src", "scripts", "tests", "configs", "requirements", "tasks", "notebooks", "docs"}
EXTENSIONS = {".py", ".json", ".yaml", ".yml", ".txt", ".md", ".toml", ".ipynb"}


def bundle_files(root: Path) -> list[Path]:
    """Include Git-visible source files through an explicit path/extension allowlist."""
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    paths = []
    for name in sorted(set(result.stdout.decode("utf-8").split("\0")) - {""}):
        relative = Path(name)
        allowed = name in ROOT_FILES or (
            relative.parts[0] in FOLDERS
            and relative.suffix in EXTENSIONS
            and not any(part.startswith(".") for part in relative.parts)
        )
        # Only the small, committed reference assets, never other downloaded datasets.
        allowed = allowed or (
            name.startswith("data/benchmark/pilot/") and relative.suffix in {".png", ".md", ".json"}
        )
        path = root / relative
        if allowed and path.is_file() and not path.is_symlink():
            paths.append(path)

    return paths


def package(root: Path, destination: Path) -> Path:
    """Write a portable snapshot with per-file checksums, including uncommitted source changes."""
    paths = bundle_files(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    hashes = {}
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for path in paths:
            name = path.relative_to(root).as_posix()
            contents = path.read_bytes()
            archive.writestr("vembody/" + name, contents)
            hashes[name] = hashlib.sha256(contents).hexdigest()

        archive.writestr(
            "vembody/bundle-manifest.json",
            json.dumps(
                {
                    "base_git_commit": commit,
                    "snapshot_includes_uncommitted_code": True,
                    "files": hashes,
                },
                indent=2,
            )
            + "\n",
        )

    return destination


if __name__ == "__main__":
    print(package(ROOT, ROOT / "artifacts/vembody-kaggle.zip"))
