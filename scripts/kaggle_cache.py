"""Export and validate a private, weights-free Kaggle llama.cpp runtime cache."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile

MANIFEST = "runtime-manifest.json"


def digest_file(path: Path) -> str:
    """Hash runtime libraries incrementally without reading the whole cache into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def environment_identity() -> dict:
    """Record conservative compatibility requirements, including native CPU instructions."""
    cpu_info = Path("/proc/cpuinfo")
    flags = []
    if cpu_info.is_file():
        for line in cpu_info.read_text().splitlines():
            if line.startswith("flags") and ":" in line:
                flags = sorted(line.partition(":")[2].split())
                break

    toolkit = subprocess.check_output(["nvcc", "--version"], text=True, timeout=15).strip()
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "libc": list(platform.libc_ver()),
        "cuda_toolkit": toolkit,
        "cpu_flags": flags,
    }


def check_binary(binary: Path, spec: dict) -> None:
    """Check shared-library resolution and the expected revision before reusing a server."""
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = (
        str(binary.parent) + os.pathsep + environment.get("LD_LIBRARY_PATH", "")
    )
    dependencies = subprocess.run(
        ["ldd", str(binary)],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
        env=environment,
    )
    if "not found" in dependencies.stdout + dependencies.stderr:
        raise RuntimeError("Cached runtime has missing shared libraries; detach it and rebuild")

    version = subprocess.run(
        [str(binary), "--version"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
        env=environment,
    )
    if spec["llama_revision"][:7] not in version.stdout + version.stderr:
        raise RuntimeError("Cached server revision differs; detach it and rebuild")


def export_runtime(binary: Path, spec: dict, destination: Path) -> Path:
    """Save the complete build/bin directory and hashes, but never weights or driver files."""
    check_binary(binary, spec)
    root = binary.parent.resolve()
    files = {}
    for path in sorted(root.iterdir()):
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise RuntimeError("Runtime library points outside build/bin")
        if resolved.is_file():
            files["bin/" + path.name] = resolved

    if "bin/llama-server" not in files:
        raise RuntimeError("Cannot export a runtime without llama-server")

    manifest = {
        "schema": 1,
        "llama_revision": spec["llama_revision"],
        "cuda_architecture": 75,
        "environment": environment_identity(),
        "files": {name: digest_file(path) for name, path in files.items()},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, name)

        archive.writestr(MANIFEST, json.dumps(manifest, indent=2) + "\n")

    print(f"Runtime cache exported: {destination}", flush=True)
    return destination


def validate_manifest(manifest: dict, spec: dict) -> None:
    """Reject incompatible caches and unsafe paths before extracting or executing anything."""
    if (
        manifest.get("schema") != 1
        or manifest.get("llama_revision") != spec["llama_revision"]
        or manifest.get("cuda_architecture") != 75
    ):
        raise RuntimeError("Runtime cache settings differ; detach the cache and rebuild")

    current = environment_identity()
    previous = manifest["environment"]
    if any(current[key] != previous[key] for key in ("system", "machine", "libc", "cuda_toolkit")):
        raise RuntimeError("Runtime cache environment differs; detach the cache and rebuild")
    if not set(previous["cpu_flags"]) <= set(current["cpu_flags"]):
        raise RuntimeError("Runtime cache requires unavailable CPU instructions; rebuild it")

    files = manifest["files"]
    if not files or "bin/llama-server" not in files:
        raise RuntimeError("Runtime cache is missing llama-server")
    for name, digest in files.items():
        path = PurePosixPath(name)
        if (
            path.parts != ("bin", path.name)
            or path.name in {".", ".."}
            or "\\" in name
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise RuntimeError("Unsafe runtime cache manifest")


def restore_runtime(work: Path, spec: dict, inputs: Path = Path("/kaggle/input")) -> Path | None:
    """Restore one attached ZIP or Kaggle-unpacked runtime, validating every declared file."""
    archives = list(inputs.rglob("vembody-runtime-t4*.zip"))
    manifests = list(inputs.rglob(MANIFEST))
    if not archives and not manifests:
        return None
    if len(archives) + len(manifests) != 1:
        raise RuntimeError("Attach exactly one runtime cache, or detach all caches to rebuild")

    archive = ZipFile(archives[0]) if archives else None
    try:
        content = archive.read(MANIFEST) if archive else manifests[0].read_bytes()
        manifest = json.loads(content)
        validate_manifest(manifest, spec)
        if archive and (
            len(archive.namelist()) != len(manifest["files"]) + 1
            or set(archive.namelist()) != {*manifest["files"], MANIFEST}
            or sum(item.file_size for item in archive.infolist()) > 2 * 1024**3
        ):
            raise RuntimeError("Unexpected runtime cache contents")

        destination = work / "cached-runtime" / hashlib.sha256(content).hexdigest()[:16]
        for name, expected in manifest["files"].items():
            target = destination / name
            if not target.resolve().is_relative_to(destination.resolve()):
                raise RuntimeError("Unsafe runtime cache destination")
            if not target.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                if archive:
                    with archive.open(name) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output)
                else:
                    source = manifests[0].parent / name
                    if not source.resolve().is_relative_to(manifests[0].parent.resolve()):
                        raise RuntimeError("Unsafe runtime cache source")
                    shutil.copyfile(source, target)

            if digest_file(target) != expected:
                raise RuntimeError("Runtime cache checksum mismatch: " + name)

        binary = destination / "bin/llama-server"
        binary.chmod(0o755)
        check_binary(binary, spec)
        print("Reusing attached runtime cache; CUDA compilation skipped", flush=True)
        return binary
    finally:
        if archive:
            archive.close()
