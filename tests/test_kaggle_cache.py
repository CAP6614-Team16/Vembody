import json
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from scripts import kaggle_cache as cache


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    """Supply tiny runtime files without executing native code or requiring CUDA."""
    binary = tmp_path / "build/bin/llama-server"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"test-server")
    (binary.parent / "libggml-cuda.so").write_bytes(b"test-cuda-backend")
    identity = {
        "system": "Linux",
        "machine": "x86_64",
        "libc": ["glibc", "2.39"],
        "cuda_toolkit": "CUDA V13.0.88",
        "cpu_flags": ["avx2"],
    }
    monkeypatch.setattr(cache, "environment_identity", lambda: identity)
    monkeypatch.setattr(cache, "check_binary", lambda *_: None)
    return binary, {"llama_revision": "60eeeb6082c1126bb8bc72902c83123cd056811b"}, identity


@pytest.mark.parametrize("unpack", [False, True])
def test_runtime_cache_roundtrip_checks_libraries_and_skips_build(deployment, tmp_path, unpack):
    binary, spec, _ = deployment
    inputs = tmp_path / "input"
    archive = cache.export_runtime(binary, spec, inputs / "vembody-runtime-t4.zip")
    if unpack:
        with ZipFile(archive) as bundle:
            bundle.extractall(inputs / "unpacked")

        archive.unlink()

    restored = cache.restore_runtime(tmp_path / "work", spec, inputs)
    assert restored.read_bytes() == b"test-server"
    assert (restored.parent / "libggml-cuda.so").read_bytes() == b"test-cuda-backend"
    assert cache.restore_runtime(tmp_path / "work", spec, inputs) == restored


@pytest.mark.parametrize("change", ["revision", "toolkit", "cpu", "hash", "path", "extra"])
def test_runtime_cache_rejects_bad_metadata_before_execution(deployment, tmp_path, change):
    binary, spec, _ = deployment
    initial = cache.export_runtime(binary, spec, tmp_path / "original.zip")
    with ZipFile(initial) as bundle:
        contents = {name: bundle.read(name) for name in bundle.namelist()}

    manifest = json.loads(contents[cache.MANIFEST])
    match change:
        case "revision":
            manifest["llama_revision"] = "wrong"
        case "toolkit":
            manifest["environment"]["cuda_toolkit"] = "different"
        case "cpu":
            manifest["environment"]["cpu_flags"].append("unavailable")
        case "hash":
            contents["bin/libggml-cuda.so"] = b"corrupted"
        case "path":
            manifest["files"]["bin/../../escape"] = "0" * 64
        case "extra":
            contents["unexpected.gguf"] = b"not-allowed"

    contents[cache.MANIFEST] = json.dumps(manifest).encode()
    inputs = tmp_path / "input"
    inputs.mkdir()
    with ZipFile(inputs / "vembody-runtime-t4.zip", "w") as bundle:
        for name, value in contents.items():
            bundle.writestr(name, value)

    with pytest.raises(RuntimeError):
        cache.restore_runtime(tmp_path / "work", spec, inputs)


def test_runtime_cache_detects_post_restore_corruption(deployment, tmp_path):
    binary, spec, _ = deployment
    inputs = tmp_path / "input"
    cache.export_runtime(binary, spec, inputs / "vembody-runtime-t4.zip")
    restored = cache.restore_runtime(tmp_path / "work", spec, inputs)
    restored.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="checksum"):
        cache.restore_runtime(tmp_path / "work", spec, inputs)


def test_no_cache_returns_none_but_multiple_caches_fail(deployment, tmp_path):
    binary, spec, _ = deployment
    inputs = tmp_path / "input"
    assert cache.restore_runtime(tmp_path / "work", spec, inputs) is None
    cache.export_runtime(binary, spec, inputs / "vembody-runtime-t4-a.zip")
    cache.export_runtime(binary, spec, inputs / "vembody-runtime-t4-b.zip")
    with pytest.raises(RuntimeError, match="exactly one"):
        cache.restore_runtime(tmp_path / "work", spec, inputs)


@pytest.mark.parametrize("failure", [None, "missing", "revision"])
def test_binary_check_rejects_missing_libraries_or_wrong_revision(tmp_path, monkeypatch, failure):
    calls = []

    def execute(command, **kwargs):
        """Return dependency/version diagnostics without launching a fixture binary."""
        calls.append((command, kwargs))
        output = "libc.so => /usr/lib/libc.so"
        if command[0] == "ldd" and failure == "missing":
            output = "libggml.so => not found"
        elif command[-1] == "--version":
            output = "wrong" if failure == "revision" else "commit 60eeeb6"
        return SimpleNamespace(stdout=output, stderr="")

    monkeypatch.setattr(cache.subprocess, "run", execute)
    binary = tmp_path / "bin/llama-server"
    spec = {"llama_revision": "60eeeb6082c1126bb8bc72902c83123cd056811b"}
    if failure:
        with pytest.raises(RuntimeError):
            cache.check_binary(binary, spec)
    else:
        cache.check_binary(binary, spec)
        assert len(calls) == 2
        assert calls[1][1]["env"]["LD_LIBRARY_PATH"].startswith(str(binary.parent))
