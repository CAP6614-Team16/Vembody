import ast
import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from scripts import kaggle_environment as environment
from scripts import kaggle_run as runner
from scripts import kaggle_ui_mate as kaggle
from scripts import package_kaggle as packaging


@pytest.fixture
def runtime(tmp_path):
    """Return paths and checked-in deployment settings without loading any weights."""
    spec = json.loads(Path("configs/kaggle-ui-mate.json").read_text())
    return kaggle.Runtime(
        tmp_path / "llama-server",
        tmp_path / "model.gguf",
        tmp_path / "projector.gguf",
        spec,
        "Tesla T4 x2",
    )


def test_server_command_is_local_pinned_and_not_a_public_endpoint(runtime):
    command = kaggle.server_command(runtime, 8000)
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--tensor-split") + 1] == "1,1"
    assert command[command.index("--ctx-size") + 1] == "8192"
    assert "--skip-chat-parsing" in command and "--mmproj" in command
    assert "--api-key" not in command
    assert command[command.index("--image-min-tokens") + 1] == "1024"
    assert command[command.index("--image-max-tokens") + 1] == "1024"


def test_prepare_checks_hashes_before_loading(tmp_path, monkeypatch):
    paths = {}
    files = {}
    for role in ("model", "projector"):
        path = tmp_path / (role + ".gguf")
        path.write_bytes(role.encode())
        paths[role + ".gguf"] = path
        files[role] = {"name": path.name, "sha256": kaggle.file_sha256(path)}

    spec = {"repo_id": "test/model", "revision": "immutable", "files": files}
    config = tmp_path / "config.json"
    config.write_text(json.dumps(spec))
    calls = []

    def download(**kwargs):
        """Return small fixture files rather than downloading model weights."""
        calls.append(kwargs)
        return str(paths[kwargs["filename"]])

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    monkeypatch.setattr(kaggle, "gpu_info", lambda: "T4 x2")
    monkeypatch.setattr(kaggle, "build_server", lambda *_: tmp_path / "server")
    actual = kaggle.prepare_runtime(tmp_path / "work", config)
    assert actual.model == paths["model.gguf"]
    assert all(call["revision"] == "immutable" and call["token"] is False for call in calls)
    paths["projector.gguf"].write_bytes(b"corrupted")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        kaggle.prepare_runtime(tmp_path / "work", config)


@pytest.mark.parametrize(
    "names,valid",
    [
        ("Tesla T4, 15360 MiB\nTesla T4, 15360 MiB", True),
        ("Tesla P100, 16000 MiB", False),
        ("", False),
    ],
)
def test_gpu_preflight(monkeypatch, names, valid):
    monkeypatch.setattr(kaggle.subprocess, "run", lambda *_, **__: SimpleNamespace(stdout=names))
    if valid:
        assert kaggle.gpu_info() == names
    else:
        with pytest.raises(RuntimeError, match="T4 x2"):
            kaggle.gpu_info()


@pytest.mark.parametrize("fail", [False, True])
def test_server_context_stops_own_process_and_restores_environment(
    runtime, tmp_path, monkeypatch, fail
):
    captured = {}

    class Process:
        stopped = False

        def __init__(self, command, **kwargs):
            """Capture only this fixture process's launch configuration."""
            captured.update(command=command, **kwargs)
            captured["process"] = self

        def poll(self):
            """Return a terminal status only after this fixture is terminated."""
            return 0 if self.stopped else None

        def terminate(self):
            """Mark this fixture stopped instead of signaling a real process."""
            self.stopped = True

        def wait(self, **kwargs):
            """Simulate successful process reaping."""
            return 0

    monkeypatch.setattr(kaggle.platform, "platform", lambda: "test-platform")
    monkeypatch.setattr(kaggle.subprocess, "Popen", Process)
    monkeypatch.setattr(kaggle, "wait_for_server", lambda *_: None)
    monkeypatch.setenv("VEMBODY_KAGGLE_MODEL", "old-model")
    monkeypatch.delenv("VEMBODY_KAGGLE_API_KEY", raising=False)
    output = tmp_path / "output"

    def session():
        """Exercise cleanup after either a successful pilot or an interruption."""
        with kaggle.serve_model(runtime, output) as base_url:
            assert base_url.startswith("http://127.0.0.1:")
            assert kaggle.os.environ["VEMBODY_KAGGLE_MODEL"] == runtime.spec["served_model"]
            key = captured["env"]["LLAMA_API_KEY"]
            assert key not in (output / "runtime.json").read_text()
            assert key not in " ".join(captured["command"])
            if fail:
                raise KeyboardInterrupt

    if fail:
        with pytest.raises(KeyboardInterrupt):
            session()
    else:
        session()

    assert captured["stdout"].closed
    assert captured["process"].stopped
    assert kaggle.os.environ["VEMBODY_KAGGLE_MODEL"] == "old-model"
    assert "VEMBODY_KAGGLE_API_KEY" not in kaggle.os.environ


def test_startup_rejects_wrong_model(runtime, monkeypatch):
    monkeypatch.setattr(
        kaggle, "urlopen", lambda *_, **__: io.BytesIO(b'{"data":[{"id":"wrong"}]}')
    )
    with pytest.raises(RuntimeError, match="unexpected model"):
        kaggle.wait_for_server(
            SimpleNamespace(poll=lambda: None), "http://localhost/v1", "key", "test"
        )


def test_startup_rejects_dead_process():
    with pytest.raises(RuntimeError, match="exited"):
        kaggle.wait_for_server(
            SimpleNamespace(poll=lambda: 1), "http://localhost/v1", "key", "test"
        )


def test_startup_timeout_is_bounded(monkeypatch):
    clock = iter([0, 181])
    monkeypatch.setattr(kaggle.time, "monotonic", lambda: next(clock))
    with pytest.raises(RuntimeError, match="did not become ready"):
        kaggle.wait_for_server(
            SimpleNamespace(poll=lambda: None), "http://localhost/v1", "key", "test"
        )


def test_pinned_source_mismatch_is_not_silently_reused(tmp_path, monkeypatch):
    (tmp_path / "llama.cpp").mkdir()
    monkeypatch.setattr(kaggle.subprocess, "check_output", lambda *_, **__: "wrong")
    with pytest.raises(RuntimeError, match="revision differs"):
        kaggle.build_server(tmp_path, {"llama_revision": "expected"})


def test_notebook_has_only_compilable_unexecuted_code_cells():
    notebook = json.loads(Path("notebooks/kaggle-ui-mate-q6.ipynb").read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell["source"]), f"kaggle-cell-{index}", "exec")


@pytest.mark.parametrize("unpack", [False, True])
def test_notebook_bootstrap_handles_zipped_and_kaggle_unpacked_inputs(tmp_path, unpack):
    content = b"# fixture source"
    import hashlib

    manifest = {"files": {"src/example.py": hashlib.sha256(content).hexdigest()}}
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    archive = input_dir / "vembody-kaggle.zip"
    with ZipFile(archive, "w") as bundle:
        bundle.writestr("vembody/src/example.py", content)
        bundle.writestr("vembody/bundle-manifest.json", json.dumps(manifest))

    if unpack:
        with ZipFile(archive) as bundle:
            bundle.extractall(input_dir)

        archive.unlink()

    notebook = json.loads(Path("notebooks/kaggle-ui-mate-q6.ipynb").read_text())
    source = "".join(notebook["cells"][1]["source"])
    for old, new in (
        ("/kaggle/input", input_dir),
        ("/kaggle/temp", tmp_path / "scratch"),
        ("/kaggle/working", tmp_path / "output"),
    ):
        for quote in ("'", '"'):
            source = source.replace(quote + old + quote, repr(str(new)))

    namespace = {}
    exec(compile(source, "notebook-bootstrap", "exec"), namespace)
    assert (namespace["REPO"] / "src/example.py").read_bytes() == content


def test_bundle_excludes_credentials_weights_git_and_run_outputs(tmp_path, monkeypatch):
    names = [
        "src/vembody/new.py",
        "configs/models/kaggle.yaml",
        "notebooks/pilot.ipynb",
        ".env",
        ".git/config",
        "runs/log.json",
        "src/.env",
        "src/weights.gguf",
        "src/__pycache__/module.pyc",
        "artifacts/output.json",
    ]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")

    monkeypatch.setattr(
        packaging.subprocess,
        "run",
        lambda *_, **__: subprocess.CompletedProcess([], 0, stdout="\0".join(names).encode()),
    )
    monkeypatch.setattr(packaging.subprocess, "check_output", lambda *_, **__: "base-commit")
    destination = packaging.package(tmp_path, tmp_path / "bundle.zip")
    with ZipFile(destination) as archive:
        actual = set(archive.namelist())
        assert actual == {
            "vembody/src/vembody/new.py",
            "vembody/configs/models/kaggle.yaml",
            "vembody/notebooks/pilot.ipynb",
            "vembody/bundle-manifest.json",
        }
        manifest = json.loads(archive.read("vembody/bundle-manifest.json"))
        assert manifest["snapshot_includes_uncommitted_code"]
        assert len(manifest["files"]["src/vembody/new.py"]) == 64


def test_versioned_cuda_driver_is_found_without_development_symlink(tmp_path, monkeypatch):
    driver = tmp_path / "libcuda.so.1"
    driver.write_bytes(b"fixture")
    listing = f"libcuda.so.1 (libc6,x86-64) => {driver}\n"
    monkeypatch.setattr(kaggle.subprocess, "run", lambda *_, **__: SimpleNamespace(stdout=listing))
    assert kaggle.find_cuda_driver() == driver.resolve()


def test_cuda_driver_search_rejects_stub_and_reports_missing_driver(tmp_path, monkeypatch):
    stub = tmp_path / "stubs" / "libcuda.so.1"
    stub.parent.mkdir()
    stub.write_bytes(b"stub")
    monkeypatch.setattr(kaggle, "CUDA_DRIVER_DIRS", (stub.parent,))
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    monkeypatch.setattr(kaggle.subprocess, "run", lambda *_, **__: SimpleNamespace(stdout=""))
    with pytest.raises(RuntimeError, match="real libcuda"):
        kaggle.find_cuda_driver()


def test_driver_search_uses_library_path_when_ldconfig_is_unavailable(tmp_path, monkeypatch):
    driver = tmp_path / "libcuda.so.1"
    driver.write_bytes(b"fixture")
    monkeypatch.setenv("LD_LIBRARY_PATH", str(tmp_path))

    def unavailable(*args, **kwargs):
        raise FileNotFoundError("ldconfig")

    monkeypatch.setattr(kaggle.subprocess, "run", unavailable)
    # Drive-letter colons are not Linux LD_LIBRARY_PATH delimiters in this Windows test.
    monkeypatch.setattr(kaggle, "CUDA_DRIVER_DIRS", (tmp_path,))
    assert kaggle.find_cuda_driver() == driver.resolve()


@pytest.mark.parametrize("status,count,valid", [(0, 2, True), (100, 0, False), (0, 1, False)])
def test_cuda_initialization_is_checked_before_cmake(monkeypatch, status, count, valid):
    def initialize(flags):
        return status

    def device_count(pointer):
        pointer._obj.value = count
        return 0

    driver = SimpleNamespace(cuInit=initialize, cuDeviceGetCount=device_count)
    monkeypatch.setattr(kaggle.ctypes, "CDLL", lambda *_: driver)
    if valid:
        assert kaggle.validate_cuda_driver(Path("libcuda.so.1")) == 2
    else:
        with pytest.raises(RuntimeError, match="driver validation failed"):
            kaggle.validate_cuda_driver(Path("libcuda.so.1"))


def test_cmake_receives_exact_driver_cache_variable(tmp_path, monkeypatch):
    (tmp_path / "llama.cpp").mkdir()
    driver = tmp_path / "libcuda.so.1"
    calls = []
    monkeypatch.setattr(kaggle.subprocess, "check_output", lambda *_, **__: "expected")
    monkeypatch.setattr(kaggle.subprocess, "run", lambda command, **kwargs: calls.append(command))
    monkeypatch.setattr(kaggle, "find_cuda_driver", lambda: driver)
    monkeypatch.setattr(kaggle, "validate_cuda_driver", lambda _: 2)
    monkeypatch.setattr(kaggle, "check_binary", lambda *_: None)
    kaggle.build_server(tmp_path, {"llama_revision": "expected"})
    assert f"-DCUDA_cuda_driver_LIBRARY:FILEPATH={driver}" in calls[0]
    assert "-DCMAKE_CUDA_ARCHITECTURES=75" in calls[0]


@pytest.mark.parametrize("isolated", [True, False])
def test_notebook_installs_only_into_private_python(tmp_path, monkeypatch, isolated):
    directory = tmp_path / "venv"
    directory.mkdir()
    (directory / "pyvenv.cfg").write_text(
        f"include-system-site-packages = {str(not isolated).lower()}\n"
    )
    calls = []
    monkeypatch.setattr(
        environment.subprocess, "run", lambda command, **kwargs: calls.append(command)
    )
    if not isolated:
        with pytest.raises(RuntimeError, match="isolated"):
            environment.create_environment(tmp_path, directory)

        assert len(calls) == 1
    else:
        python = environment.create_environment(tmp_path, directory)
        assert calls[0] == [sys.executable, "-m", "venv", "--without-pip", str(directory)]
        pip = [sys.executable, "-I", "-m", "pip", "--python", str(python)]
        assert calls[1] == [
            *pip,
            "install",
            "-r",
            str(tmp_path / "requirements/kaggle-runtime.txt"),
            "-e",
            str(tmp_path),
        ]
        assert calls[2] == [*pip, "check"]


def test_prepared_runtime_survives_isolated_stage_boundary(runtime, tmp_path, monkeypatch):
    for path in (runtime.binary, runtime.model, runtime.projector):
        path.write_bytes(b"fixture")

    state = tmp_path / "prepared.json"
    runner.save_runtime(state, runtime)
    monkeypatch.setattr(runner, "gpu_info", lambda: "fresh T4 check")
    restored = runner.load_runtime(state, runtime.spec)
    assert restored.model == runtime.model
    assert restored.gpu_info == "fresh T4 check"
    with pytest.raises(RuntimeError, match="settings differ"):
        runner.load_runtime(state, {**runtime.spec, "quantization": "different"})

    runtime.model.unlink()
    with pytest.raises(RuntimeError, match="missing"):
        runner.load_runtime(state, runtime.spec)


def test_notebook_launches_both_stages_with_isolated_interpreter():
    notebook = json.loads(Path("notebooks/kaggle-ui-mate-q6.ipynb").read_text())
    install = "".join(notebook["cells"][2]["source"])
    assert "create_environment" in install
    assert "sys.executable, '-m', 'pip'" not in install
    assert 'sys.modules.pop("kaggle_environment", None)' in install
    assert 'MODE = "autonomous"' in install
    assert 'AUTONOMOUS_PROFILE = "brief"' in install
    for index in (4, 5):
        source = "".join(notebook["cells"][index]["source"])
        calls = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run"
        ]
        assert len(calls) == 1
        command = calls[0].args[0]
        assert isinstance(command, ast.List)
        assert ast.unparse(command.elts[0]) == "str(PYTHON)"
        assert isinstance(command.elts[1], ast.Constant) and command.elts[1].value == "-I"

    assert '"prepare"' in "".join(notebook["cells"][4]["source"])
    assert "--cache-output" in "".join(notebook["cells"][4]["source"])
    assert "MODE" in "".join(notebook["cells"][5]["source"])
    assert "FileLink" not in "".join(notebook["cells"][6]["source"])


@pytest.mark.parametrize("minimum,maximum", [(8, 1024), (2048, 1024), (True, 1024)])
def test_server_rejects_insufficient_or_inconsistent_image_budget(runtime, minimum, maximum):
    runtime.spec.update(image_min_tokens=minimum, image_max_tokens=maximum)
    with pytest.raises(ValueError, match="image_min_tokens"):
        kaggle.server_command(runtime, 8000)


def test_existing_compiled_server_is_checked_without_recompiling(tmp_path, monkeypatch):
    binary = tmp_path / "llama.cpp/build/bin/llama-server"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"fixture")
    checked = []
    monkeypatch.setattr(kaggle.subprocess, "check_output", lambda *_, **__: "expected")
    monkeypatch.setattr(kaggle, "find_cuda_driver", lambda: Path("libcuda.so.1"))
    monkeypatch.setattr(kaggle, "validate_cuda_driver", lambda _: 2)
    monkeypatch.setattr(kaggle, "check_binary", lambda *args: checked.append(args))

    def no_compilation(*args, **kwargs):
        pytest.fail("An already compiled compatible server should not be rebuilt")

    monkeypatch.setattr(kaggle.subprocess, "run", no_compilation)
    assert kaggle.build_server(tmp_path, {"llama_revision": "expected"}) == binary
    assert len(checked) == 1


def test_notebook_setup_replaces_a_cached_helper_from_an_old_bundle(tmp_path, monkeypatch):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    helper = scripts / "kaggle_environment.py"
    helper.write_text("def create_environment(repo, directory):\n    return directory / 'python'\n")
    monkeypatch.syspath_prepend(str(scripts))
    monkeypatch.setitem(sys.modules, "kaggle_environment", SimpleNamespace(create_environment=None))
    notebook = json.loads(Path("notebooks/kaggle-ui-mate-q6.ipynb").read_text())
    calls = []
    namespace = {
        "REPO": tmp_path,
        "sys": sys,
        "Path": Path,
        "snapshot": "fixture-snapshot",
        "subprocess": SimpleNamespace(run=lambda command, **kwargs: calls.append(command)),
    }
    exec(compile("".join(notebook["cells"][2]["source"]), "notebook-setup", "exec"), namespace)
    assert namespace["create_environment"].__code__.co_filename == str(helper)
    assert namespace["PYTHON"].name == "python"
    assert "-I" in calls[0] and "check_ui_mate_compatibility.py" in calls[0][2]
