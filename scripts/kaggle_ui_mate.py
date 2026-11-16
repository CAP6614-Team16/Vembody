"""Prepare a pinned Q6_K server and bounded pilot inside a Kaggle GPU session."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import secrets
import socket
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from scripts.kaggle_cache import check_binary, restore_runtime

CUDA_DRIVER_DIRS = (
    Path("/usr/lib/x86_64-linux-gnu"),
    Path("/lib/x86_64-linux-gnu"),
    Path("/usr/lib64-nvidia"),
    Path("/usr/local/nvidia/lib64"),
    Path("/usr/local/nvidia/lib"),
    Path("/usr/lib64"),
)


@dataclass(frozen=True)
class Runtime:
    """Resolved local model files, server binary, and non-secret provenance."""

    binary: Path
    model: Path
    projector: Path
    spec: dict
    gpu_info: str


def file_sha256(path: Path) -> str:
    """Hash a checkpoint incrementally rather than loading gigabytes into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def gpu_info() -> str:
    """Verify the expected two T4 GPUs before spending time building/downloading."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    lines = result.stdout.strip().splitlines()
    if len(lines) != 2 or any("T4" not in line for line in lines):
        raise RuntimeError("Select Kaggle's GPU T4 x2 accelerator before continuing")

    return result.stdout.strip()


def find_cuda_driver() -> Path:
    """Locate a real versioned driver library even without the development libcuda.so link."""
    candidates = []
    try:
        result = subprocess.run(
            ["/sbin/ldconfig", "-p"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        for line in result.stdout.splitlines():
            name, separator, path = line.partition("=>")
            parts = name.split()
            if separator and parts and parts[0] in {"libcuda.so", "libcuda.so.1"}:
                candidates.append(Path(path.strip()))
    except (OSError, subprocess.SubprocessError):
        pass

    directories = [
        Path(value) for value in os.environ.get("LD_LIBRARY_PATH", "").split(":") if value
    ]
    for directory in [*directories, *CUDA_DRIVER_DIRS]:
        candidates.extend(directory / name for name in ("libcuda.so.1", "libcuda.so"))

    for candidate in candidates:
        if candidate.is_file() and "stubs" not in candidate.resolve().parts:
            return candidate.resolve()

    raise RuntimeError("Cannot locate the real libcuda.so.1; restart a GPU-enabled Kaggle session")


def validate_cuda_driver(path: Path) -> int:
    """Confirm the driver initializes CUDA and exposes both GPUs before starting CMake."""
    try:
        driver = ctypes.CDLL(str(path))
        driver.cuInit.argtypes = [ctypes.c_uint]
        driver.cuInit.restype = ctypes.c_int
        driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        driver.cuDeviceGetCount.restype = ctypes.c_int
        count = ctypes.c_int()
        initialized = driver.cuInit(0)
        status = driver.cuDeviceGetCount(ctypes.byref(count)) if initialized == 0 else initialized
    except (OSError, AttributeError) as exc:
        raise RuntimeError("The located CUDA driver library cannot be loaded") from exc

    if status != 0 or count.value != 2:
        raise RuntimeError(f"CUDA driver validation failed: status={status}, GPUs={count.value}")

    return count.value


def build_server(work: Path, spec: dict) -> Path:
    """Build the pinned llama.cpp CUDA server for Turing compute capability 7.5."""
    source = work / "llama.cpp"
    if not source.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                spec["llama_tag"],
                "https://github.com/ggml-org/llama.cpp.git",
                str(source),
            ],
            check=True,
        )

    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True)
    if revision.strip() != spec["llama_revision"]:
        raise RuntimeError("Cached llama.cpp revision differs; choose a fresh work directory")

    driver = find_cuda_driver()
    devices = validate_cuda_driver(driver)
    print(f"CUDA driver: {driver} ({devices} GPUs initialized)", flush=True)
    build = source / "build"
    binary = build / "bin" / "llama-server"
    if binary.is_file():
        check_binary(binary, spec)
        print("Reusing the existing compiled server", flush=True)
        return binary

    subprocess.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            "-DGGML_CUDA=ON",
            "-DCMAKE_CUDA_ARCHITECTURES=75",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DLLAMA_CURL=OFF",
            f"-DCUDA_cuda_driver_LIBRARY:FILEPATH={driver}",
        ],
        check=True,
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build),
            "--config",
            "Release",
            "--target",
            "llama-server",
            "-j",
            "2",
        ],
        check=True,
    )
    check_binary(binary, spec)
    return binary


def prepare_runtime(work: Path, config_path: Path) -> Runtime:
    """Download only Q6_K and its FP16 vision projector, verifying both checksums."""
    from huggingface_hub import hf_hub_download

    devices = gpu_info()
    print(devices, flush=True)
    work.mkdir(parents=True, exist_ok=True)
    spec = json.loads(config_path.read_text(encoding="utf-8"))
    binary = restore_runtime(work, spec)
    if binary is None:
        binary = build_server(work, spec)
    else:
        validate_cuda_driver(find_cuda_driver())
    files = {}
    for role, item in spec["files"].items():
        path = Path(
            hf_hub_download(
                repo_id=spec["repo_id"],
                filename=item["name"],
                revision=spec["revision"],
                cache_dir=str(work / "weights"),
                token=False,
            )
        )
        print(f"Checking {item['name']}", flush=True)
        if file_sha256(path) != item["sha256"]:
            raise RuntimeError(f"Checkpoint checksum mismatch: {item['name']}")

        files[role] = path

    return Runtime(binary, files["model"], files["projector"], spec, devices)


def server_command(runtime: Runtime, port: int) -> list[str]:
    """Keep inference on localhost, distribute layers across both GPUs, and retain native XML."""
    minimum = runtime.spec["image_min_tokens"]
    maximum = runtime.spec["image_max_tokens"]
    if type(minimum) is not int or type(maximum) is not int or not 1024 <= minimum <= maximum:
        raise ValueError("UI-Mate grounding requires 1024 <= image_min_tokens <= image_max_tokens")

    return [
        str(runtime.binary),
        "--model",
        str(runtime.model),
        "--mmproj",
        str(runtime.projector),
        "--alias",
        runtime.spec["served_model"],
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--n-gpu-layers",
        "99",
        "--split-mode",
        "layer",
        "--tensor-split",
        "1,1",
        "--ctx-size",
        str(runtime.spec["context_size"]),
        "--parallel",
        "1",
        "--batch-size",
        "512",
        "--ubatch-size",
        "128",
        "--threads",
        "4",
        "--image-min-tokens",
        str(runtime.spec["image_min_tokens"]),
        "--image-max-tokens",
        str(runtime.spec["image_max_tokens"]),
        "--jinja",
        "--skip-chat-parsing",
        "--reasoning-format",
        "none",
        "--no-webui",
    ]


def wait_for_server(process: subprocess.Popen, base_url: str, key: str, model: str) -> None:
    """Wait at most three minutes and confirm the expected served model before inference."""
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("llama-server exited during startup; inspect server.log")

        try:
            request = Request(base_url + "/models", headers={"Authorization": "Bearer " + key})
            with urlopen(request, timeout=3) as response:
                data = json.load(response)

            if model not in {item["id"] for item in data["data"]}:
                raise RuntimeError("The endpoint is serving an unexpected model")

            health_url = base_url.removesuffix("/v1") + "/health"
            with urlopen(health_url, timeout=3) as response:
                health = json.load(response)

            if health.get("status") == "ok":
                return
        except (URLError, TimeoutError):
            pass

        time.sleep(2)

    raise RuntimeError("llama-server did not become ready; inspect server.log")


def stop_server(process: subprocess.Popen) -> None:
    """Terminate only the server this notebook started and verify its exit."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    if process.poll() is None:
        raise RuntimeError("The notebook's llama-server did not stop")


@contextmanager
def serve_model(runtime: Runtime, output: Path) -> Iterator[str]:
    """Serve locally with an ephemeral key and always stop on completion or interruption."""
    output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    key = secrets.token_urlsafe(32)
    base_url = f"http://127.0.0.1:{port}/v1"
    values = {
        "VEMBODY_KAGGLE_BASE_URL": base_url,
        "VEMBODY_KAGGLE_MODEL": runtime.spec["served_model"],
        "VEMBODY_KAGGLE_API_KEY": key,
    }
    old = {name: os.environ.get(name) for name in values}
    environment = {**os.environ, "LLAMA_API_KEY": key}
    environment["LD_LIBRARY_PATH"] = (
        str(runtime.binary.parent) + os.pathsep + environment.get("LD_LIBRARY_PATH", "")
    )
    command = server_command(runtime, port)
    provenance = {
        "model": runtime.spec,
        "gpu": runtime.gpu_info,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "command": command,
        "base_url": base_url,
    }
    (output / "runtime.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    with (output / "server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
        try:
            wait_for_server(process, base_url, key, runtime.spec["served_model"])
            os.environ.update(values)
            print("Q6_K server ready on localhost", flush=True)
            yield base_url
        finally:
            try:
                stop_server(process)
            finally:
                for name, value in old.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value


def smoke_model(repo: Path, output: Path, adapter: str = "ui-mate") -> None:
    """Check one screenshot/action without executing it; persist latency and original output."""
    from vembody.models.registry import build_model
    from vembody.models.settings import load_endpoint_config
    from vembody.parsing import parse_model_response
    from vembody.results import to_jsonable
    from vembody.sandbox import MinesweeperSandbox
    from vembody.tasks import agent_task, load_task
    from vembody.validation import ActionPolicy, validate_action

    task = replace(load_task(repo / "tasks/benchmark/development/t1-01.yaml"), timeout_seconds=300)
    sandbox = MinesweeperSandbox(task, output / "smoke-screenshots")
    config = load_endpoint_config(repo / "configs/models/kaggle.yaml")
    model = build_model(adapter, task, model_config=config)
    try:
        observation = sandbox.capture(task, "screenshot-smoke", 0, [])
        response = model.generate_action(agent_task(task), observation, [])
        (output / "smoke-response.json").write_text(
            json.dumps(to_jsonable(response), indent=2) + "\n", encoding="utf-8"
        )
        action = parse_model_response(response.raw_text).action
        validate_action(action, ActionPolicy.from_task(task))
        print(f"Valid proposed action: {to_jsonable(action)}; {response.elapsed_seconds:.2f}s")
    finally:
        sandbox.shutdown()


def run_pilot(repo: Path, output: Path, adapter: str = "ui-mate") -> list[dict]:
    """Run only the first grounding task once with a documented five-minute smoke budget."""
    from vembody.evaluation.harness import ExperimentConfig, run_experiment

    config = ExperimentConfig(
        name="ui-mate-q6-pilot",
        model=adapter,
        task_paths=[str(repo / "tasks/benchmark/development/t1-01.yaml")],
        trials=1,
        timeout_seconds=300,
        model_config=str(repo / "configs/models/kaggle.yaml"),
        labels={"purpose": "pilot", "memory": "off", "quantization": "Q6_K"},
    )
    rows = run_experiment(config, output / "pilot")
    print(json.dumps(rows, indent=2))
    return rows
