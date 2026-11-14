import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml
from PIL import Image

from vembody.cli import main
from vembody.errors import VembodyError
from vembody.models.http import HttpVisionModel
from vembody.models.settings import load_endpoint_config
from vembody.models.ui_mate import BriefUiMateModel
from vembody.observations import Observation
from vembody.parsing import parse_model_response
from vembody.results import StepResult, to_jsonable
from vembody.tasks import load_task

TASK = "tasks/benchmark/development/t1-01.yaml"
SECRET = "test-only-credential"


@pytest.fixture
def endpoint(monkeypatch):
    """Serve deterministic completions locally without contacting a paid service."""
    task = load_task(TASK)
    state = {
        "requests": [],
        "status": 200,
        "headers": {},
        "body": json.dumps(
            {
                "model": "test-vlm",
                "choices": [
                    {
                        "message": {"content": json.dumps({"action": task.mock_actions[0]})},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 40, "completion_tokens": 20, "total_tokens": 60},
            }
        ).encode(),
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            """Capture the request and return the configured test response."""
            state["requests"].append(
                {
                    "path": self.path,
                    "headers": dict(self.headers),
                    "payload": json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
                }
            )
            self.send_response(state["status"])
            for key, value in state["headers"].items():
                self.send_header(key, value)

            self.send_header("Content-Length", str(len(state["body"])))
            self.end_headers()
            self.wfile.write(state["body"])

        def log_message(self, *_):
            """Suppress test-server access logs."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    state["url"] = f"http://127.0.0.1:{server.server_port}/v1"
    for profile in ("ANANT", "HOSTED"):
        monkeypatch.setenv(f"VEMBODY_{profile}_BASE_URL", state["url"])
        monkeypatch.setenv(f"VEMBODY_{profile}_MODEL", "test-vlm")
        monkeypatch.setenv(f"VEMBODY_{profile}_API_KEY", SECRET)

    try:
        yield state
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def screenshot_observation(tmp_path):
    """Create a small image observation with metadata that must not reach the server."""
    path = tmp_path / "screenshot.png"
    Image.new("RGB", (32, 24), "green").save(path)
    return Observation(
        "episode",
        0,
        "task",
        screenshot_path=str(path),
        metadata={"private": "private-grading-evidence"},
    )


@pytest.mark.parametrize("profile", ["anant", "hosted"])
def test_profiles_share_safe_http_contract(endpoint, tmp_path, profile):
    config = load_endpoint_config(f"configs/models/{profile}.yaml")
    model = HttpVisionModel(config)
    response = model.generate_action(
        load_task(TASK),
        screenshot_observation(tmp_path),
        [StepResult(None, error="private-history-evidence")],
    )

    assert parse_model_response(response.raw_text).action.type.value == "click"
    (request,) = endpoint["requests"]
    assert request["path"] == "/v1/chat/completions"
    assert request["headers"]["Authorization"] == f"Bearer {SECRET}"
    payload = request["payload"]
    assert payload["model"] == "test-vlm"
    assert payload["temperature"] == 0 and payload["max_tokens"] == config.max_tokens
    assert payload["stream"] is False
    prompt, image = payload["messages"][1]["content"]
    assert json.loads(prompt["text"])["instruction"] == load_task(TASK).instruction
    assert "private-" not in json.dumps(payload)
    assert "expected_state_hash" not in json.dumps(payload)
    assert "mock_actions" not in json.dumps(payload)
    assert str(tmp_path) not in json.dumps(payload)
    with Image.open(io.BytesIO(base64.b64decode(image["image_url"]["url"].split(",")[1]))) as sent:
        assert sent.size == (32, 24)

    assert response.metadata["prompt_tokens"] == 40
    assert response.metadata["image_count"] == 1
    assert response.elapsed_seconds > 0
    assert SECRET not in json.dumps(response.metadata)
    assert SECRET not in json.dumps(to_jsonable(config))
    assert SECRET not in repr(config)


def test_brief_profile_sends_larger_budget_and_records_real_http_response(
    endpoint, tmp_path, monkeypatch
):
    for suffix, value in (
        ("BASE_URL", endpoint["url"]),
        ("MODEL", "test-vlm"),
        ("API_KEY", SECRET),
    ):
        monkeypatch.setenv(f"VEMBODY_KAGGLE_{suffix}", value)

    native = (
        "<action>Reveal the next cell.</action><tool_call><function=computer_use>"
        "<parameter=action>left_click</parameter><parameter=coordinate>[500,250]</parameter>"
        "</function></tool_call>"
    )
    body = json.loads(endpoint["body"])
    body["choices"][0]["message"]["content"] = native
    endpoint["body"] = json.dumps(body).encode()
    config = load_endpoint_config("configs/models/kaggle-autonomous.yaml")
    model = BriefUiMateModel(config)
    response = model.generate_action(
        load_task("tasks/benchmark/development/t4-01.yaml"), screenshot_observation(tmp_path), []
    )
    assert parse_model_response(response.raw_text).action.type.value == "click"
    (request,) = endpoint["requests"]
    payload = request["payload"]
    assert payload["max_tokens"] == 4096 and payload["temperature"] == 0
    assert config.timeout_seconds == 240
    assert "one short sentence" in payload["messages"][0]["content"][0]["text"]
    assert "expected_state_hash" not in json.dumps(payload)
    assert "private-" not in json.dumps(payload)
    assert response.metadata["prompt_version"] == "ui-mate-minesweeper-v3-brief"
    assert response.metadata["finish_reason"] == "stop"
    assert response.metadata["native_response"] == native


@pytest.mark.parametrize("status", [401, 429, 503, 302, 307])
def test_http_errors_do_not_retry_redirect_or_expose_credentials(endpoint, tmp_path, status):
    endpoint["status"] = status
    endpoint["headers"] = {"Location": endpoint["url"] + "/elsewhere"}
    endpoint["body"] = SECRET.encode()
    model = HttpVisionModel(load_endpoint_config("configs/models/anant.yaml"))
    with pytest.raises(VembodyError, match=f"HTTP {status}") as error:
        model.generate_action(load_task(TASK), screenshot_observation(tmp_path), [])

    assert SECRET not in str(error.value)
    assert len(endpoint["requests"]) == 1


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b"{}",
        b"[]",
        b'{"choices": []}',
        b'{"choices": [{"message": {"content": null}}]}',
    ],
)
def test_bad_server_envelopes_are_rejected(endpoint, tmp_path, body):
    endpoint["body"] = body
    model = HttpVisionModel(load_endpoint_config("configs/models/anant.yaml"))
    with pytest.raises(VembodyError, match="invalid chat-completion envelope"):
        model.generate_action(load_task(TASK), screenshot_observation(tmp_path), [])


def test_oversized_response_is_rejected(endpoint, tmp_path):
    endpoint["body"] = b" " * (1024 * 1024 + 1)
    model = HttpVisionModel(load_endpoint_config("configs/models/anant.yaml"))
    with pytest.raises(VembodyError, match="size limit"):
        model.generate_action(load_task(TASK), screenshot_observation(tmp_path), [])


def test_timeout_is_bounded_and_not_retried(endpoint, tmp_path, monkeypatch):
    model = HttpVisionModel(load_endpoint_config("configs/models/anant.yaml"))
    calls = []

    def timeout(request, **kwargs):
        """Simulate a timed-out socket without waiting in the test suite."""
        calls.append(kwargs["timeout"])
        raise TimeoutError(SECRET)

    monkeypatch.setattr(model.opener, "open", timeout)
    with pytest.raises(VembodyError, match="timed out") as error:
        model.generate_action(load_task(TASK), screenshot_observation(tmp_path), [])

    assert calls == [60]
    assert SECRET not in str(error.value)


def test_missing_screenshot_does_not_contact_endpoint(endpoint):
    model = HttpVisionModel(load_endpoint_config("configs/models/anant.yaml"))
    with pytest.raises(VembodyError, match="requires a screenshot"):
        model.generate_action(load_task(TASK), Observation("e", 0, "t"), [])

    assert not endpoint["requests"]


@pytest.mark.parametrize("profile", ["anant", "hosted"])
def test_run_and_eval_use_selected_profile(endpoint, tmp_path, profile):
    options = ["--model", "http", "--model-config", f"configs/models/{profile}.yaml"]
    log = tmp_path / "episode.jsonl"
    assert main(["run", "--task", TASK, "--log", str(log), *options]) == 0
    episode = json.loads(log.read_text().splitlines()[-1])["episode"]
    assert episode["evaluation"]["status"] == "success"
    assert (
        main(
            [
                "eval",
                "--tasks",
                TASK,
                "--trials",
                "1",
                "--name",
                "http-test",
                "--out",
                str(tmp_path),
                *options,
            ]
        )
        == 0
    )
    experiment = tmp_path / "http-test"
    manifest = json.loads((experiment / "manifest.json").read_text())
    assert manifest["model_endpoint"]["model"] == "test-vlm"
    assert manifest["model_endpoint"]["base_url"] == endpoint["url"]
    assert manifest["model_endpoint"]["prompt_version"] == "screenshot-action-v1"
    row = json.loads((experiment / "episodes.jsonl").read_text())
    assert row["success"] and row["prompt_tokens_total"] == 40
    assert len(endpoint["requests"]) == 2
    for path in tmp_path.rglob("*.json*"):
        assert SECRET not in path.read_text()


def test_bad_action_json_still_uses_strict_loop_parser(endpoint, tmp_path):
    endpoint["body"] = json.dumps(
        {"choices": [{"message": {"content": "not action JSON"}}]}
    ).encode()
    log = tmp_path / "invalid.jsonl"
    assert (
        main(
            [
                "run",
                "--task",
                TASK,
                "--model",
                "http",
                "--model-config",
                "configs/models/anant.yaml",
                "--log",
                str(log),
            ]
        )
        == 0
    )
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert lines[-1]["episode"]["termination_reason"] == "parse_error"
    assert lines[0]["step"]["execution"] is None
    assert lines[0]["step"]["model_metadata"]["prompt_version"] == "screenshot-action-v1"


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/v1",
        "https://user:password@example.com/v1",
        "https://example.com/v1?key=secret",
        "file:///tmp/model",
        "https://example.com:invalid/v1",
        "https://example.com/v1#fragment",
    ],
)
def test_unsafe_endpoint_urls_are_rejected(endpoint, monkeypatch, url):
    monkeypatch.setenv("VEMBODY_ANANT_BASE_URL", url)
    with pytest.raises(ValueError):
        load_endpoint_config("configs/models/anant.yaml")


@pytest.mark.parametrize(
    "field,value",
    [
        ("timeout_seconds", 0),
        ("timeout_seconds", float("nan")),
        ("timeout_seconds", True),
        ("temperature", -1),
        ("temperature", float("inf")),
        ("max_tokens", True),
        ("max_tokens", 0),
        ("unexpected", "field"),
    ],
)
def test_invalid_profile_settings_are_rejected(endpoint, tmp_path, field, value):
    config = yaml.safe_load(Path("configs/models/anant.yaml").read_text())
    config[field] = value
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        load_endpoint_config(path)


def test_missing_auth_fails_before_creating_experiment(endpoint, monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("VEMBODY_ANANT_API_KEY")
    assert (
        main(
            [
                "eval",
                "--tasks",
                TASK,
                "--model",
                "http",
                "--model-config",
                "configs/models/anant.yaml",
                "--name",
                "missing-auth",
                "--out",
                str(tmp_path),
            ]
        )
        == 1
    )
    assert "VEMBODY_ANANT_API_KEY" in capsys.readouterr().err
    assert not (tmp_path / "missing-auth").exists()
    assert not endpoint["requests"]


def test_missing_or_mismatched_profile_is_an_explicit_error(tmp_path):
    assert main(["run", "--task", TASK, "--model", "http"]) == 1
    assert (
        main(
            [
                "run",
                "--task",
                TASK,
                "--model",
                "mock",
                "--model-config",
                "configs/models/anant.yaml",
            ]
        )
        == 1
    )
    with pytest.raises(ValueError):
        load_endpoint_config(tmp_path / "missing.yaml")


@pytest.mark.parametrize("variable", ["VEMBODY_ANANT_BASE_URL", "VEMBODY_ANANT_MODEL"])
def test_missing_connection_variables_are_named(endpoint, monkeypatch, variable):
    monkeypatch.delenv(variable)
    with pytest.raises(ValueError, match=variable):
        load_endpoint_config("configs/models/anant.yaml")


def test_hosted_https_and_trailing_slash_are_supported(endpoint, monkeypatch):
    monkeypatch.setenv("VEMBODY_HOSTED_BASE_URL", "https://provider.example/api/v1/")
    config = load_endpoint_config("configs/models/hosted.yaml")
    assert config.base_url == "https://provider.example/api/v1"
    assert config.model == "test-vlm"


@pytest.mark.parametrize("contents", ["[", "- list-not-mapping", "api_key_env: 123"])
def test_invalid_profile_files_are_rejected(tmp_path, contents):
    path = tmp_path / "invalid.yaml"
    path.write_text(contents)
    with pytest.raises(ValueError):
        load_endpoint_config(path)


def test_endpoint_failure_never_executes_an_action(endpoint, tmp_path):
    endpoint["status"] = 503
    log = tmp_path / "failed.jsonl"
    assert (
        main(
            [
                "run",
                "--task",
                TASK,
                "--model",
                "http",
                "--model-config",
                "configs/models/anant.yaml",
                "--log",
                str(log),
            ]
        )
        == 0
    )
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert records[0]["step"]["execution"] is None
    assert records[-1]["episode"]["termination_reason"] == "model_failure"
    assert len(endpoint["requests"]) == 1


def test_ui_mate_native_endpoint_completes_grounding_task(endpoint, tmp_path):
    task = load_task(TASK)
    target = task.mock_actions[0]
    native = (
        "<action>Reveal the target.</action><tool_call><function=computer_use>"
        "<parameter=action>left_click</parameter><parameter=coordinate>"
        f"[{target['x'] * 999}, {target['y'] * 999}]</parameter></function></tool_call>"
    )
    body = json.loads(endpoint["body"])
    body["choices"][0]["message"]["content"] = native
    endpoint["body"] = json.dumps(body).encode()
    assert (
        main(
            [
                "eval",
                "--tasks",
                TASK,
                "--model",
                "ui-mate",
                "--model-config",
                "configs/models/anant.yaml",
                "--trials",
                "1",
                "--name",
                "native",
                "--out",
                str(tmp_path),
            ]
        )
        == 0
    )
    directory = tmp_path / "native"
    row = json.loads((directory / "episodes.jsonl").read_text())
    assert row["success"] and row["prompt_tokens_total"] == 40
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["model_endpoint"]["prompt_version"] == "ui-mate-minesweeper-v2"
    log = next((directory / "logs").rglob("*.jsonl"))
    step = json.loads(log.read_text().splitlines()[0])["step"]
    assert step["model_metadata"]["native_response"] == native
    assert step["action"]["type"] == "click"
    assert step["model_metadata"]["image_size"] == [336, 376]
    assert step["model_metadata"]["encoded_image_size"] == [320, 384]
    assert step["model_metadata"]["image_transform"] == "factor32-no-padding"
    payload = endpoint["requests"][0]["payload"]
    image_block = payload["messages"][1]["content"][0]
    with Image.open(
        io.BytesIO(base64.b64decode(image_block["image_url"]["url"].split(",")[1]))
    ) as sent:
        assert sent.size == (320, 384)


def test_ui_mate_parse_failure_preserves_native_output_and_tokens(endpoint, tmp_path):
    native = "pyautogui.click(10, 20)"
    body = json.loads(endpoint["body"])
    body["choices"][0]["message"]["content"] = native
    endpoint["body"] = json.dumps(body).encode()
    log = tmp_path / "bad-native.jsonl"
    assert (
        main(
            [
                "run",
                "--task",
                TASK,
                "--model",
                "ui-mate",
                "--model-config",
                "configs/models/anant.yaml",
                "--log",
                str(log),
            ]
        )
        == 0
    )
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert records[-1]["episode"]["termination_reason"] == "parse_error"
    assert records[0]["step"]["execution"] is None
    assert records[0]["step"]["model_metadata"]["native_response"] == native
    assert records[0]["step"]["model_metadata"]["prompt_tokens"] == 40


def test_kaggle_profile_resolves_pinned_quantization(endpoint, monkeypatch):
    monkeypatch.setenv("VEMBODY_KAGGLE_BASE_URL", endpoint["url"])
    monkeypatch.setenv("VEMBODY_KAGGLE_MODEL", "UI-Mate-9B-Q6_K")
    monkeypatch.setenv("VEMBODY_KAGGLE_API_KEY", SECRET)
    config = load_endpoint_config("configs/models/kaggle.yaml")
    spec = json.loads(Path("configs/kaggle-ui-mate.json").read_text())
    assert config.quantization == spec["quantization"] == "Q6_K"
    assert config.model_revision == spec["revision"]
    assert config.backend == "llama.cpp-" + spec["llama_tag"]
    assert config.top_p == 0.95 and config.max_tokens == 2048


def test_native_success_claim_does_not_establish_task_success(endpoint, tmp_path):
    native = (
        "<tool_call><function=computer_use><parameter=action>finished</parameter>"
        "<parameter=status>success</parameter></function></tool_call>"
    )
    body = json.loads(endpoint["body"])
    body["choices"][0]["message"]["content"] = native
    endpoint["body"] = json.dumps(body).encode()
    log = tmp_path / "claimed-success.jsonl"
    assert (
        main(
            [
                "run",
                "--task",
                TASK,
                "--model",
                "ui-mate",
                "--model-config",
                "configs/models/anant.yaml",
                "--log",
                str(log),
            ]
        )
        == 0
    )
    episode = json.loads(log.read_text().splitlines()[-1])["episode"]
    assert episode["termination_reason"] == "finish_requested"
    assert episode["evaluation"]["status"] == "failure"


def test_autonomous_endpoint_receives_a_goal_not_hidden_targets(endpoint, tmp_path):
    task_path = "tasks/benchmark/development/t4-01.yaml"
    native = (
        "<tool_call><function=computer_use><parameter=action>finished</parameter>"
        "<parameter=status>success</parameter></function></tool_call>"
    )
    body = json.loads(endpoint["body"])
    body["choices"][0]["message"]["content"] = native
    endpoint["body"] = json.dumps(body).encode()
    assert (
        main(
            [
                "eval",
                "--tasks",
                task_path,
                "--model",
                "ui-mate",
                "--model-config",
                "configs/models/anant.yaml",
                "--trials",
                "1",
                "--name",
                "autonomous",
                "--out",
                str(tmp_path),
            ]
        )
        == 0
    )
    (request,) = endpoint["requests"]
    content = json.dumps(request["payload"]["messages"])
    assert load_task(task_path).instruction in content
    assert not any(
        key in content
        for key in (
            "expected_state_hash",
            "mock_actions",
            "revealed",
            "benchmark",
            "seed",
            "safe_total",
        )
    )
    row = json.loads((tmp_path / "autonomous/episodes.jsonl").read_text())
    assert row["max_steps"] == 40
    assert not row["success"] and row["termination_reason"] == "finish_requested"
