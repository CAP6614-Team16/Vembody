"""Screenshot-to-action client for compatible vision chat-completion servers."""

from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from PIL import Image

from ..errors import VembodyError
from ..observations import Observation
from ..results import StepResult
from ..tasks import TaskSpec
from .base import ModelResponse
from .prompts import PROMPT_VERSION, SYSTEM_PROMPT, response_schema, task_prompt
from .settings import EndpointConfig

MAX_RESPONSE_BYTES = 1024 * 1024


class _NoRedirect(HTTPRedirectHandler):
    """Prevent redirects from forwarding credentials or screenshots elsewhere."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """Require an explicit endpoint change rather than following redirects."""
        return None


class HttpVisionModel:
    """Send the current screenshot and recent own actions, never private state."""

    prompt_version = PROMPT_VERSION

    def __init__(self, config: EndpointConfig) -> None:
        """Keep the resolved profile and create a redirect-blocking HTTP client."""
        self.config = config
        self.opener = build_opener(_NoRedirect())

    def build_messages(
        self, task: TaskSpec, image_url: str, history: list[StepResult] = ()
    ) -> list[dict]:
        """Build the generic JSON baseline's instruction, recent actions, and image."""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": task_prompt(task, history)},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ]

    def response_format(self, task: TaskSpec) -> dict | None:
        """Constrain replies to the task's action schema (OpenAI structured outputs)."""
        return {
            "type": "json_schema",
            "json_schema": {"name": "action", "strict": True, "schema": response_schema(task)},
        }

    def prepare_image(self, image: Image.Image) -> Image.Image:
        """Keep generic endpoint screenshots unresized; specialized adapters can override."""
        return image.convert("RGB")

    def generate_action(
        self, task: TaskSpec, observation: Observation, history: list[StepResult]
    ) -> ModelResponse:
        """Request one action; leave strict JSON parsing and validation to the loop."""
        started = perf_counter()
        if observation.screenshot_path is None:
            raise VembodyError("HTTP vision inference requires a screenshot")

        image_buffer = io.BytesIO()
        with Image.open(Path(observation.screenshot_path)) as image:
            size = image.size
            prepared = self.prepare_image(image)
            encoded_size = prepared.size
            prepared.save(image_buffer, format="PNG")

        encoded = base64.b64encode(image_buffer.getvalue()).decode("ascii")
        payload = {
            "model": self.config.model,
            "messages": self.build_messages(task, f"data:image/png;base64,{encoded}", history),
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "stream": False,
        }
        response_format = self.response_format(task)
        if response_format is not None:
            payload["response_format"] = response_format
        request = Request(
            self.config.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": "Bearer " + self.config.api_key(),
            },
            method="POST",
        )

        try:
            with self.opener.open(
                request, timeout=min(self.config.timeout_seconds, task.timeout_seconds)
            ) as response:
                body = response.read(MAX_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            exc.close()
            raise VembodyError(
                f"Model endpoint returned HTTP {exc.code}; no retry was made"
            ) from None
        except (URLError, OSError, TimeoutError):
            raise VembodyError(
                "Model endpoint unavailable or timed out; no retry was made"
            ) from None

        if len(body) > MAX_RESPONSE_BYTES:
            raise VembodyError("Model endpoint response exceeded the size limit")

        try:
            data = json.loads(body)
            choice = data["choices"][0]
            raw_text = choice["message"]["content"]
            if not isinstance(raw_text, str) or not raw_text.strip():
                raise ValueError("No text response")
        except (ValueError, KeyError, IndexError, TypeError):
            raise VembodyError(
                "Model endpoint returned an invalid chat-completion envelope"
            ) from None

        metadata = {
            "provider": "http",
            "model": self.config.model,
            "base_url": self.config.base_url,
            "prompt_version": self.prompt_version,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "model_revision": self.config.model_revision,
            "quantization": self.config.quantization,
            "backend": self.config.backend,
            "image_size": list(size),
            "encoded_image_size": list(encoded_size),
            "response_model": data.get("model"),
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage", {}),
            "image_count": 1,
        }
        usage = data.get("usage")
        if isinstance(usage, dict):
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = usage.get(key)
                if (
                    not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and math.isfinite(value)
                    and value >= 0
                ):
                    metadata[key] = value

        return ModelResponse(raw_text, perf_counter() - started, metadata)
