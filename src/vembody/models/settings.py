"""Credential-free settings for a remote vision-model endpoint."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from ..config import load_config


@dataclass(frozen=True)
class EndpointConfig:
    """Resolved connection settings; store a key's variable name, never its value."""

    base_url: str
    model: str
    api_key_env: str
    timeout_seconds: float = 60.0
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    model_revision: str | None = None
    quantization: str | None = None
    backend: str | None = None

    def api_key(self) -> str:
        """Read the authentication secret without including it in logs or manifests."""
        value = os.environ.get(self.api_key_env, "").strip()
        if not value or "\r" in value or "\n" in value:
            raise ValueError(f"Set a valid {self.api_key_env} environment variable")

        return value


def load_endpoint_config(path: str | Path | None) -> EndpointConfig:
    """Validate a profile and resolve its URL/model from environment variables."""
    if path is None:
        raise ValueError("Remote models require --model-config")

    try:
        data = load_config(path)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError("Could not read model endpoint configuration") from exc

    allowed = {
        "base_url_env",
        "model_env",
        "api_key_env",
        "timeout_seconds",
        "max_tokens",
        "temperature",
        "top_p",
        "model_revision",
        "quantization",
        "backend",
    }
    if set(data) - allowed:
        raise ValueError("Unknown model endpoint configuration field")

    names = {}
    for field in ("base_url_env", "model_env", "api_key_env"):
        name = data.get(field)
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"{field} must name an environment variable")

        names[field] = name

    resolved = {}
    for field in ("base_url_env", "model_env"):
        value = os.environ.get(names[field], "").strip()
        if not value or any(ord(char) < 32 for char in value):
            raise ValueError(f"Set a valid {names[field]} environment variable")

        resolved[field] = value

    base_url = resolved["base_url_env"].rstrip("/")
    try:
        url = urlsplit(base_url)
        valid_port = url.port is None or url.port > 0
    except ValueError as exc:
        raise ValueError("Invalid endpoint base URL") from exc

    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or not valid_port
        or url.username is not None
        or url.password is not None
        or url.query
        or url.fragment
        or " " in base_url
    ):
        raise ValueError("Endpoint URL must be HTTP(S), without credentials, query, or fragment")

    if url.scheme == "http" and url.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Remote endpoints require HTTPS; use a localhost SSH tunnel for HTTP")

    timeout = data.get("timeout_seconds", 60.0)
    temperature = data.get("temperature", 0.0)
    top_p = data.get("top_p", 1.0)
    for name, value in (
        ("timeout_seconds", timeout),
        ("temperature", temperature),
        ("top_p", top_p),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} must be finite and numeric")

    if timeout <= 0 or not 0 <= temperature <= 2:
        raise ValueError("Timeout must be positive and temperature must be between 0 and 2")

    if not 0 < top_p <= 1:
        raise ValueError("top_p must be between 0 (exclusive) and 1")

    identity = {}
    for name in ("model_revision", "quantization", "backend"):
        value = data.get(name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{name} must be a non-empty string")

        identity[name] = value

    max_tokens = data.get("max_tokens", 256)
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")

    config = EndpointConfig(
        base_url,
        resolved["model_env"],
        names["api_key_env"],
        float(timeout),
        max_tokens,
        float(temperature),
        float(top_p),
        **identity,
    )
    config.api_key()
    return config
