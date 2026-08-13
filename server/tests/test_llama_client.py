"""
Tests for the llama-server HTTP client.

Every test drives the real request-building and validation code through an
httpx.MockTransport, so no socket is opened and no model is loaded — these run
identically on CI, which has neither a GPU nor the vendored binary.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from inference.config import LlamaServerConfig, ModelConfig
from inference.llama_client import (
    LlamaServerClient,
    LlamaServerResponseError,
    LlamaServerUnavailableError,
)

MESSAGES: list[Any] = [
    {"role": "system", "content": "You are Ryuji."},
    {"role": "user", "content": "Say hi."},
]


def _completion_body(text: str = "Yo, what's up?") -> dict[str, Any]:
    """A minimal well-formed OpenAI chat-completion response."""
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def _client(handler: Any) -> LlamaServerClient:
    return LlamaServerClient(
        LlamaServerConfig(), ModelConfig(), transport=httpx.MockTransport(handler)
    )


# --- happy path --------------------------------------------------------------


def test_create_chat_completion_returns_body() -> None:
    client = _client(lambda req: httpx.Response(200, json=_completion_body()))
    body = client.create_chat_completion(MESSAGES)
    assert body["choices"][0]["message"]["content"] == "Yo, what's up?"


def test_request_targets_openai_chat_endpoint() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["method"] = request.method
        return httpx.Response(200, json=_completion_body())

    _client(handler).create_chat_completion(MESSAGES)
    assert seen["path"] == "/v1/chat/completions"
    assert seen["method"] == "POST"


def test_defaults_come_from_model_config_not_the_server() -> None:
    """Generation parameters stay owned by this repo, not by the child's launch flags."""
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=_completion_body())

    cfg = ModelConfig()
    _client(handler).create_chat_completion(MESSAGES)
    assert seen["max_tokens"] == cfg.max_tokens
    assert seen["temperature"] == cfg.temperature
    assert seen["top_p"] == cfg.top_p
    assert seen["repeat_penalty"] == cfg.repeat_penalty


def test_explicit_arguments_override_config_defaults() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=_completion_body())

    _client(handler).create_chat_completion(
        MESSAGES, max_tokens=7, temperature=0.1, top_p=0.5, repeat_penalty=1.9
    )
    assert seen["max_tokens"] == 7
    assert seen["temperature"] == 0.1
    assert seen["top_p"] == 0.5
    assert seen["repeat_penalty"] == 1.9


def test_streaming_is_disabled() -> None:
    """The pipeline indexes the full body; a streamed reply would break that."""
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=_completion_body())

    _client(handler).create_chat_completion(MESSAGES)
    assert seen["stream"] is False


def test_messages_are_forwarded_unchanged() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=_completion_body())

    _client(handler).create_chat_completion(MESSAGES)
    assert seen["messages"] == MESSAGES


# --- transport failures ------------------------------------------------------


def test_connect_error_raises_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(LlamaServerUnavailableError, match="cannot reach"):
        _client(handler).create_chat_completion(MESSAGES)


def test_timeout_raises_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow")

    with pytest.raises(LlamaServerUnavailableError, match="did not respond"):
        _client(handler).create_chat_completion(MESSAGES)


# --- protocol failures -------------------------------------------------------


def test_non_200_raises_response_error() -> None:
    client = _client(lambda req: httpx.Response(500, text="internal error"))
    with pytest.raises(LlamaServerResponseError, match="HTTP 500"):
        client.create_chat_completion(MESSAGES)


def test_non_json_body_raises_response_error() -> None:
    client = _client(lambda req: httpx.Response(200, text="<html>nope</html>"))
    with pytest.raises(LlamaServerResponseError, match="non-JSON"):
        client.create_chat_completion(MESSAGES)


def test_empty_choices_raises_response_error() -> None:
    client = _client(lambda req: httpx.Response(200, json={"choices": []}))
    with pytest.raises(LlamaServerResponseError, match="no choices"):
        client.create_chat_completion(MESSAGES)


def test_missing_message_content_raises_response_error() -> None:
    client = _client(lambda req: httpx.Response(200, json={"choices": [{"message": {}}]}))
    with pytest.raises(LlamaServerResponseError, match="no message content"):
        client.create_chat_completion(MESSAGES)


# --- health ------------------------------------------------------------------


def test_is_healthy_true_on_200() -> None:
    assert _client(lambda req: httpx.Response(200, json={"status": "ok"})).is_healthy()


def test_is_healthy_false_while_model_loading() -> None:
    """llama-server answers 503 until the weights finish loading."""
    assert not _client(lambda req: httpx.Response(503)).is_healthy()


def test_is_healthy_false_when_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    assert not _client(handler).is_healthy()


def test_is_healthy_never_raises_on_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    assert not _client(handler).is_healthy()


def test_context_manager_closes_client() -> None:
    with _client(lambda req: httpx.Response(200, json=_completion_body())) as client:
        assert client.create_chat_completion(MESSAGES)["choices"]
