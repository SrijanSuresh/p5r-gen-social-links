"""
Proves InferencePipeline drives the real LlamaServerClient.

The other suites test the client and the pipeline separately; this one wires the
genuine article to the genuine article, with only the socket replaced. It is the test
that would have caught the port breaking the contract between them.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from inference.config import LlamaServerConfig, ModelConfig
from inference.llama_client import LlamaServerClient, LlamaServerUnavailableError
from inference.pipeline import InferencePipeline

RYUJI_ID = 8


def _pipeline(handler: Any) -> tuple[InferencePipeline, ModelConfig]:
    cfg = ModelConfig()
    client = LlamaServerClient(
        LlamaServerConfig(), cfg, transport=httpx.MockTransport(handler)
    )
    return InferencePipeline(client, cfg), cfg


def test_generate_returns_cleaned_text() -> None:
    pipeline, _ = _pipeline(
        lambda req: httpx.Response(
            200,
            json={"choices": [{"message": {"content": "  Yo, let's hit the gym!  "}}]},
        )
    )
    assert pipeline.generate(RYUJI_ID, 4, "at the gym") == "Yo, let's hit the gym!"


def test_generate_sends_system_and_user_roles() -> None:
    """build_prompt's two-layer prompt must survive the trip over HTTP."""
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    pipeline, _ = _pipeline(handler)
    pipeline.generate(RYUJI_ID, 4, "at the gym")

    roles = [message["role"] for message in seen["messages"]]
    assert roles == ["system", "user"]
    assert all(message["content"] for message in seen["messages"])


def test_generate_applies_configured_sampling() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    pipeline, cfg = _pipeline(handler)
    pipeline.generate(RYUJI_ID, 4, "at the gym")

    assert seen["temperature"] == cfg.temperature
    assert seen["max_tokens"] == cfg.max_tokens
    assert seen["repeat_penalty"] == cfg.repeat_penalty


def test_generate_truncates_to_display_budget() -> None:
    """P5R's bubble is finite; the cap must still apply across the new transport."""
    cfg = ModelConfig()
    long_text = "Dude. " * 200
    pipeline, _ = _pipeline(
        lambda req: httpx.Response(200, json={"choices": [{"message": {"content": long_text}}]})
    )
    assert len(pipeline.generate(RYUJI_ID, 4, "gym")) <= cfg.max_response_chars


def test_backend_failure_propagates_as_llama_server_error() -> None:
    """The endpoint maps this to 503 — it must not surface as a generic error."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    pipeline, _ = _pipeline(handler)
    with pytest.raises(LlamaServerUnavailableError):
        pipeline.generate(RYUJI_ID, 4, "gym")


# --- per-record length budget -------------------------------------------------


class _CapturingBackend:
    """Records the messages it was handed, and returns a fixed over-long line."""

    def __init__(self, reply: str = "A" * 200) -> None:
        self.messages: list[dict[str, str]] = []
        self._reply = reply

    def create_chat_completion(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.messages = messages
        return {"choices": [{"message": {"content": self._reply}}]}


def test_generate_clips_to_the_record_budget() -> None:
    from inference.config import ModelConfig
    from inference.pipeline import InferencePipeline

    backend = _CapturingBackend()
    pipeline = InferencePipeline(backend, ModelConfig())
    assert len(pipeline.generate(8, 4, "at the gym", max_chars=30)) <= 30


def test_generate_falls_back_to_configured_budget() -> None:
    """None means "no record in hand", not "no limit"."""
    from inference.config import ModelConfig
    from inference.pipeline import InferencePipeline

    cfg = ModelConfig()
    backend = _CapturingBackend()
    pipeline = InferencePipeline(backend, cfg)
    assert len(pipeline.generate(8, 4, "at the gym")) <= cfg.max_response_chars


def test_budget_reaches_the_prompt_not_only_the_clip() -> None:
    """
    Clipping alone would leave the model writing to a length it was never told about,
    so every short record would end mid-word instead of ending early.
    """
    from inference.config import ModelConfig
    from inference.pipeline import InferencePipeline

    backend = _CapturingBackend(reply="Yo.")
    InferencePipeline(backend, ModelConfig()).generate(8, 4, "gym", max_chars=30)
    # The user half, not the system half: the system prompt is the cache prefix and has
    # to stay identical between requests.
    assert "30 characters" in backend.messages[1]["content"]
