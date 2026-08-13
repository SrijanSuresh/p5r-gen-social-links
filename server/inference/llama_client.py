"""
HTTP client for a llama-server instance.

Replaces the in-process `llama_cpp.Llama` object. The method surface is deliberately
identical (`create_chat_completion`, returning the OpenAI response shape), because
that schema is exactly what llama-server exposes at /v1/chat/completions — the
binding was mirroring it locally. Keeping the seam means InferencePipeline, and
therefore build_prompt and clean_response, never learn that inference moved out of
process. See learning.md Ch. 62.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

import httpx

from .config import LlamaServerConfig, ModelConfig


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class LlamaServerError(RuntimeError):
    """Base class for llama-server transport and protocol failures."""


class LlamaServerUnavailableError(LlamaServerError):
    """The server could not be reached, or did not answer within the timeout."""


class LlamaServerResponseError(LlamaServerError):
    """The server answered, but with an error status or an unusable body."""


class LlamaServerClient:
    """
    Talks to llama-server over loopback HTTP.

    The client owns a persistent httpx.Client so successive generations reuse one TCP
    connection; on loopback the handshake is cheap, but reconnecting per request also
    burns an ephemeral port each time, which matters when a hang-out fires a request
    every few seconds for an entire session.
    """

    def __init__(
        self,
        server_cfg: LlamaServerConfig,
        model_cfg: ModelConfig,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """
        `transport` overrides the network layer. Tests pass httpx.MockTransport to
        exercise the real request-building and response-validation code against
        synthetic replies, with no socket and no model.
        """
        self._cfg = server_cfg
        self._model_cfg = model_cfg
        self._http = httpx.Client(
            base_url=server_cfg.base_url,
            timeout=server_cfg.request_timeout_s,
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "LlamaServerClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def is_healthy(self) -> bool:
        """
        True once the child process is up and the weights are loaded.

        llama-server returns 503 while still loading and 200 when ready, so this
        distinguishes "starting" from "serving" without a separate readiness probe.
        Never raises: callers poll this in a loop and treat any failure as not-ready.
        """
        try:
            response = self._http.get("/health", timeout=2.0)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    def create_chat_completion(
        self,
        messages: list[ChatMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repeat_penalty: float | None = None,
    ) -> dict[str, Any]:
        """
        Mirrors llama_cpp.Llama.create_chat_completion.

        Unset arguments fall back to ModelConfig rather than to llama-server's own
        defaults, so generation parameters stay owned by this repo regardless of how
        the child process was launched.
        """
        payload: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens if max_tokens is not None else self._model_cfg.max_tokens,
            "temperature": (
                temperature if temperature is not None else self._model_cfg.temperature
            ),
            "top_p": top_p if top_p is not None else self._model_cfg.top_p,
            # repeat_penalty is a llama.cpp extension, not part of the OpenAI schema.
            # llama-server accepts it on /v1/chat/completions and ignores it if the
            # build ever drops support, so sending it is safe either way.
            "repeat_penalty": (
                repeat_penalty
                if repeat_penalty is not None
                else self._model_cfg.repeat_penalty
            ),
            "stream": False,
        }

        try:
            response = self._http.post("/v1/chat/completions", json=payload)
        except httpx.TimeoutException as exc:
            raise LlamaServerUnavailableError(
                f"llama-server did not respond within "
                f"{self._cfg.request_timeout_s}s at {self._cfg.base_url}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LlamaServerUnavailableError(
                f"cannot reach llama-server at {self._cfg.base_url}: {exc}"
            ) from exc

        if response.status_code != 200:
            raise LlamaServerResponseError(
                f"llama-server returned HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )

        try:
            body: dict[str, Any] = response.json()
        except ValueError as exc:
            raise LlamaServerResponseError(
                f"llama-server returned a non-JSON body: {response.text[:200]}"
            ) from exc

        # Validate the path InferencePipeline indexes into, so a schema change surfaces
        # here with context rather than as a bare KeyError deep in the pipeline.
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LlamaServerResponseError(
                f"llama-server response has no choices: {str(body)[:200]}"
            )
        message = choices[0].get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise LlamaServerResponseError(
                f"llama-server choice has no message content: {str(choices[0])[:200]}"
            )

        return body
