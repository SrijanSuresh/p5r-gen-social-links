"""Inference pipeline: build prompt → generate → postprocess."""

from __future__ import annotations

from typing import Any, Protocol

from .config import ModelConfig
from social_link.prompt_builder import build_prompt
from inference.postprocess import clean_response


class ChatCompletionBackend(Protocol):
    """
    Anything that can answer an OpenAI-shaped chat completion.

    Structural rather than nominal on purpose: this was `llama_cpp.Llama`, is now
    `LlamaServerClient`, and is a MagicMock under test. All three satisfy it without
    inheriting from anything, which is what let inference move out of process without
    touching this file's logic. See learning.md Ch. 62.
    """

    def create_chat_completion(
        self,
        messages: list[Any],
        max_tokens: int | None = ...,
        temperature: float | None = ...,
        top_p: float | None = ...,
        repeat_penalty: float | None = ...,
    ) -> dict[str, Any]: ...


class InferencePipeline:
    """Wraps a chat-completion backend into a single generate() call."""

    def __init__(self, model: ChatCompletionBackend, cfg: ModelConfig) -> None:
        self._model = model
        self._cfg   = cfg

    def generate(self, confidant_id: int, rank: int, context: str) -> str:
        system_prompt, user_prompt = build_prompt(confidant_id, rank, context)

        response = self._model.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=self._cfg.max_tokens,
            temperature=self._cfg.temperature,
            top_p=self._cfg.top_p,
            repeat_penalty=self._cfg.repeat_penalty,
        )

        raw = response["choices"][0]["message"]["content"].strip()
        return clean_response(raw, self._cfg.max_response_chars)
