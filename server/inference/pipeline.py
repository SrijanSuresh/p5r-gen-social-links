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


def _token_budget(max_chars: int, ceiling: int) -> int:
    """
    Tokens worth generating for a line of ``max_chars`` characters.

    Decode dominates the round trip. At 20 of 32 layers on the GPU, generating a fixed 32
    tokens costs roughly two seconds regardless of how short the destination is — and a
    thirty-character bubble needs about ten. The rest was generated and then thrown away
    by the character clip.

    English runs near four characters per token, so the budget is chars/3 with a few
    tokens of headroom: enough that the model reaches a sentence end on its own rather
    than being cut off at the limit, which is the failure that produces fragments.
    """
    return max(8, min(ceiling, max_chars // 3 + 6))


class InferencePipeline:
    """Wraps a chat-completion backend into a single generate() call."""

    def __init__(self, model: ChatCompletionBackend, cfg: ModelConfig) -> None:
        self._model = model
        self._cfg   = cfg

    def generate(
        self,
        confidant_id: int,
        rank: int,
        context: str,
        max_chars: int | None = None,
    ) -> str:
        """
        Produce one line, clipped to what the destination can actually display.

        ``max_chars`` is the capacity of the specific message record the mod is about to
        overwrite, and it varies per line: a one-row record holds about 30 characters, a
        two-row one about 75. A fixed budget therefore cannot be right. Generating 53
        characters for a 30-character record produced "You're finally here, I've been"
        on screen, which reads as a bug rather than as a short line.

        None keeps the configured default, for callers that have no record in hand.
        """
        budget = self._cfg.max_response_chars if max_chars is None else max_chars
        system_prompt, user_prompt = build_prompt(confidant_id, rank, context, budget)

        response = self._model.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=_token_budget(budget, self._cfg.max_tokens),
            temperature=self._cfg.temperature,
            top_p=self._cfg.top_p,
            repeat_penalty=self._cfg.repeat_penalty,
        )

        raw = response["choices"][0]["message"]["content"].strip()
        return clean_response(raw, budget)
