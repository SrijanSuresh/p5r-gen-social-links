"""Inference pipeline: build prompt → generate → postprocess."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import ModelConfig
from social_link.prompt_builder import build_prompt
from inference.postprocess import clean_response

if TYPE_CHECKING:
    import llama_cpp


class InferencePipeline:
    """Wraps a llama-cpp Llama instance into a single generate() call."""

    def __init__(self, model: "llama_cpp.Llama", cfg: ModelConfig) -> None:
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
