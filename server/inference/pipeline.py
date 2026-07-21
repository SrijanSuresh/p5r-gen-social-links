"""Inference pipeline: tokenize → generate → decode → trim."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import ModelConfig
from social_link.prompt_builder import build_prompt

if TYPE_CHECKING:
    import torch
    from transformers import AutoTokenizer  # type: ignore[import-untyped]


class InferencePipeline:
    """Wraps model + tokenizer into a single generate() call."""

    def __init__(self, model: object, tokenizer: "AutoTokenizer", cfg: ModelConfig) -> None:
        self._model     = model
        self._tokenizer = tokenizer
        self._cfg       = cfg

    def generate(self, confidant_id: int, rank: int, context: str) -> str:
        import torch  # noqa: PLC0415 — lazy import; torch only needed at inference time

        system_prompt, user_prompt = build_prompt(confidant_id, rank, context)

        full_prompt = (
            f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
        )

        inputs = self._tokenizer(full_prompt, return_tensors="pt").to(self._cfg.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._cfg.max_new_tokens,
                temperature=self._cfg.temperature,
                top_p=self._cfg.top_p,
                repetition_penalty=self._cfg.repetition_penalty,
                do_sample=True,
            )

        # Decode only the newly generated tokens (skip the prompt)
        new_ids  = output_ids[0, inputs["input_ids"].shape[1]:]
        raw_text = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        return raw_text[: self._cfg.max_response_chars]