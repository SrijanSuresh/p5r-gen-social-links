"""Loads a GPTQ 4-bit quantized causal LM and its tokenizer."""

from __future__ import annotations
from typing import TYPE_CHECKING

from transformers import AutoTokenizer, GenerationConfig  # type: ignore[import-untyped]

from .config import ModelConfig

if TYPE_CHECKING:
    from auto_gptq import AutoGPTQForCausalLM  # type: ignore[import-untyped]


def load_model(cfg: ModelConfig) -> tuple["AutoGPTQForCausalLM", "AutoTokenizer"]:
    """
    Load a GPTQ-quantized model from HuggingFace Hub.

    Requires:
        - CUDA-capable GPU with enough VRAM (4-bit Llama-7B needs ~4 GB)
        - auto-gptq installed: pip install auto-gptq
    """
    from auto_gptq import AutoGPTQForCausalLM  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)

    model = AutoGPTQForCausalLM.from_quantized(
        cfg.model_id,
        revision=cfg.revision,
        use_safetensors=True,
        trust_remote_code=False,
        device=cfg.device,
        use_triton=cfg.use_triton,   # False on Windows; True only on Linux with triton installed
        inject_fused_attention=True,
        inject_fused_mlp=True,
    )
    model.eval()
    return model, tokenizer