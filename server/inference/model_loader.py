"""Loads a GGUF model via llama-cpp-python with full GPU offload."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .config import ModelConfig

log = logging.getLogger(__name__)


def load_model(cfg: ModelConfig) -> "llama_cpp.Llama":  # type: ignore[name-defined]
    """
    Load a GGUF quantized model from a local file path.

    Requires:
        - llama-cpp-python installed with CUDA support:
            pip install llama-cpp-python \\
              --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122
        - The GGUF file at cfg.model_path (relative to server/ directory).
          Download with:
            huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \\
              --include "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" \\
              --local-dir server/models
    """
    from llama_cpp import Llama  # noqa: PLC0415

    # Resolve path relative to this file's parent (server/)
    model_path = Path(__file__).parent.parent / cfg.model_path
    if not model_path.exists():
        raise FileNotFoundError(
            f"GGUF model not found at {model_path}. "
            "Run: huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF "
            "--include 'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf' --local-dir server/models"
        )

    log.info("Loading %s (n_gpu_layers=%s)…", model_path.name, cfg.n_gpu_layers)
    model = Llama(
        model_path=str(model_path),
        n_gpu_layers=cfg.n_gpu_layers,   # -1 = all layers on GPU
        n_ctx=cfg.n_ctx,
        verbose=False,                    # suppress llama.cpp token-by-token logs
    )
    log.info("Model loaded.")
    return model
