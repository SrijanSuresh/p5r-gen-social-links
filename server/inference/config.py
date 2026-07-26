"""Model configuration for llama-cpp-python inference."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    # Path to the GGUF model file.
    # Download with: huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF
    #                  --include "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    #                  --local-dir server/models
    model_path: str = field(
        default_factory=lambda: os.getenv(
            "MODEL_PATH",
            "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        )
    )

    # -1 = offload all layers to GPU (recommended for RTX 4060 8GB).
    # Set to e.g. 28 to keep 4 layers on CPU if VRAM is tight with P5R running.
    n_gpu_layers: int = -1

    # Context window. Our prompts are short (~300 tokens); 2048 is plenty.
    n_ctx: int = 2048

    # Generation parameters
    max_tokens: int = 80          # ~2-3 sentences of dialogue
    temperature: float = 0.8
    top_p: float = 0.9
    repeat_penalty: float = 1.1

    # Dialogue must fit in P5R's display area — hard cap before sending to game.
    max_response_chars: int = 200
