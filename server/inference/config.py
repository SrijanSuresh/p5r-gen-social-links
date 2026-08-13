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

    def __post_init__(self) -> None:
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError(f"temperature must be in [0, 2], got {self.temperature}")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if not (1 <= self.max_tokens <= 512):
            raise ValueError(f"max_tokens must be in [1, 512], got {self.max_tokens}")
        if not (128 <= self.n_ctx <= 32768):
            raise ValueError(f"n_ctx must be in [128, 32768], got {self.n_ctx}")


@dataclass(frozen=True)
class LlamaServerConfig:
    """
    Location and launch parameters for the llama-server child process.

    Inference runs in an upstream ggml-org `llama-server.exe` rather than through the
    llama-cpp-python binding. The binding is a C extension pinned to a CPython ABI
    (no cp313 wheel exists, and PyPI ships source only), so using it would require the
    CUDA Toolkit and MSVC on every machine running the mod. See learning.md Ch. 62.
    """

    host: str = field(default_factory=lambda: os.getenv("LLAMA_HOST", "127.0.0.1"))

    # Distinct from the FastAPI port (8765) — both listen on loopback simultaneously.
    port: int = field(default_factory=lambda: int(os.getenv("LLAMA_PORT", "8080")))

    # Path to llama-server.exe, relative to server/. Populated by
    # scripts/fetch-llama-server.ps1, which extracts the release zips into vendor/.
    binary_path: str = field(
        default_factory=lambda: os.getenv("LLAMA_BINARY", "vendor/llama-server.exe")
    )

    # Seconds to wait for the child to report /health ok. A cold start reads ~4.9 GB
    # from disk and uploads it to VRAM; on a warm page cache this is a few seconds,
    # on a cold one it can exceed a minute.
    startup_timeout_s: float = 180.0

    # How often to poll /health while waiting for startup.
    startup_poll_s: float = 1.0

    # Per-request timeout for generation calls.
    request_timeout_s: float = 60.0

    # Set false to attach to an already-running llama-server instead of spawning one —
    # useful when iterating on the FastAPI layer without paying model load each restart.
    autostart: bool = field(
        default_factory=lambda: os.getenv("LLAMA_AUTOSTART", "1") != "0"
    )

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def __post_init__(self) -> None:
        if not (1 <= self.port <= 65535):
            raise ValueError(f"port must be in [1, 65535], got {self.port}")
        if self.startup_timeout_s <= 0:
            raise ValueError(
                f"startup_timeout_s must be positive, got {self.startup_timeout_s}"
            )
        if self.request_timeout_s <= 0:
            raise ValueError(
                f"request_timeout_s must be positive, got {self.request_timeout_s}"
            )
