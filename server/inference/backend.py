"""
Which inference engine serves a request, and what it would take to add the second one.

The project has always intended two: the llama-server child process that works today, and
a Triton path built on the 4-bit dequantize-matmul kernel in ``kernels/``. This module is
the seam between them — it resolves a name to a backend and, for the one that is not
finished, says precisely what is missing instead of failing somewhere deeper with a
confusing error.

Naming the gap is the point. ``dequant_matmul`` is one kernel, and one kernel is not an
engine: see :data:`TRITON_MISSING_PIECES` for the distance between them.
"""

from __future__ import annotations

import os
from enum import Enum


class Backend(str, Enum):
    """An inference engine the server can be pointed at."""

    LLAMA_SERVER = "llama"
    TRITON = "triton"

    @classmethod
    def names(cls) -> list[str]:
        return [member.value for member in cls]


DEFAULT_BACKEND = Backend.LLAMA_SERVER


class UnknownBackendError(ValueError):
    """The configured backend name is not one this server knows."""


class BackendNotBuiltError(NotImplementedError):
    """
    The requested backend is a real plan with real code behind it, and is not finished.

    Separate from :class:`UnknownBackendError` because the remedies differ: a typo in
    ``INFERENCE_BACKEND`` is fixed by the person who set it, and this one is fixed by
    writing the missing pieces.
    """


TRITON_MISSING_PIECES: tuple[str, ...] = (
    "a GGUF or safetensors weight loader that produces the packed int4 layout "
    "kernels/reference.py documents",
    "a tokenizer and detokenizer for Llama-3.1",
    "the rest of the transformer block: RMSNorm, RoPE, attention, SwiGLU",
    "a KV cache, without which every token re-reads the whole prompt",
    "sampling (temperature, top-p, repetition penalty) to match what the prompt "
    "builder already asks for",
    "a chat-completions shim, so the pipeline's ChatCompletionBackend protocol is "
    "satisfied without the pipeline changing",
)
"""
What stands between the kernel and a working Triton backend.

Written down rather than left implicit because the kernel's existence makes the backend
look closer than it is. A quantized matmul is the arithmetic inside one linear layer;
everything above is the model.
"""


def resolve(name: str | None = None) -> Backend:
    """
    Parse a backend name, defaulting to the one that works.

    Reads ``INFERENCE_BACKEND`` when ``name`` is omitted. Case and surrounding whitespace
    are forgiven because this arrives from a shell.

    Raises:
        UnknownBackendError: the name is not a member of :class:`Backend`.
    """
    raw = (name if name is not None else os.getenv("INFERENCE_BACKEND", "")).strip()
    if not raw:
        return DEFAULT_BACKEND

    try:
        return Backend(raw.lower())
    except ValueError as exc:
        raise UnknownBackendError(
            f"unknown INFERENCE_BACKEND {raw!r}; expected one of {', '.join(Backend.names())}"
        ) from exc


def require_available(backend: Backend) -> None:
    """
    Fail now, with a complete explanation, rather than part-way through a request.

    Raises:
        BackendNotBuiltError: the backend is planned but unfinished.
    """
    if backend is Backend.TRITON:
        raise BackendNotBuiltError(
            "the Triton backend is not built yet. The 4-bit dequantize-matmul kernel "
            "exists and is checked against kernels/reference.py, but a kernel is not an "
            "engine. Still needed:\n"
            + "\n".join(f"  - {piece}" for piece in TRITON_MISSING_PIECES)
            + f"\n\nRun with INFERENCE_BACKEND={Backend.LLAMA_SERVER.value} "
            "(the default) to use llama-server."
        )
