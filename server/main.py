"""FastAPI server: receives game state, returns LLM-generated dialogue."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from inference.config import LlamaServerConfig, ModelConfig
from inference.llama_client import LlamaServerClient, LlamaServerError
from inference.llama_process import LlamaServerProcess, LlamaServerStartupError
from inference.pipeline import InferencePipeline
from inference.queue import InferenceQueue
from social_link.arcana import get_confidant
from social_link.mock_responses import get_mock_response

log = logging.getLogger(__name__)

# Sentinel object used when MOCK_LLM=1 is set — avoids loading any model.
_MOCK = object()

_pipeline: InferencePipeline | object | None = None
_queue      = InferenceQueue()
_cfg        = ModelConfig()
_server_cfg = LlamaServerConfig()
_process: LlamaServerProcess | None = None
_client: LlamaServerClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Bring up the inference backend, then tear it down.

    Inference runs in a llama-server child process rather than an in-process binding,
    so a failure here degrades the API to 503 instead of killing it. See
    learning.md Ch. 62.
    """
    global _pipeline, _process, _client
    if os.getenv("MOCK_LLM"):
        log.info("MOCK_LLM=1 — skipping model load, returning canned responses.")
        _pipeline = _MOCK
    else:
        try:
            if _server_cfg.autostart:
                _process = LlamaServerProcess(_server_cfg, _cfg)
                _process.start()  # returns only once the weights are resident
            else:
                log.info(
                    "LLAMA_AUTOSTART=0 — attaching to existing llama-server at %s",
                    _server_cfg.base_url,
                )
            _client   = LlamaServerClient(_server_cfg, _cfg)
            _pipeline = InferencePipeline(_client, _cfg)
            log.info("Inference backend ready at %s", _server_cfg.base_url)
        except LlamaServerStartupError as exc:
            # Narrow: a backend that will not start is expected and recoverable, and
            # the message already carries the child's exit code and log tail.
            log.warning("llama-server unavailable (%s). /generate will return 503.", exc)

    yield

    if _client is not None:
        _client.close()
        _client = None
    if _process is not None:
        _process.stop()
        _process = None
    _pipeline = None


app = FastAPI(title="P5R Generative Social Links", version="0.3.0", lifespan=lifespan)


class GenerateRequest(BaseModel):
    confidant_id: int = Field(..., ge=0, le=50)
    rank: int = Field(..., ge=1, le=10)
    context: str = Field(..., max_length=1024)
    character_name: str = Field(..., max_length=64)


class GenerateResponse(BaseModel):
    text: str
    session_id: int


@app.get("/health")
async def health() -> dict[str, str]:
    if _pipeline is _MOCK:
        return {"status": "mock"}
    status = "ready" if _pipeline is not None else "model_not_loaded"
    return {"status": status}


@app.get("/ready")
async def ready() -> dict[str, bool]:
    """Returns {"ready": true} once the model is loaded and inference can begin."""
    is_ready = _pipeline is not None
    return {"ready": is_ready}


@app.get("/stats")
async def stats() -> dict[str, object]:
    return {
        "total_requests":    _queue.total_requests,
        "total_drops":       _queue.total_drops,
        "total_completions": _queue.total_completions,
        "avg_latency_ms":    _queue.avg_latency_ms,
        "model_loaded":      _pipeline is not None and _pipeline is not _MOCK,
    }


@app.post("/clear-stats")
async def clear_stats() -> dict[str, str]:
    """Reset inference counters — useful between test sessions without restarting."""
    _queue.clear_stats()
    return {"result": "cleared"}


@app.get("/model-info")
async def model_info() -> dict[str, object]:
    is_mock  = _pipeline is _MOCK
    is_real  = _pipeline is not None and not is_mock
    info: dict[str, object] = {
        "mode":         "mock" if is_mock else ("real" if is_real else "not_loaded"),
        "model_path":   _cfg.model_path,
        "n_gpu_layers": _cfg.n_gpu_layers,
        "n_ctx":        _cfg.n_ctx,
        "max_tokens":   _cfg.max_tokens,
        "temperature":  _cfg.temperature,
        # Inference runs out of process; surfacing where and whether we spawned it
        # makes a misconfigured LLAMA_AUTOSTART=0 diagnosable from the mod's log.
        "backend":         "llama-server",
        "backend_url":     _server_cfg.base_url,
        "backend_managed": _process is not None and _process.is_running,
    }
    return info


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if _pipeline is _MOCK:
        mock_text = get_mock_response(req.confidant_id, req.rank)
        return GenerateResponse(text=mock_text, session_id=_queue.total_requests)

    async def _run() -> str:
        # generate() blocks for the whole round-trip (~2s). Running it on the event
        # loop would stall /health and /stats for that entire window, which the mod
        # polls — so it goes to a worker thread.
        return await asyncio.to_thread(
            _pipeline.generate,  # type: ignore[union-attr]
            req.confidant_id,
            req.rank,
            req.context,
        )

    try:
        result = await _queue.run_if_idle(_run)
    except LlamaServerError as exc:
        # The child died or stopped answering after startup. 503 tells the mod to
        # fall back to scripted dialogue rather than retry a broken backend.
        raise HTTPException(status_code=503, detail=f"Inference backend: {exc}") from exc

    if result is None:
        raise HTTPException(status_code=429, detail="Inference already in-flight.")
    return GenerateResponse(text=result, session_id=_queue.total_completions)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="127.0.0.1", port=8765)