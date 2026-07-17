"""
FastAPI server: receives game state from the C# mod, returns LLM-generated dialogue.
The model is loaded lazily on first request to keep startup fast.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from inference.config import ModelConfig
from inference.pipeline import InferencePipeline

log = logging.getLogger(__name__)

_pipeline: InferencePipeline | None = None
_cfg = ModelConfig()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _pipeline
    log.info("Loading model %s …", _cfg.model_id)
    try:
        from inference.model_loader import load_model
        model, tokenizer = load_model(_cfg)
        _pipeline = InferencePipeline(model, tokenizer, _cfg)
        log.info("Model ready.")
    except Exception as exc:  # noqa: BLE001 — broad catch intentional at startup boundary
        log.warning("Model load failed (%s). /generate will return 503.", exc)
    yield
    # Cleanup on shutdown — release VRAM
    _pipeline = None


app = FastAPI(title="P5R Generative Social Links", version="0.2.0", lifespan=lifespan)


class GenerateRequest(BaseModel):
    confidant_id: int = Field(..., ge=0, le=25)
    rank: int = Field(..., ge=1, le=10)
    context: str = Field(..., max_length=1024)
    character_name: str = Field(..., max_length=64)


class GenerateResponse(BaseModel):
    text: str


@app.get("/health")
async def health() -> dict[str, str]:
    status = "ready" if _pipeline is not None else "model_not_loaded"
    return {"status": status}


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    text = _pipeline.generate(req.confidant_id, req.rank, req.context)
    return GenerateResponse(text=text)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="127.0.0.1", port=8765)