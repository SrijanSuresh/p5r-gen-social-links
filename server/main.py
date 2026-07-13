"""FastAPI server: receives game state from C# mod, returns LLM-generated dialogue."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

app = FastAPI(title="P5R Generative Social Links Server", version="0.1.0")


class GenerateRequest(BaseModel):
    confidant_id: int = Field(..., ge=0, le=25, description="Arcana index 0-25")
    rank: int = Field(..., ge=1, le=10)
    context: str = Field(..., max_length=1024)
    character_name: str = Field(..., max_length=64)


class GenerateResponse(BaseModel):
    text: str


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    # Placeholder — real inference pipeline wired in Chapter 6
    raise HTTPException(status_code=501, detail="Inference not yet implemented")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8765)
