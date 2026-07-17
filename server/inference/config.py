"""Model configuration for 4-bit quantized inference."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = "TheBloke/Llama-2-7B-Chat-GPTQ"
    revision: str = "gptq-4bit-32g-actorder_True"
    device: str = "cuda"
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    # Dialogue must fit in P5R''s buffer — keep well under 256 chars
    max_response_chars: int = 200