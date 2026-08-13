"""Tests for ModelConfig validation."""

import pytest
from inference.config import LlamaServerConfig, ModelConfig


def test_default_config_is_valid() -> None:
    cfg = ModelConfig()
    assert cfg.temperature == 0.8
    assert cfg.top_p == 0.9
    assert cfg.max_tokens == 80
    assert cfg.n_ctx == 2048


def test_temperature_zero_is_valid() -> None:
    cfg = ModelConfig(temperature=0.0)
    assert cfg.temperature == 0.0


def test_temperature_two_is_valid() -> None:
    cfg = ModelConfig(temperature=2.0)
    assert cfg.temperature == 2.0


def test_temperature_negative_raises() -> None:
    with pytest.raises(ValueError, match="temperature"):
        ModelConfig(temperature=-0.1)


def test_temperature_over_two_raises() -> None:
    with pytest.raises(ValueError, match="temperature"):
        ModelConfig(temperature=2.1)


def test_top_p_zero_raises() -> None:
    with pytest.raises(ValueError, match="top_p"):
        ModelConfig(top_p=0.0)


def test_top_p_one_is_valid() -> None:
    cfg = ModelConfig(top_p=1.0)
    assert cfg.top_p == 1.0


def test_max_tokens_zero_raises() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        ModelConfig(max_tokens=0)


def test_max_tokens_512_is_valid() -> None:
    cfg = ModelConfig(max_tokens=512)
    assert cfg.max_tokens == 512


def test_n_ctx_too_small_raises() -> None:
    with pytest.raises(ValueError, match="n_ctx"):
        ModelConfig(n_ctx=64)


# --- LlamaServerConfig -------------------------------------------------------


def test_llama_server_default_config_is_valid() -> None:
    cfg = LlamaServerConfig()
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 8080
    assert cfg.autostart is True


def test_llama_server_base_url_composes_host_and_port() -> None:
    cfg = LlamaServerConfig(host="127.0.0.1", port=9999)
    assert cfg.base_url == "http://127.0.0.1:9999"


def test_llama_server_port_does_not_collide_with_fastapi() -> None:
    # The FastAPI app binds 8765; the child must not default to the same port.
    assert LlamaServerConfig().port != 8765


def test_llama_server_port_zero_raises() -> None:
    with pytest.raises(ValueError, match="port"):
        LlamaServerConfig(port=0)


def test_llama_server_port_above_range_raises() -> None:
    with pytest.raises(ValueError, match="port"):
        LlamaServerConfig(port=70000)


def test_llama_server_negative_startup_timeout_raises() -> None:
    with pytest.raises(ValueError, match="startup_timeout_s"):
        LlamaServerConfig(startup_timeout_s=-1.0)


def test_llama_server_zero_request_timeout_raises() -> None:
    with pytest.raises(ValueError, match="request_timeout_s"):
        LlamaServerConfig(request_timeout_s=0.0)
