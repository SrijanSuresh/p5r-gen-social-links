"""Tests for ModelConfig validation."""

import pytest
from inference.config import ModelConfig


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
