"""Integration tests for the /generate endpoint using shared fixtures."""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from tests.conftest import RYUJI_REQUEST


@pytest.mark.asyncio
async def test_generate_returns_text(client_with_mock: AsyncClient) -> None:
    r = await client_with_mock.post("/generate", json=RYUJI_REQUEST)
    assert r.status_code == 200
    assert len(r.json()["text"]) > 0


@pytest.mark.asyncio
async def test_generate_text_is_string(client_with_mock: AsyncClient) -> None:
    r = await client_with_mock.post("/generate", json=RYUJI_REQUEST)
    assert isinstance(r.json()["text"], str)


@pytest.mark.asyncio
async def test_generate_503_without_model(client_no_model: AsyncClient) -> None:
    r = await client_no_model.post("/generate", json=RYUJI_REQUEST)
    assert r.status_code == 503


@pytest.mark.asyncio
async def test_generate_rejects_invalid_confidant_id(client_with_mock: AsyncClient) -> None:
    bad = {**RYUJI_REQUEST, "confidant_id": -1}
    r = await client_with_mock.post("/generate", json=bad)
    assert r.status_code == 422  # Pydantic validation error


@pytest.mark.asyncio
async def test_generate_rejects_rank_out_of_range(client_with_mock: AsyncClient) -> None:
    bad = {**RYUJI_REQUEST, "rank": 11}
    r = await client_with_mock.post("/generate", json=bad)
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_generate_rejects_empty_context(client_with_mock: AsyncClient) -> None:
    bad = {**RYUJI_REQUEST, "context": ""}
    # Empty string is still ≥ 0 chars — should succeed (context is not required to be non-empty)
    r = await client_with_mock.post("/generate", json=bad)
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_generate_rejects_context_too_long(client_with_mock: AsyncClient) -> None:
    bad = {**RYUJI_REQUEST, "context": "x" * 1025}
    r = await client_with_mock.post("/generate", json=bad)
    assert r.status_code == 422
