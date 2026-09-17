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


@pytest.mark.asyncio
async def test_generate_response_contains_session_id(client_with_mock: AsyncClient) -> None:
    """Each /generate response must include a monotonic session_id integer."""
    r = await client_with_mock.post("/generate", json=RYUJI_REQUEST)
    assert r.status_code == 200
    body = r.json()
    assert "session_id" in body
    assert isinstance(body["session_id"], int)


@pytest.mark.asyncio
async def test_generate_session_id_increments(client_with_mock: AsyncClient) -> None:
    """session_id should be strictly greater on the second call than the first."""
    r1 = await client_with_mock.post("/generate", json=RYUJI_REQUEST)
    r2 = await client_with_mock.post("/generate", json=RYUJI_REQUEST)
    assert r2.json()["session_id"] > r1.json()["session_id"]


# --- the addressee reaches the pipeline, and absence is not an error ------------
#
# The field is the last link in a chain that starts in a hex dump: the archive names a
# record's speaker, the turn order turns that into a listener, and only here does it
# become something the model can read. A silent drop anywhere along it looks exactly
# like the bug it fixes (Ch. 81).


@pytest.mark.asyncio
async def test_generate_forwards_the_addressee_to_the_pipeline(
    client_with_mock: AsyncClient, mock_pipeline
) -> None:
    body = {**RYUJI_REQUEST, "addressee": "Girl's Father"}
    r = await client_with_mock.post("/generate", json=body)

    assert r.status_code == 200
    assert "Girl's Father" in mock_pipeline.generate.call_args.args


@pytest.mark.asyncio
async def test_generate_defaults_the_addressee_to_empty(
    client_with_mock: AsyncClient, mock_pipeline
) -> None:
    # An older mod build does not send the field at all. It has to keep working, and the
    # prompt has to omit the claim rather than default it to Joker.
    r = await client_with_mock.post("/generate", json=RYUJI_REQUEST)

    assert r.status_code == 200
    assert mock_pipeline.generate.call_args.args[-1] == ""


@pytest.mark.asyncio
async def test_generate_rejects_an_implausibly_long_addressee(
    client_with_mock: AsyncClient,
) -> None:
    bad = {**RYUJI_REQUEST, "addressee": "x" * 65}
    r = await client_with_mock.post("/generate", json=bad)
    assert r.status_code == 422
