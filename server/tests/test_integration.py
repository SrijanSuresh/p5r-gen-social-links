"""End-to-end integration tests using mock pipeline — no GPU required."""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from tests.conftest import RYUJI_REQUEST


@pytest.mark.asyncio
async def test_full_round_trip_ryuji(client_with_mock: AsyncClient) -> None:
    """Full round-trip: POST /generate for Ryuji returns in-character text."""
    r = await client_with_mock.post("/generate", json=RYUJI_REQUEST)
    assert r.status_code == 200
    body = r.json()
    assert "text" in body
    assert len(body["text"]) > 10


@pytest.mark.asyncio
async def test_stats_after_round_trip(client_with_mock: AsyncClient) -> None:
    """Stats endpoint reflects completions after a successful generate."""
    await client_with_mock.post("/generate", json=RYUJI_REQUEST)
    r = await client_with_mock.get("/stats")
    body = r.json()
    assert body["total_completions"] >= 1
    assert body["avg_latency_ms"] is not None


@pytest.mark.asyncio
async def test_generate_then_health(client_with_mock: AsyncClient) -> None:
    """Server remains healthy after a generate call."""
    await client_with_mock.post("/generate", json=RYUJI_REQUEST)
    r = await client_with_mock.get("/health")
    assert r.json()["status"] in ("ready", "mock")


@pytest.mark.asyncio
async def test_generate_then_ready(client_with_mock: AsyncClient) -> None:
    """Server reports ready after a generate call."""
    await client_with_mock.post("/generate", json=RYUJI_REQUEST)
    r = await client_with_mock.get("/ready")
    assert r.json()["ready"] is True


@pytest.mark.asyncio
async def test_high_rank_request_succeeds(client_with_mock: AsyncClient) -> None:
    """Rank 10 (max bond) request succeeds."""
    req = {**RYUJI_REQUEST, "rank": 10}
    r = await client_with_mock.post("/generate", json=req)
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_all_confidant_ids_in_roster_generate(client_with_mock: AsyncClient) -> None:
    """All 22 confidants in the roster return 200."""
    from social_link.arcana import CONFIDANTS
    for cid in CONFIDANTS:
        req = {**RYUJI_REQUEST, "confidant_id": cid}
        r = await client_with_mock.post("/generate", json=req)
        assert r.status_code == 200, f"Failed for confidant_id={cid}"


@pytest.mark.asyncio
async def test_generate_then_clear_stats_then_generate(client_with_mock: AsyncClient) -> None:
    """Stats reset mid-session: second generate after clear should see total_completions=1."""
    await client_with_mock.post("/generate", json=RYUJI_REQUEST)
    await client_with_mock.post("/clear-stats")
    await client_with_mock.post("/generate", json=RYUJI_REQUEST)
    r = await client_with_mock.get("/stats")
    assert r.json()["total_completions"] == 1


@pytest.mark.asyncio
async def test_model_info_after_generate(client_with_mock: AsyncClient) -> None:
    """/model-info returns a valid mode after a generate call."""
    await client_with_mock.post("/generate", json=RYUJI_REQUEST)
    r = await client_with_mock.get("/model-info")
    assert r.status_code == 200
    assert r.json()["mode"] in ("mock", "real", "not_loaded")
