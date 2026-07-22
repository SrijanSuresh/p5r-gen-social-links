"""Smoke tests for the FastAPI server — run without a GPU or model."""

import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import MagicMock


@pytest.mark.asyncio
async def test_health_no_model() -> None:
    import main as srv
    srv._pipeline = None
    transport = ASGITransport(app=srv.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "model_not_loaded"


@pytest.mark.asyncio
async def test_generate_503_without_model() -> None:
    import main as srv
    srv._pipeline = None
    transport = ASGITransport(app=srv.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/generate", json={
            "confidant_id": 1, "rank": 3,
            "context": "test", "character_name": "Ryuji Sakamoto",
        })
    assert r.status_code == 503


@pytest.mark.asyncio
async def test_generate_with_mocked_pipeline() -> None:
    import main as srv
    mock = MagicMock()
    mock.generate.return_value = "Man, that''s rough. We''ll figure it out!"
    srv._pipeline = mock
    transport = ASGITransport(app=srv.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/generate", json={
            "confidant_id": 1, "rank": 5,
            "context": "Ryuji is talking about his injury.",
            "character_name": "Ryuji Sakamoto",
        })
    assert r.status_code == 200
    assert len(r.json()["text"]) > 0
    srv._pipeline = None