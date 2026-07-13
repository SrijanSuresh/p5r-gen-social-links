"""Smoke tests for the FastAPI server scaffold."""

import pytest
from httpx import AsyncClient, ASGITransport
from server.main import app


@pytest.mark.asyncio
async def test_health_endpoint() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_generate_returns_501_before_model_loaded() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/generate", json={
            "confidant_id": 1,
            "rank": 3,
            "context": "Ryuji is talking about training.",
            "character_name": "Ryuji Sakamoto",
        })
    assert response.status_code == 501
