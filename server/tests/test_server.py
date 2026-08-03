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


@pytest.mark.asyncio
async def test_stats_endpoint_returns_expected_keys() -> None:
    import main as srv
    transport = ASGITransport(app=srv.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/stats")
    assert r.status_code == 200
    body = r.json()
    assert "total_requests" in body
    assert "total_drops" in body
    assert "total_completions" in body
    assert "avg_latency_ms" in body
    assert "model_loaded" in body
    assert isinstance(body["total_requests"], int)


@pytest.mark.asyncio
async def test_ready_false_without_model() -> None:
    import main as srv
    srv._pipeline = None
    transport = ASGITransport(app=srv.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/ready")
    assert r.status_code == 200
    assert r.json() == {"ready": False}


@pytest.mark.asyncio
async def test_ready_true_with_mock_pipeline() -> None:
    import main as srv
    srv._pipeline = srv._MOCK
    transport = ASGITransport(app=srv.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/ready")
    assert r.json() == {"ready": True}
    srv._pipeline = None


@pytest.mark.asyncio
async def test_model_info_no_model() -> None:
    import main as srv
    srv._pipeline = None
    transport = ASGITransport(app=srv.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "not_loaded"
    assert "model_path" in body
    assert "temperature" in body


@pytest.mark.asyncio
async def test_model_info_mock_mode() -> None:
    import main as srv
    srv._pipeline = srv._MOCK
    transport = ASGITransport(app=srv.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/model-info")
    assert r.json()["mode"] == "mock"
    srv._pipeline = None


@pytest.mark.asyncio
async def test_stats_completions_increment_after_generate() -> None:
    import main as srv
    from inference.queue import InferenceQueue
    srv._queue = InferenceQueue()  # fresh queue for isolation
    mock = MagicMock()
    mock.generate.return_value = "Yo, let's go!"
    srv._pipeline = mock
    transport = ASGITransport(app=srv.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/generate", json={
            "confidant_id": 8, "rank": 4,
            "context": "gym scene", "character_name": "Ryuji Sakamoto",
        })
        r = await client.get("/stats")
    assert r.json()["total_completions"] == 1
    assert r.json()["total_requests"] == 1
    srv._pipeline = None