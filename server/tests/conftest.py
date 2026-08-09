"""Shared pytest fixtures for server tests."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from httpx import AsyncClient, ASGITransport


@pytest.fixture
def mock_pipeline() -> MagicMock:
    """A mock InferencePipeline that returns a canned Ryuji response."""
    mock = MagicMock()
    mock.generate.return_value = "Yo, let's crush it today!"
    return mock


@pytest.fixture
async def client_with_mock(mock_pipeline: MagicMock) -> AsyncClient:
    """AsyncClient wired to the FastAPI app with a mock pipeline."""
    import main as srv
    from inference.queue import InferenceQueue
    srv._pipeline = mock_pipeline
    srv._queue = InferenceQueue()
    transport = ASGITransport(app=srv.app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    srv._pipeline = None


@pytest.fixture
async def client_no_model() -> AsyncClient:
    """AsyncClient wired to the FastAPI app with no model loaded."""
    import main as srv
    srv._pipeline = None
    transport = ASGITransport(app=srv.app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


RYUJI_REQUEST = {
    "confidant_id": 8,
    "rank": 4,
    "context": "[Scene 51] Hang-out with Ryuji Sakamoto (rank 4/10).",
    "character_name": "Ryuji Sakamoto",
}
