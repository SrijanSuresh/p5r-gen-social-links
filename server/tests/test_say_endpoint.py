"""
Phase 5's input channel over HTTP: the page posts, the mod collects (Ch. 87).

The division of labour is the thing being tested. The server holds the line and hands it
over on request; it never decides which record should answer it, because /generate runs
several records ahead of the player and only the mod knows where the cursor is.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from tests.conftest import RYUJI_REQUEST


@pytest.fixture(autouse=True)
def _empty_store() -> None:
    """The store is module-level state, so each test starts with nothing pending."""
    import main as srv

    srv._player_input.clear()


# --- submitting ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_line_is_accepted_and_echoed_back_cleaned(
    client_with_mock: AsyncClient,
) -> None:
    r = await client_with_mock.post("/say", json={"text": "  so how   have you been?  "})

    assert r.status_code == 200
    assert r.json() == {"accepted": True, "text": "so how have you been?"}


@pytest.mark.asyncio
async def test_an_empty_line_is_rejected_with_a_reason(
    client_with_mock: AsyncClient,
) -> None:
    r = await client_with_mock.post("/say", json={"text": "   "})

    assert r.status_code == 422
    assert "nothing to say" in r.json()["detail"]


@pytest.mark.asyncio
async def test_a_line_over_the_store_limit_is_rejected(
    client_with_mock: AsyncClient,
) -> None:
    r = await client_with_mock.post("/say", json={"text": "x" * 300})

    assert r.status_code == 422
    assert "limit is" in r.json()["detail"]


@pytest.mark.asyncio
async def test_a_line_over_the_field_limit_never_reaches_the_store(
    client_with_mock: AsyncClient,
) -> None:
    # Pydantic rejects it first — the cheap check before the contract check.
    r = await client_with_mock.post("/say", json={"text": "x" * 2000})
    assert r.status_code == 422


# --- collecting ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_nothing_pending_reports_nothing(client_with_mock: AsyncClient) -> None:
    r = await client_with_mock.get("/player-line")

    assert r.status_code == 200
    assert r.json() == {"pending": False, "text": "", "age_seconds": 0.0}


@pytest.mark.asyncio
async def test_peeking_leaves_the_line_for_the_request_that_will_use_it(
    client_with_mock: AsyncClient,
) -> None:
    await client_with_mock.post("/say", json={"text": "you free later?"})

    first = await client_with_mock.get("/player-line")
    second = await client_with_mock.get("/player-line")

    assert first.json()["pending"] is True
    assert first.json()["text"] == "you free later?"
    # Peeking twice still finds it: the mod looks before it commits to a record.
    assert second.json()["text"] == "you free later?"


@pytest.mark.asyncio
async def test_consuming_takes_the_line_exactly_once(
    client_with_mock: AsyncClient,
) -> None:
    await client_with_mock.post("/say", json={"text": "you free later?"})

    taken = await client_with_mock.get("/player-line", params={"consume": "true"})
    after = await client_with_mock.get("/player-line")

    assert taken.json()["text"] == "you free later?"
    # One typed line must not be answered by two records.
    assert after.json()["pending"] is False


@pytest.mark.asyncio
async def test_sending_again_replaces_an_uncollected_line(
    client_with_mock: AsyncClient,
) -> None:
    await client_with_mock.post("/say", json={"text": "wait no"})
    await client_with_mock.post("/say", json={"text": "actually this"})

    r = await client_with_mock.get("/player-line", params={"consume": "true"})
    assert r.json()["text"] == "actually this"


@pytest.mark.asyncio
async def test_clearing_discards_a_pending_line(client_with_mock: AsyncClient) -> None:
    await client_with_mock.post("/say", json={"text": "never mind"})

    cleared = await client_with_mock.post("/player-line/clear")
    after = await client_with_mock.get("/player-line")

    assert cleared.status_code == 200
    assert after.json()["pending"] is False


# --- the server never injects a line on its own -------------------------------


@pytest.mark.asyncio
async def test_generate_does_not_consume_a_pending_line_by_itself(
    client_with_mock: AsyncClient, mock_pipeline
) -> None:
    """
    A pre-generation request is for a record the player has not reached. If /generate
    silently took the pending line, the confidant would answer it several bubbles too
    late — worse than not having the feature (Ch. 87). So the caller must pass it.
    """
    await client_with_mock.post("/say", json={"text": "are you okay?"})

    r = await client_with_mock.post("/generate", json=RYUJI_REQUEST)

    assert r.status_code == 200
    assert mock_pipeline.generate.call_args.args[-1] == ""      # player_line
    # And the line is still waiting for whoever actually wants it.
    still = await client_with_mock.get("/player-line")
    assert still.json()["text"] == "are you okay?"


@pytest.mark.asyncio
async def test_generate_forwards_a_player_line_it_is_given(
    client_with_mock: AsyncClient, mock_pipeline
) -> None:
    body = {**RYUJI_REQUEST, "player_line": "are you okay?"}

    r = await client_with_mock.post("/generate", json=body)

    assert r.status_code == 200
    assert mock_pipeline.generate.call_args.args[-1] == "are you okay?"


@pytest.mark.asyncio
async def test_generate_rejects_a_player_line_over_the_limit(
    client_with_mock: AsyncClient,
) -> None:
    body = {**RYUJI_REQUEST, "player_line": "x" * 300}
    r = await client_with_mock.post("/generate", json=body)
    assert r.status_code == 422


# --- the page and the counters -----------------------------------------------


@pytest.mark.asyncio
async def test_the_typing_page_is_served(client_with_mock: AsyncClient) -> None:
    r = await client_with_mock.get("/")

    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    # It has to post to the route that exists, and bound the box to the store's limit.
    assert "/say" in r.text
    assert "280" in r.text


@pytest.mark.asyncio
async def test_stats_reports_the_channel(client_with_mock: AsyncClient) -> None:
    await client_with_mock.post("/say", json={"text": "hey"})
    await client_with_mock.get("/player-line", params={"consume": "true"})

    body = (await client_with_mock.get("/stats")).json()
    assert body["player_lines_submitted"] >= 1
    assert body["player_lines_collected"] >= 1
    assert "player_lines_expired" in body
