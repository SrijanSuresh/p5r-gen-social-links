"""Tests for per-character mock responses."""

from social_link.mock_responses import get_mock_response, MOCK_LINES
from social_link.arcana import CONFIDANTS


def test_ryuji_mock_contains_slang() -> None:
    result = get_mock_response(8, 4)
    assert "dude" in result.lower() or "crush" in result.lower()


def test_morgana_mock_is_sassy() -> None:
    result = get_mock_response(2, 3)
    assert "slack" in result.lower() or "trickster" in result.lower() or "leader" in result.lower()


def test_rank_appears_in_response() -> None:
    result = get_mock_response(8, 7)
    assert "7" in result


def test_unknown_id_returns_default() -> None:
    result = get_mock_response(99, 1)
    assert "[MOCK" in result
    assert len(result) > 0


def test_all_known_confidants_have_mock_lines() -> None:
    for cid in CONFIDANTS:
        result = get_mock_response(cid, 5)
        assert result.startswith("[MOCK rank 5]"), f"No mock line for confidant {cid}"
        assert len(result) > len("[MOCK rank 5] ")


def test_mock_lines_count_matches_roster() -> None:
    assert len(MOCK_LINES) == len(CONFIDANTS)
