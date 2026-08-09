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


def test_mock_response_format_has_prefix() -> None:
    """Every mock response must start with [MOCK rank N] for easy identification in logs."""
    for cid in CONFIDANTS:
        for rank in (1, 5, 10):
            result = get_mock_response(cid, rank)
            assert result.startswith(f"[MOCK rank {rank}]"), (
                f"Confidant {cid} rank {rank} response missing prefix: {result[:30]}"
            )


def test_mock_lines_all_non_empty() -> None:
    """All 22 canned lines must have actual content."""
    for cid, line in MOCK_LINES.items():
        assert len(line.strip()) > 5, f"Mock line for confidant {cid} is too short"


def test_mock_response_different_ranks_same_format() -> None:
    """Rank number in prefix should vary; content should stay the same canned line."""
    r5 = get_mock_response(8, 5)
    r9 = get_mock_response(8, 9)
    assert "[MOCK rank 5]" in r5
    assert "[MOCK rank 9]" in r9
    assert r5 != r9  # different rank numbers make them distinct
