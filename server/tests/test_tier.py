"""Tests for the rank-to-tier mapping module."""

from __future__ import annotations

import pytest

from social_link.tier import tier_note, tier_label, TIER_LABELS


@pytest.mark.parametrize("rank,expected_fragment", [
    (1, "just met"),
    (2, "just met"),
    (3, "Acquaintances"),
    (5, "Acquaintances"),
    (6, "Close friends"),
    (8, "Close friends"),
    (9, "Deepest bond"),
    (10, "Deepest bond"),
])
def test_tier_note_correct_text(rank: int, expected_fragment: str) -> None:
    assert expected_fragment in tier_note(rank)


def test_tier_note_returns_string() -> None:
    for rank in range(1, 11):
        result = tier_note(rank)
        assert isinstance(result, str)
        assert len(result) > 0


def test_tier_label_stranger() -> None:
    assert tier_label(1) == "stranger"
    assert tier_label(2) == "stranger"


def test_tier_label_acquaintance() -> None:
    assert tier_label(3) == "acquaintance"
    assert tier_label(5) == "acquaintance"


def test_tier_label_close_friend() -> None:
    assert tier_label(6) == "close_friend"
    assert tier_label(8) == "close_friend"


def test_tier_label_deepest_bond() -> None:
    assert tier_label(9) == "deepest_bond"
    assert tier_label(10) == "deepest_bond"


def test_tier_labels_dict_covers_all_ranks() -> None:
    """Every rank 1-10 maps to exactly one label."""
    for rank in range(1, 11):
        matches = [label for label, r in TIER_LABELS.items() if rank in r]
        assert len(matches) == 1, f"Rank {rank} matched {len(matches)} labels"
