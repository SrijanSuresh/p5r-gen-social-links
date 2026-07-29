# Unit tests for prompt construction logic.
import pytest
from social_link.prompt_builder import build_prompt, _tier_note


def test_build_prompt_ryuji() -> None:
    system, user = build_prompt(confidant_id=8, rank=3, context="Track practice discussion")
    assert "Ryuji Sakamoto" in system
    assert "Chariot" in system
    assert "rank 3/10" in system
    assert "Track practice discussion" in user


def test_build_prompt_unknown_confidant() -> None:
    with pytest.raises(KeyError):
        build_prompt(confidant_id=999, rank=1, context="test")


def test_tier_note_rank_1_is_reserved() -> None:
    note = _tier_note(1)
    assert "just met" in note or "reserved" in note


def test_tier_note_rank_4_is_acquaintance() -> None:
    note = _tier_note(4)
    assert "Casual" in note or "warming" in note


def test_tier_note_rank_7_is_close_friends() -> None:
    note = _tier_note(7)
    assert "Close" in note or "banter" in note


def test_tier_note_rank_9_is_deepest_bond() -> None:
    note = _tier_note(9)
    assert "trust" in note or "warmth" in note


def test_tier_note_rank_boundary_2_is_reserved() -> None:
    assert "just met" in _tier_note(2) or "reserved" in _tier_note(2)


def test_tier_note_rank_boundary_3_is_acquaintance() -> None:
    assert "warming" in _tier_note(3) or "Casual" in _tier_note(3)
