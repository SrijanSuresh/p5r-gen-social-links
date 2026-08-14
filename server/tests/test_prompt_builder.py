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


def test_system_prompt_contains_personality_blurb() -> None:
    """Character notes (personality blurb) must appear in the system prompt."""
    system, _ = build_prompt(confidant_id=8, rank=5, context="gym")
    # Ryuji's blurb: "Loud, loyal, hot-headed best friend..."
    assert "loyal" in system.lower() or "loud" in system.lower() or "hot-headed" in system.lower()


def test_system_prompt_has_no_ai_rule() -> None:
    """The 'do not reference you are an AI' rule must be present."""
    system, _ = build_prompt(confidant_id=8, rank=5, context="gym")
    assert "AI" in system


def test_system_prompt_has_no_name_start_rule() -> None:
    """The prompt must contain the instruction not to start with the character's name."""
    system, _ = build_prompt(confidant_id=8, rank=5, context="gym")
    assert "Do NOT start" in system


def test_user_prompt_includes_scene_context_label() -> None:
    system, user = build_prompt(confidant_id=8, rank=3, context="ramen shop")
    assert "[Scene context:" in user
    assert "ramen shop" in user


def test_all_confidants_build_prompt_without_error() -> None:
    """Every registered confidant must produce a valid (system, user) tuple."""
    from social_link.arcana import CONFIDANTS
    for cid in CONFIDANTS:
        system, user = build_prompt(cid, 5, "test context")
        assert len(system) > 50
        assert len(user) > 10


# --- per-record length budget -------------------------------------------------
#
# The line is written into one specific message record, and records differ: one row
# holds roughly 30 characters, two rows roughly 75. A fixed budget overran the short
# ones — "You're finally here, I've been" reached the screen with the rest cut off.


def test_budget_appears_in_the_system_prompt() -> None:
    system, _ = build_prompt(8, 4, "at the gym", max_chars=30)
    assert "30 characters" in system


def test_budget_is_also_given_in_words() -> None:
    """Models count words far more reliably than characters."""
    system, _ = build_prompt(8, 4, "at the gym", max_chars=50)
    assert "10 words" in system


def test_word_budget_never_drops_below_three() -> None:
    """A tiny record must still ask for a sentence, not for one word."""
    system, _ = build_prompt(8, 4, "at the gym", max_chars=8)
    assert "3 words" in system


def test_two_records_of_different_size_get_different_budgets() -> None:
    narrow, _ = build_prompt(8, 4, "at the gym", max_chars=30)
    wide,   _ = build_prompt(8, 4, "at the gym", max_chars=75)
    assert narrow != wide


def test_default_budget_is_used_when_none_given() -> None:
    """Callers with no record in hand must still get a usable prompt."""
    system, _ = build_prompt(8, 4, "at the gym")
    assert "characters" in system
