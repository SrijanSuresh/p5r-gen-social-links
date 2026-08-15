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


def test_the_stated_target_sits_below_the_real_capacity() -> None:
    """
    The model overshoots whatever number it is given, and an overshoot is discarded
    whole rather than trimmed — the clip only keeps text ending on a sentence boundary.
    Aiming below capacity leaves room for the overshoot to land inside it.
    """
    _, user = build_prompt(8, 4, "at the gym", max_chars=30)
    assert "30 characters" not in user
    assert "25 characters" in user      # 85% of 30


def test_the_target_never_collapses_on_a_tiny_record() -> None:
    _, user = build_prompt(8, 4, "at the gym", max_chars=12)
    assert "12 characters" in user


def test_the_prompt_asks_for_a_finished_sentence() -> None:
    """An unfinished line is thrown away, so finishing matters more than length."""
    _, user = build_prompt(8, 4, "at the gym", max_chars=40)
    assert "Finish the sentence" in user


def test_budget_is_also_given_in_words() -> None:
    """Models count words far more reliably than characters."""
    _, user = build_prompt(8, 4, "at the gym", max_chars=50)
    assert "8 words" in user      # 85% of 50 is 42, at ~5 chars a word


def test_word_budget_never_drops_below_three() -> None:
    """A tiny record must still ask for a sentence, not for one word."""
    _, user = build_prompt(8, 4, "at the gym", max_chars=8)
    assert "3 words" in user


def test_two_records_of_different_size_get_different_budgets() -> None:
    _, narrow = build_prompt(8, 4, "at the gym", max_chars=30)
    _, wide   = build_prompt(8, 4, "at the gym", max_chars=75)
    assert narrow != wide


def test_the_system_prompt_does_not_vary_with_the_budget() -> None:
    """
    The point of moving the limit out of the system prompt.

    llama-server reuses the KV cache for the longest identical prefix between requests,
    and the system prompt is that prefix. Baking a per-record number into it made every
    request differ from the first token, so nothing was reusable and generation slowed
    from ~1.5s to ~3.5s once the prompt grew a voice section and four lines of history.
    """
    narrow, _ = build_prompt(8, 4, "at the gym", max_chars=30)
    wide,   _ = build_prompt(8, 4, "at the gym", max_chars=98)
    assert narrow == wide


def test_the_system_prompt_is_stable_across_scenes() -> None:
    """Only the user half may change within one hang-out, or the cache is worthless."""
    a, _ = build_prompt(8, 4, "at the gym", max_chars=40)
    b, _ = build_prompt(8, 4, "somewhere else entirely", max_chars=60)
    assert a == b


def test_default_budget_is_used_when_none_given() -> None:
    """Callers with no record in hand must still get a usable prompt."""
    _, user = build_prompt(8, 4, "at the gym")
    assert "characters" in user


# --- replacing a specific line ------------------------------------------------
#
# Pre-generation hands the model the scripted line it is displacing, so the rules have
# to say what to do with it. Without this, two consecutive records came back as "You
# comin' back here every week like me now?" and "You comin' here more often now?" — one
# sentence twice, because every request carried the same scene blurb and nothing else.


def test_rules_mention_replacing_a_line() -> None:
    system, _ = build_prompt(8, 4, "at the gym")
    assert "replacing" in system


def test_rules_forbid_restating_an_earlier_line() -> None:
    system, _ = build_prompt(8, 4, "at the gym")
    assert "already said" in system


def test_context_reaches_the_user_prompt_verbatim() -> None:
    """The line being replaced travels in the context, so it must survive intact."""
    original = 'The line you are replacing is: "A towel?"'
    _, user = build_prompt(8, 4, original)
    assert original in user


# --- voice ---------------------------------------------------------------------
#
# personality_blurb alone produced "Guess I'm cool with paying for the session if that's
# what keeps this place running." from Ryuji: true to the description, and nothing like
# the character. How someone sounds is separate information from who they are.


def test_ryuji_gets_his_speech_style() -> None:
    system, _ = build_prompt(8, 4, "at the gym")
    assert "bro" in system
    assert "ain't" in system


def test_ryuji_is_allowed_to_swear() -> None:
    system, _ = build_prompt(8, 4, "at the gym")
    assert "bullshit" in system
    assert "Do not censor" in system


def test_makoto_is_not() -> None:
    """One global profanity setting would break more characters than it fixed."""
    system, _ = build_prompt(3, 4, "at school")
    assert "You do not swear" in system


def test_a_confidant_without_a_style_still_builds() -> None:
    system, _ = build_prompt(11, 4, "reading fortunes")
    assert "How you talk:" in system


def test_continuity_rule_is_stated() -> None:
    system, _ = build_prompt(8, 4, "at the gym")
    assert "one conversation" in system


def test_voice_outranks_correctness() -> None:
    system, _ = build_prompt(8, 4, "at the gym")
    assert "Sound like yourself before you sound correct" in system
