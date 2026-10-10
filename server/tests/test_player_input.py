"""
The player's typed line, from submission to collection (learning.md Ch. 87).

The store is deliberately small and the behaviour that matters is all about *when* a line
stops being valid: replaced by a retype, consumed by a generation, or aged out.
"""

from __future__ import annotations

import pytest

from social_link.player_input import (
    DEFAULT_TTL_SECONDS,
    MAX_LINE_CHARS,
    PlayerInputError,
    PlayerInputStore,
    normalise,
)


# --- cleaning what a browser sends -------------------------------------------


def test_surrounding_and_repeated_whitespace_collapses() -> None:
    assert normalise("  so   what   now?  ") == "so what now?"


def test_newlines_become_spaces() -> None:
    # The user half of the prompt is built from one-per-line `[...]` blocks, so a
    # submitted paragraph would otherwise break its structure.
    assert normalise("first\nsecond\r\nthird") == "first second third"


def test_control_characters_are_dropped() -> None:
    assert normalise("hey\x00there\x07") == "heythere"


def test_an_empty_submission_is_refused() -> None:
    for blank in ("", "   ", "\n\n", "\x00"):
        with pytest.raises(PlayerInputError, match="nothing to say"):
            normalise(blank)


def test_a_line_longer_than_the_limit_is_refused() -> None:
    with pytest.raises(PlayerInputError, match="the limit is"):
        normalise("x" * (MAX_LINE_CHARS + 1))
    assert len(normalise("x" * MAX_LINE_CHARS)) == MAX_LINE_CHARS


def test_non_text_is_refused() -> None:
    with pytest.raises(PlayerInputError, match="must be text"):
        normalise(42)  # type: ignore[arg-type]


# --- holding one line --------------------------------------------------------


def test_a_submitted_line_can_be_collected_once() -> None:
    store = PlayerInputStore()
    store.submit("you sure about that?", now=100.0)

    taken = store.take(now=101.0)
    assert taken is not None and taken.text == "you sure about that?"

    # The second collector gets nothing: one typed line must not be answered by several
    # records.
    assert store.take(now=102.0) is None


def test_peeking_does_not_consume() -> None:
    store = PlayerInputStore()
    store.submit("still here", now=0.0)

    assert store.peek(now=1.0) is not None
    assert store.peek(now=1.0) is not None
    assert store.take(now=1.0) is not None
    assert store.peek(now=1.0) is None


def test_retyping_replaces_the_uncollected_line() -> None:
    # The second line is the one the player meant.
    store = PlayerInputStore()
    store.submit("wait no", now=0.0)
    store.submit("actually this", now=1.0)

    taken = store.take(now=2.0)
    assert taken is not None and taken.text == "actually this"
    assert store.take(now=2.0) is None


def test_an_empty_store_yields_nothing() -> None:
    store = PlayerInputStore()
    assert store.peek(now=0.0) is None
    assert store.take(now=0.0) is None


def test_clear_forgets_a_pending_line() -> None:
    store = PlayerInputStore()
    store.submit("never mind", now=0.0)
    store.clear()
    assert store.take(now=1.0) is None


# --- expiry ------------------------------------------------------------------


def test_a_line_older_than_the_ttl_is_not_collected() -> None:
    # Typed, then abandoned. It must not surface in a scene half an hour later.
    store = PlayerInputStore(ttl_seconds=10.0)
    store.submit("hello?", now=0.0)

    assert store.take(now=10.5) is None
    assert store.expired_count == 1


def test_a_line_inside_the_ttl_survives() -> None:
    store = PlayerInputStore(ttl_seconds=10.0)
    store.submit("hello?", now=0.0)

    taken = store.take(now=9.9)
    assert taken is not None
    assert store.expired_count == 0


def test_peeking_also_expires_a_stale_line() -> None:
    store = PlayerInputStore(ttl_seconds=5.0)
    store.submit("anyone?", now=0.0)

    assert store.peek(now=6.0) is None
    assert store.take(now=6.0) is None


def test_the_default_ttl_covers_typing_and_a_retype() -> None:
    assert DEFAULT_TTL_SECONDS >= 60.0


def test_a_nonpositive_ttl_is_refused() -> None:
    with pytest.raises(PlayerInputError, match="must be positive"):
        PlayerInputStore(ttl_seconds=0.0)


# --- counters, which are what /stats reports --------------------------------


def test_counters_track_submissions_collections_and_expiries() -> None:
    store = PlayerInputStore(ttl_seconds=5.0)

    store.submit("one", now=0.0)
    store.take(now=1.0)
    store.submit("two", now=2.0)
    store.take(now=20.0)        # expired before anything collected it

    assert store.submitted_count == 2
    assert store.collected_count == 1
    assert store.expired_count == 1
