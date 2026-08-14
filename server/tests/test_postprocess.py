"""Tests for the response post-processor."""

from inference.postprocess import clean_response, _truncate_at_sentence


def test_strips_ooc_commentary() -> None:
    raw = "You think so? (OOC: I am an AI language model) That means a lot."
    assert "(OOC:" not in clean_response(raw)
    assert "That means a lot." in clean_response(raw)


def test_truncates_to_max_chars() -> None:
    # A 300-character run with no punctuation has no sentence boundary to cut back to,
    # so it is rejected outright rather than sliced.
    assert clean_response("A" * 300, max_chars=200) == ""


def test_long_text_with_punctuation_keeps_whole_sentences() -> None:
    text = "First one here. " + "B" * 300
    assert clean_response(text, max_chars=200) == "First one here."


def test_strips_ai_disclosure() -> None:
    raw = "As an AI language model, I cannot. But let me try anyway!"
    result = clean_response(raw)
    assert "As an AI" not in result


def test_strips_name_prefix_single() -> None:
    raw = "Ryuji: Yo, let's train harder!"
    result = clean_response(raw)
    assert not result.startswith("Ryuji:")
    assert "train harder" in result


def test_strips_name_prefix_full_name() -> None:
    raw = "Ryuji Sakamoto: We're gonna crush it today."
    result = clean_response(raw)
    assert not result.startswith("Ryuji")
    assert "crush it today" in result


def test_truncate_at_sentence_boundary() -> None:
    text = "First sentence. Second sentence goes over the limit here."
    result = _truncate_at_sentence(text, max_chars=20)
    assert result == "First sentence."


def test_no_sentence_boundary_yields_nothing() -> None:
    """
    Reverses an earlier expectation. This used to assert a word-boundary fallback, and
    its assertions -- "does not end with a space", "is not longer than the limit" -- are
    both satisfied by the empty string, so it could never have caught the change.

    Word-boundary truncation shipped "Yeah, membership's pretty" and "Where'd I put those
    extra" into speech bubbles: severed mid-thought, and worse to read than the scripted
    line they replaced. With nothing returned the mod leaves the record alone and the
    player gets canon dialogue.
    """
    assert _truncate_at_sentence("No punctuation at all just words here", max_chars=15) == ""


def test_partial_sentence_after_a_complete_one_is_dropped() -> None:
    text = "Yo. Now here comes a much longer clause that will not fit"
    assert _truncate_at_sentence(text, max_chars=30) == "Yo."


def test_short_record_rejects_an_overlong_line() -> None:
    """A fifteen-character bubble cannot hold a sentence, so it must get none."""
    assert clean_response("Yeah, membership is pretty reasonable", max_chars=15) == ""


def test_a_line_that_fits_is_untouched_by_the_new_rule() -> None:
    assert clean_response("Yo, let's go!", max_chars=30) == "Yo, let's go!"


def test_short_text_not_truncated() -> None:
    text = "Short."
    assert clean_response(text, max_chars=200) == "Short."


def test_emoji_are_dropped_and_the_line_survives() -> None:
    """
    Emoji cannot reach the bubble intact: the mod writes ASCII, so each one becomes a
    literal '?'. The line around them must stay readable.

    This previously asserted emoji "should survive as unknown glyphs" — but its
    assertions never checked, so it passed either way. Now it pins the real behaviour.
    """
    result = clean_response("Let's go! 🔥 We'll crush this!")
    assert "🔥" not in result
    assert result.isascii()
    assert "crush this" in result


def test_japanese_characters_are_stripped_not_passed_through() -> None:
    """
    Reverses an earlier expectation, because the destination turned out to be ASCII.

    This test previously asserted Japanese survived, written while the dialogue buffer
    was assumed to be UTF-16. It is not: the buffer was confirmed single-byte ASCII
    (learning.md Ch. 60), and the mod writes it with Encoding.ASCII, which maps every
    character above 0x7F to a literal '?'.

    So passing Japanese through does not render Japanese — it renders "????" in the
    speech bubble. Dropping it leaves a clean, readable English line instead.
    """
    result = clean_response("Man, that's tough... でも頑張ろう。")
    assert "頑張ろう" not in result
    assert result.isascii()
    assert "Man, that's tough..." in result


def test_empty_string_returns_empty() -> None:
    assert clean_response("") == ""


def test_whitespace_only_returns_empty() -> None:
    assert clean_response("   \n\t  ") == ""


def test_ooc_mid_sentence_removal_leaves_valid_text() -> None:
    raw = "You think so? (OOC: remember this is a game) That means everything."
    result = clean_response(raw)
    assert "OOC" not in result
    assert "means everything" in result


def test_multiple_ooc_markers_all_removed() -> None:
    raw = "(OOC: aside one) Hello! (OOC: aside two) Nice to meet you."
    result = clean_response(raw)
    assert "OOC" not in result
    assert "Hello" in result
    assert "Nice to meet" in result

# --- game-buffer safety -------------------------------------------------------
#
# The mod writes the result with Encoding.ASCII into a fixed-length slot, so anything
# these miss becomes visible in the speech bubble.


def test_stage_directions_are_removed() -> None:
    """Observed live: Ryuji returned a line ending in *pumps fist*."""
    assert "*" not in clean_response("Let's go! *pumps fist*")


def test_stage_direction_mid_sentence_is_removed() -> None:
    assert "*" not in clean_response("Yo *grins* you ready?")


def test_text_survives_stage_direction_removal() -> None:
    assert "Let's go!" in clean_response("Let's go! *pumps fist*")


def test_wrapping_quotes_are_stripped() -> None:
    """The model quotes its line about half the time; the bubble never shows them."""
    assert clean_response('"Yo, what\'s up?"') == "Yo, what's up?"


def test_unquoted_line_is_unchanged() -> None:
    assert clean_response("Yo, what's up?") == "Yo, what's up?"


def test_internal_quotes_are_preserved() -> None:
    """Stripping the outer pair here would corrupt a line with quoted speech."""
    source = '"Yo," he said, "let\'s go"'
    assert clean_response(source) == source


def test_curly_apostrophe_is_folded() -> None:
    """Encoding.ASCII would render this as 'slackin?off' in-game."""
    assert clean_response("You've been slackin’ off") == "You've been slackin' off"


def test_em_dash_is_folded() -> None:
    assert clean_response("Yo — you good?") == "Yo - you good?"


def test_ellipsis_is_folded() -> None:
    assert clean_response("Well… maybe.") == "Well... maybe."


def test_curly_double_quotes_are_folded_then_stripped() -> None:
    assert clean_response("“Yo, dude!”") == "Yo, dude!"


def test_output_is_always_pure_ascii() -> None:
    """A catch-all: whatever the model emits, the game buffer receives ASCII."""
    messy = "Café — “yo” … über 日本語 \U0001f600 done."
    result = clean_response(messy)
    assert result.isascii(), result


def test_non_breaking_space_becomes_a_plain_space() -> None:
    assert clean_response("Yo there") == "Yo there"


def test_two_quoted_utterances_are_merged() -> None:
    """Observed live: the model returned '"Come on!" "Let\'s go!"' as one response."""
    result = clean_response('"Come on!" "Let\'s go!"')
    assert '"' not in result
    assert result == "Come on! Let's go!"


def test_three_quoted_segments_are_merged() -> None:
    assert clean_response('"A." "B." "C."') == "A. B. C."


def test_quoted_speech_inside_a_line_is_not_unwrapped() -> None:
    """The safeguard: this is narration, not a wrapped utterance."""
    source = 'He said, "go" and left'
    assert clean_response(source) == source
