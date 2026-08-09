"""Tests for the response post-processor."""

from inference.postprocess import clean_response, _truncate_at_sentence


def test_strips_ooc_commentary() -> None:
    raw = "You think so? (OOC: I am an AI language model) That means a lot."
    assert "(OOC:" not in clean_response(raw)
    assert "That means a lot." in clean_response(raw)


def test_truncates_to_max_chars() -> None:
    # Old behaviour: hard truncate; new: sentence boundary, may be < max_chars
    result = clean_response("A" * 300, max_chars=200)
    assert len(result) <= 200


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


def test_truncate_falls_back_to_word_boundary() -> None:
    text = "No punctuation at all just words here"
    result = _truncate_at_sentence(text, max_chars=15)
    assert not result.endswith(" ")
    assert len(result) <= 15


def test_short_text_not_truncated() -> None:
    text = "Short."
    assert clean_response(text, max_chars=200) == "Short."


def test_emoji_in_response_passes_through() -> None:
    """Emoji should survive cleaning — the game renderer may handle them as unknown glyphs."""
    raw = "Let's go! 🔥 We'll crush this!"
    result = clean_response(raw)
    assert "crush this" in result
    assert len(result) > 0


def test_japanese_characters_pass_through() -> None:
    """Japanese characters from mixed-language LLM output must not be stripped."""
    raw = "Man, that's tough... でも頑張ろう。"
    result = clean_response(raw)
    assert "頑張ろう" in result


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