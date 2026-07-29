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