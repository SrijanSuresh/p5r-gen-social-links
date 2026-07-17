"""Tests for the response post-processor."""

from inference.postprocess import clean_response


def test_strips_ooc_commentary() -> None:
    raw = "You think so? (OOC: I am an AI language model) That means a lot."
    assert "(OOC:" not in clean_response(raw)
    assert "That means a lot." in clean_response(raw)


def test_truncates_to_max_chars() -> None:
    long_text = "A" * 300
    assert len(clean_response(long_text, max_chars=200)) == 200


def test_strips_ai_disclosure() -> None:
    raw = "As an AI language model, I cannot. But let me try anyway!"
    result = clean_response(raw)
    assert "As an AI" not in result