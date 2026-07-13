# Unit tests for prompt construction logic.
import pytest
from social_link.prompt_builder import build_prompt


def test_build_prompt_ryuji() -> None:
    system, user = build_prompt(confidant_id=1, rank=3, context="Track practice discussion")
    assert "Ryuji Sakamoto" in system
    assert "Chariot" in system
    assert "rank 3/10" in system
    assert "Track practice discussion" in user


def test_build_prompt_unknown_confidant() -> None:
    with pytest.raises(KeyError):
        build_prompt(confidant_id=999, rank=1, context="test")
