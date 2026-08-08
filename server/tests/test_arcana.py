"""Full coverage test for every defined confidant in arcana.py."""

import pytest
from social_link.arcana import get_confidant, CONFIDANTS


def test_all_confidants_have_required_fields() -> None:
    for cid, c in CONFIDANTS.items():
        assert c.id == cid, f"ID mismatch for confidant {cid}"
        assert c.name, f"Empty name for confidant {cid}"
        assert c.arcana, f"Empty arcana for confidant {cid}"
        assert c.personality_blurb, f"Empty blurb for confidant {cid}"


def test_get_confidant_returns_correct_instance() -> None:
    ryuji = get_confidant(8)
    assert ryuji.name == "Ryuji Sakamoto"
    assert ryuji.arcana == "Chariot"


def test_get_confidant_raises_on_unknown_id() -> None:
    with pytest.raises(KeyError):
        get_confidant(99)