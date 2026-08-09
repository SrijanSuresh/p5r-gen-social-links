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


def test_p5r_exclusive_kasumi() -> None:
    kasumi = get_confidant(22)
    assert kasumi.name == "Kasumi Yoshizawa"
    assert kasumi.arcana == "Faith"
    assert len(kasumi.personality_blurb) > 0


def test_p5r_exclusive_maruki() -> None:
    maruki = get_confidant(23)
    assert maruki.name == "Takuto Maruki"
    assert maruki.arcana == "Councillor"


def test_full_confidant_count() -> None:
    # 20 base confidants + 2 P5R exclusives (Kasumi=22, Maruki=23) = 22 total
    assert len(CONFIDANTS) == 22


def test_protagonist_joker_is_not_in_roster() -> None:
    """Joker (the protagonist) is not a confidant — the roster is NPCs only."""
    for c in CONFIDANTS.values():
        assert "Joker" not in c.name
        assert "Akira" not in c.name
        assert "Ren Amamiya" not in c.name


def test_confidant_ids_are_positive_ints() -> None:
    for cid in CONFIDANTS:
        assert isinstance(cid, int)
        assert cid > 0


def test_arcana_strings_are_title_case() -> None:
    """Arcana names should be properly capitalised for display in the LLM prompt."""
    for cid, c in CONFIDANTS.items():
        assert c.arcana[0].isupper(), f"Arcana '{c.arcana}' for confidant {cid} not title-case"


def test_personality_blurbs_end_with_period_or_not_empty() -> None:
    """Blurbs should be non-empty; we don't enforce period but they must have content."""
    for cid, c in CONFIDANTS.items():
        assert len(c.personality_blurb.strip()) > 10, f"Blurb too short for confidant {cid}"


def test_all_confidant_names_unique() -> None:
    names = [c.name for c in CONFIDANTS.values()]
    assert len(names) == len(set(names)), "Duplicate confidant names found"