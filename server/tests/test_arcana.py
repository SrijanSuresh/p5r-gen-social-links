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


def test_takemi_id_confirmed_from_game_memory() -> None:
    """ID 14 = Tae Takemi — confirmed via live ConfidantId read during hang-out."""
    takemi = get_confidant(14)
    assert takemi.name == "Tae Takemi"
    assert takemi.arcana == "Death"


def test_justine_caroline_at_id_12() -> None:
    """ID 12 was missing from the original mapping, causing all IDs 13+ to be wrong."""
    twins = get_confidant(12)
    assert twins.name == "Justine and Caroline"
    assert twins.arcana == "Strength"


def test_iwai_shifted_to_13() -> None:
    iwai = get_confidant(13)
    assert iwai.name == "Munehisa Iwai"


def test_lavenza_at_22() -> None:
    lavenza = get_confidant(22)
    assert lavenza.name == "Lavenza"
    assert lavenza.arcana == "World"


def test_get_confidant_raises_on_unknown_id() -> None:
    with pytest.raises(KeyError):
        get_confidant(99)


def test_p5r_exclusive_kasumi_provisional() -> None:
    """Kasumi ID is provisional (25) — verify by checking mod console during her hang-out."""
    kasumi = get_confidant(25)
    assert kasumi.name == "Kasumi Yoshizawa"
    assert kasumi.arcana == "Faith"


def test_p5r_exclusive_maruki_provisional() -> None:
    """Maruki ID is provisional (26) — verify by checking mod console during his hang-out."""
    maruki = get_confidant(26)
    assert maruki.name == "Takuto Maruki"
    assert maruki.arcana == "Councillor"


def test_full_confidant_count() -> None:
    # IDs 1–22 confirmed + 2 provisional P5R exclusives (25, 26) = 24 total
    assert len(CONFIDANTS) == 24


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
    for cid, c in CONFIDANTS.items():
        assert c.arcana[0].isupper(), f"Arcana '{c.arcana}' for confidant {cid} not title-case"


def test_personality_blurbs_end_with_period_or_not_empty() -> None:
    for cid, c in CONFIDANTS.items():
        assert len(c.personality_blurb.strip()) > 10, f"Blurb too short for confidant {cid}"


def test_all_confidant_names_unique() -> None:
    names = [c.name for c in CONFIDANTS.values()]
    assert len(names) == len(set(names)), "Duplicate confidant names found"


def test_no_id_gap_in_base_roster() -> None:
    """IDs 1–22 must all be present — no gaps in the confirmed range."""
    for expected_id in range(1, 23):
        assert expected_id in CONFIDANTS, f"Missing confirmed confidant ID {expected_id}"
