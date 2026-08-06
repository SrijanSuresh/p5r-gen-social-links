"""Tests verifying that build_prompt() produces correctly structured prompts."""

from __future__ import annotations

import pytest

from social_link.prompt_builder import build_prompt, _tier_note, SYSTEM_TEMPLATE


RYUJI_ID  = 8
MAKOTO_ID = 3
KASUMI_ID = 22


def _system(confidant_id: int, rank: int, context: str = "gym scene") -> str:
    return build_prompt(confidant_id, rank, context)[0]


def _user(confidant_id: int, rank: int, context: str = "gym scene") -> str:
    return build_prompt(confidant_id, rank, context)[1]


def test_system_prompt_contains_character_name() -> None:
    system = _system(RYUJI_ID, 4)
    assert "Ryuji" in system


def test_system_prompt_contains_arcana() -> None:
    system = _system(RYUJI_ID, 4)
    assert "Chariot" in system


def test_system_prompt_contains_rank() -> None:
    system = _system(RYUJI_ID, 7)
    assert "7/10" in system or "7" in system


def test_system_prompt_contains_tier_note() -> None:
    system = _system(RYUJI_ID, 3)
    assert "Acquaintances" in system or "polite" in system or "Close" in system


def test_user_prompt_starts_with_scene_context() -> None:
    user = _user(RYUJI_ID, 4, "gym training")
    assert user.startswith("[Scene context: gym training]")


def test_user_prompt_ends_with_character_colon() -> None:
    user = _user(RYUJI_ID, 4)
    assert user.endswith("Ryuji Sakamoto:")


def test_user_prompt_context_is_embedded() -> None:
    context = "fighting shadows in Kamoshida palace"
    user = _user(RYUJI_ID, 4, context)
    assert context in user


def test_system_has_no_name_rule() -> None:
    system = _system(RYUJI_ID, 5)
    assert "Do NOT start your response with the character" in system


def test_build_prompt_returns_two_strings() -> None:
    result = build_prompt(RYUJI_ID, 5, "context")
    assert len(result) == 2
    assert all(isinstance(s, str) for s in result)


def test_kasumi_prompt_uses_correct_name() -> None:
    system, user = build_prompt(KASUMI_ID, 3, "ballet practice")
    assert "Kasumi" in system
    assert "Kasumi" in user


def test_tier_note_boundaries() -> None:
    assert "just met" in _tier_note(1).lower() or "polite" in _tier_note(1).lower()
    assert "Acquaintances" in _tier_note(3) or "warming" in _tier_note(3)
    assert "Close" in _tier_note(6) or "friends" in _tier_note(6).lower()
    assert "Deepest" in _tier_note(9) or "trust" in _tier_note(9).lower()


def test_makoto_prompt_contains_arcana() -> None:
    system = _system(MAKOTO_ID, 6)
    assert "Priestess" in system
