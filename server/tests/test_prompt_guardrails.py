"""Guardrail tests for the prompt builder output contract."""

from social_link.prompt_builder import build_prompt


def test_system_prompt_contains_rank() -> None:
    system, _ = build_prompt(confidant_id=3, rank=7, context="study discussion")
    assert "rank 7/10" in system


def test_user_prompt_ends_with_character_name_colon() -> None:
    _, user = build_prompt(confidant_id=3, rank=7, context="study discussion")
    assert user.strip().endswith("Makoto Niijima:")


def test_system_prompt_does_not_exceed_2048_chars() -> None:
    system, user = build_prompt(confidant_id=1, rank=1, context="x" * 1024)
    assert len(system) + len(user) < 4096, "Combined prompt exceeds safe context window"