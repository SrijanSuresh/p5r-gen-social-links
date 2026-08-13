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

# --- Phantom Thieves grounding ------------------------------------------------
#
# The model raises the Phantom Thieves unprompted but misplaces the speaker: the
# first real generation had Ryuji proposing to "take down those Phantom Thieves".
# These lock in that each confidant is positioned correctly, and — more importantly —
# that confidants who must not know are never told.

RYUJI, MAKOTO, MORGANA = 8, 3, 2
TAKEMI, KAWAKAMI, OHYA, IWAI = 14, 15, 16, 13
SOJIRO, SAE, MISHIMA = 6, 21, 19


def test_teammate_is_told_they_are_a_phantom_thief() -> None:
    system, _ = build_prompt(RYUJI, 4, "at the gym")
    assert "member of the Phantom Thieves" in system


def test_teammate_is_told_not_to_treat_the_team_as_enemies() -> None:
    """The exact defect observed in the first real generation."""
    system, _ = build_prompt(RYUJI, 4, "at the gym")
    assert "never speak of the phantom thieves as enemies" in system.lower()


def test_all_teammates_get_teammate_grounding() -> None:
    for confidant_id in (RYUJI, MAKOTO, MORGANA):
        system, _ = build_prompt(confidant_id, 5, "scene")
        assert "member of the Phantom Thieves" in system, confidant_id


def test_unaware_confidant_is_told_joker_is_an_ordinary_student() -> None:
    system, _ = build_prompt(TAKEMI, 4, "at the clinic")
    assert "ordinary high-school student" in system


def test_unaware_confidants_are_never_told_joker_is_a_thief() -> None:
    """
    The inverse bug is worse than the one being fixed: Takemi or Kawakami casually
    discussing Phantom Thief business breaks the story outright.
    """
    for confidant_id in (TAKEMI, KAWAKAMI, OHYA, IWAI):
        system, _ = build_prompt(confidant_id, 5, "scene")
        assert "You do NOT know" in system, confidant_id
        assert "member of the Phantom Thieves" not in system, confidant_id


def test_aware_non_member_keeps_the_secret_without_joining() -> None:
    for confidant_id in (SOJIRO, SAE, MISHIMA):
        system, _ = build_prompt(confidant_id, 5, "scene")
        assert "keep that secret" in system, confidant_id
        assert "You are not a member" in system, confidant_id


def test_every_confidant_gets_exactly_one_grounding() -> None:
    """No confidant may be left unplaced, and none may receive contradictory lines."""
    from social_link.arcana import CONFIDANTS
    from social_link.prompt_builder import (
        CONFIDANT_GROUNDING,
        TEAMMATE_GROUNDING,
        UNAWARE_GROUNDING,
    )

    for confidant_id in CONFIDANTS:
        system, _ = build_prompt(confidant_id, 5, "scene")
        present = [
            g
            for g in (TEAMMATE_GROUNDING, CONFIDANT_GROUNDING, UNAWARE_GROUNDING)
            if g in system
        ]
        assert len(present) == 1, f"{confidant_id} got {len(present)} grounding blocks"
