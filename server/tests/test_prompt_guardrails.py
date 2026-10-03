"""Guardrail tests for the prompt builder output contract."""

from inference.config import ModelConfig
from social_link.prompt_builder import build_prompt


def test_system_prompt_contains_rank() -> None:
    system, _ = build_prompt(confidant_id=3, rank=7, context="study discussion")
    assert "rank 7/10" in system


def test_user_prompt_ends_with_character_name_colon() -> None:
    _, user = build_prompt(confidant_id=3, rank=7, context="study discussion")
    assert user.strip().endswith("Makoto Niijima:")


# Chars per token for English, conservatively. The real figure is nearer 4; 3 leaves the
# bound on the safe side of a prompt full of names and punctuation, which tokenize worse.
CHARS_PER_TOKEN = 3


def test_the_prompt_leaves_room_to_generate_inside_n_ctx() -> None:
    """
    Bound the prompt by the window it actually has to fit in.

    This used to assert a flat `< 4096` characters, which is a round number rather than a
    constraint, and the prompt had grown to within 114 characters of it — so the next
    feature to add a rule would have failed the check whatever that rule was. The limit
    that exists is `ModelConfig.n_ctx`, minus the tokens reserved for the answer.

    Deriving it from the config also means changing `n_ctx` moves this automatically,
    instead of leaving a magic number that used to be right.
    """
    cfg = ModelConfig()
    budget_chars = (cfg.n_ctx - cfg.max_tokens) * CHARS_PER_TOKEN

    # A 1024-character context is the most Pydantic will accept on /generate, so this is
    # the largest prompt the server can be asked to build.
    system, user = build_prompt(confidant_id=1, rank=1, context="x" * 1024)
    total = len(system) + len(user)

    assert total < budget_chars, (
        f"prompt is {total} chars; n_ctx={cfg.n_ctx} leaves room for about "
        f"{budget_chars} once {cfg.max_tokens} tokens are reserved for the answer"
    )


def test_the_system_half_stays_well_inside_the_window_on_its_own() -> None:
    # The system prompt is the KV-cache prefix, so its length is paid once per scene
    # rather than once per line (Ch. 81). It is still the half that grows every time a
    # rule is added, and this is the line that will notice when it has gone too far.
    cfg = ModelConfig()
    system, _ = build_prompt(confidant_id=1, rank=1, context="scene")

    half_the_window = cfg.n_ctx * CHARS_PER_TOKEN // 2
    assert len(system) < half_the_window, (
        f"system prompt is {len(system)} chars, over half the {cfg.n_ctx}-token window"
    )

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


# --- multi-speaker scenes -----------------------------------------------------
#
# A rank-up is not a monologue. Takemi's involves a patient and her father, and the mod
# now names them in the history it sends. Before that the model was told every earlier
# line was the confidant's own, so it took credit for the father's lines and then
# answered itself.


def test_only_lines_marked_as_yours_are_the_speakers() -> None:
    system, _ = build_prompt(TAKEMI, 4, "clinic")
    assert '"You:"' in system


def test_told_not_to_answer_as_another_character() -> None:
    system, _ = build_prompt(TAKEMI, 4, "clinic")
    assert "never answer as them" in system.lower()


def test_told_to_reply_when_the_last_line_was_someone_elses() -> None:
    system, _ = build_prompt(TAKEMI, 4, "clinic")
    assert "when the last line is somebody else's, reply to it" in system.lower()


def test_the_reply_rule_names_the_character() -> None:
    """The rule interpolates {name}; a stray brace would ship the literal placeholder."""
    system, _ = build_prompt(TAKEMI, 4, "clinic")
    assert "as Tae Takemi would" in system
    assert "{name}" not in system


def test_multi_speaker_rules_reach_every_confidant() -> None:
    from social_link.arcana import CONFIDANTS

    for confidant_id in CONFIDANTS:
        system, _ = build_prompt(confidant_id, 3, "scene")
        assert "you are hearing them, not remembering them" in system
