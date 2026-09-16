"""Who the line is spoken to, and where that fact is allowed to live (Ch. 81).

Attribution fixed whose voice a line is written in. These cover the other half: a
rank-up scene has a cast, and Takemi answering a patient's father must not be aimed at
the protagonist standing silently in the corner.
"""

from social_link.prompt_builder import (
    PROTAGONIST,
    _addressee_note,
    build_prompt,
)

# One confidant per grounding branch: Ryuji is a Phantom Thief, Sojiro knows and keeps
# the secret, Takemi does not know at all.
RYUJI, SOJIRO, TAKEMI = 8, 6, 14
FATHER = "Girl's Father"


# --- the note itself ----------------------------------------------------------


def test_addressing_the_protagonist_names_him_plainly() -> None:
    note = _addressee_note(PROTAGONIST, "Tae Takemi")
    assert note == "[You are speaking to Joker.]"


def test_addressing_the_cast_says_who_and_says_it_is_not_joker() -> None:
    note = _addressee_note(FATHER, "Tae Takemi")
    assert FATHER in note
    # Both halves are load-bearing. Without the first the line gets aimed at Joker;
    # without the second the model writes as though he had left the room.
    assert "not to Joker" in note
    assert "present and listening" in note


def test_an_unknown_addressee_produces_no_line_at_all() -> None:
    # Empty is not "Joker". The mod sends it when the turn demonstrably changed hands
    # to somebody the scene did not name, and defaulting to Joker there would reinstate
    # the assumption this whole feature removes.
    assert _addressee_note("", "Tae Takemi") == ""
    assert _addressee_note("   ", "Tae Takemi") == ""
    assert _addressee_note(None, "Tae Takemi") == ""  # type: ignore[arg-type]


def test_the_speaker_is_never_told_they_are_talking_to_themselves() -> None:
    # The turn model skips a speaker's own earlier bubbles rather than answering them,
    # so this should be unreachable; it is cheaper to be certain than to rely on that.
    assert _addressee_note("Tae Takemi", "Tae Takemi") == ""
    assert _addressee_note("tae takemi", "Tae Takemi") == ""


# --- placement: the system half has to stay byte-identical across a scene -----


def test_the_addressee_never_touches_the_system_prompt() -> None:
    """
    llama-server reuses the KV cache for the longest identical prefix between requests,
    and the system prompt is that prefix. Baking max_chars into it cost ~2s per line
    (Ch. 72); a three-party scene changes addressee often enough to cost the same way.
    """
    to_joker, _ = build_prompt(TAKEMI, 5, "clinic", addressee=PROTAGONIST)
    to_father, _ = build_prompt(TAKEMI, 5, "clinic", addressee=FATHER)
    unknown, _ = build_prompt(TAKEMI, 5, "clinic", addressee="")

    assert to_joker == to_father == unknown


def test_the_addressee_appears_in_the_user_half() -> None:
    _, user = build_prompt(TAKEMI, 5, "clinic", addressee=FATHER)
    assert FATHER in user


def test_the_addressee_is_stated_before_the_length_rule() -> None:
    # Who you are talking to changes what the line is for; the budget only changes how
    # long it runs. The model weights the earlier instruction more heavily.
    _, user = build_prompt(TAKEMI, 5, "clinic", addressee=FATHER)
    assert user.index(FATHER) < user.index("Aim for about")


def test_an_unknown_addressee_leaves_the_user_half_as_it_was() -> None:
    _, with_unknown = build_prompt(TAKEMI, 5, "clinic", addressee="")
    _, without_arg = build_prompt(TAKEMI, 5, "clinic")
    assert with_unknown == without_arg


# --- the world grounding must survive being made addressee-neutral ------------


def test_world_grounding_no_longer_asserts_who_is_listening() -> None:
    # The phrase welded two facts together: what Joker permanently is, and who happens
    # to be hearing this line. Only the first belongs in a cached prefix.
    for confidant_id in (RYUJI, SOJIRO, TAKEMI):
        system, _ = build_prompt(confidant_id, 5, "scene")
        assert "person you are talking to" not in system


def test_world_grounding_still_places_the_speaker() -> None:
    # Neutralising the phrasing must not cost the fact it was carrying, which is the
    # thing that stopped Ryuji proposing to take down the Phantom Thieves.
    ryuji, _ = build_prompt(RYUJI, 5, "gym")
    assert "member of the Phantom Thieves" in ryuji
    assert "led by Joker" in ryuji

    sojiro, _ = build_prompt(SOJIRO, 5, "Leblanc")
    assert "Joker leads the Phantom Thieves" in sojiro
    assert "not a member yourself" in sojiro

    takemi, _ = build_prompt(TAKEMI, 5, "clinic")
    assert "do NOT know that Joker is connected" in takemi
    assert "ordinary high-school student" in takemi


def test_the_rules_tell_the_model_an_addressee_is_coming() -> None:
    system, _ = build_prompt(TAKEMI, 5, "clinic")
    assert "told who you are speaking to" in system
