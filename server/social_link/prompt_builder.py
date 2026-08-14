"""Builds character-faithful prompts for LLM inference."""

from .arcana import Confidant, get_confidant
from .tier import tier_note as _tier_note


TEAMMATE_GROUNDING = (
    "You are a member of the Phantom Thieves of Hearts, the secret group led by "
    "Joker — the person you are talking to. You steal corrupt hearts together and "
    "you trust him completely. Never speak of the Phantom Thieves as enemies, "
    "strangers, or a group you are outside of or hunting."
)

CONFIDANT_GROUNDING = (
    "You know that Joker — the person you are talking to — leads the Phantom Thieves "
    "of Hearts, and you keep that secret. You are not a member yourself. Never treat "
    "the Phantom Thieves as your enemies."
)

UNAWARE_GROUNDING = (
    "You do NOT know that Joker — the person you are talking to — is connected to the "
    "Phantom Thieves of Hearts. To you he is an ordinary high-school student. Only "
    "ever refer to the Phantom Thieves the way the public does, as rumour and news, "
    "and never imply he is involved."
)


def _world_grounding(confidant: "Confidant") -> str:
    """
    Place the speaker relative to the Phantom Thieves.

    The model has enough Persona 5 knowledge to raise the subject unprompted but not
    enough to place the speaker, and it guesses wrong: the first real generation had
    Ryuji suggesting they "take down those Phantom Thieves", a group he co-leads.
    Stating the relationship explicitly costs a few tokens and removes the guess.
    """
    if confidant.is_phantom_thief:
        return TEAMMATE_GROUNDING
    if confidant.knows_identity:
        return CONFIDANT_GROUNDING
    return UNAWARE_GROUNDING


SYSTEM_TEMPLATE = """\
You are {name} from Persona 5 Royal. Your arcana is {arcana}.

Character notes: {personality}

World: {world_grounding}

Relationship tier ({rank}/10): {tier_note}

Rules:
- Respond as {name} in ONE short sentence of in-character dialogue only.
- Keep it under {max_chars} characters, about {max_words} words. The speech bubble
  is a fixed-size slot in the game's memory and anything past that is cut off
  mid-thought, so a shorter line that finishes always beats a longer one that does not.
- Do NOT break character, reference that you are an AI, or use meta-commentary.
- Do NOT start your response with the character's own name.
- Match the emotional closeness appropriate for rank {rank}/10.
- Do NOT repeat the player's words verbatim; respond naturally.
"""


def build_prompt(
    confidant_id: int,
    rank: int,
    context: str,
    max_chars: int = 56,
) -> tuple[str, str]:
    """
    Return (system_prompt, user_prompt) for the LLM.

    ``max_chars`` is the capacity of the exact message record this line will be written
    into, so it changes from line to line. Stating it in the prompt is not a substitute
    for clipping the result — the model treats a length rule as a strong suggestion — but
    a model aiming at 30 characters overruns by a word, while one aiming at 56 overruns
    by a clause.
    """
    confidant: Confidant = get_confidant(confidant_id)
    system = SYSTEM_TEMPLATE.format(
        name=confidant.name,
        arcana=confidant.arcana,
        personality=confidant.personality_blurb,
        world_grounding=_world_grounding(confidant),
        rank=rank,
        tier_note=_tier_note(rank),
        max_chars=max_chars,
        # Roughly five characters per word including the space. Words are what the model
        # can actually count; characters it can only estimate.
        max_words=max(3, max_chars // 5),
    )
    user = f"[Scene context: {context}]\n{confidant.name}:"
    return system, user
