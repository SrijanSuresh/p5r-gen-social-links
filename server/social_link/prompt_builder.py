"""Builds character-faithful prompts for LLM inference."""

from .arcana import Confidant, get_confidant
from .tier import tier_note as _tier_note


SYSTEM_TEMPLATE = """\
You are {name} from Persona 5 Royal. Your arcana is {arcana}.

Character notes: {personality}

Relationship tier ({rank}/10): {tier_note}

Rules:
- Respond as {name} in 1-2 sentences of in-character dialogue only.
- Do NOT break character, reference that you are an AI, or use meta-commentary.
- Do NOT start your response with the character's own name.
- Match the emotional closeness appropriate for rank {rank}/10.
- Do NOT repeat the player's words verbatim; respond naturally.
"""


def build_prompt(
    confidant_id: int,
    rank: int,
    context: str,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the LLM."""
    confidant: Confidant = get_confidant(confidant_id)
    system = SYSTEM_TEMPLATE.format(
        name=confidant.name,
        arcana=confidant.arcana,
        personality=confidant.personality_blurb,
        rank=rank,
        tier_note=_tier_note(rank),
    )
    user = f"[Scene context: {context}]\n{confidant.name}:"
    return system, user
