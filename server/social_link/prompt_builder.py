"""Builds character-faithful prompts for LLM inference."""

from .arcana import Confidant, get_confidant


SYSTEM_TEMPLATE = """\
You are {name} from Persona 5 Royal. Your arcana is {arcana}.

Character notes: {personality}

Rules:
- Respond as {name} in 1-3 sentences of in-character dialogue.
- Do NOT break character or reference that you are an AI.
- Match the emotional tone appropriate for Social Link rank {rank}/10.
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
    )
    user = f"[Scene context: {context}]\n{confidant.name}:"
    return system, user
