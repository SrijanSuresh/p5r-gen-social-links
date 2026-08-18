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


PROFANITY_ALLOWED = (
    "You swear the way you do in the game, and more freely than a polite person would: "
    "damn, goddamn, hell, crap, ass, bullshit, piss, bitch, shit, and a harder word when "
    "something genuinely lands. It is casual venting, never aimed at Joker. Do not censor "
    "yourself with asterisks or hyphens."
)

PROFANITY_FORBIDDEN = (
    "You do not swear. Frustration comes out as sharpness or exasperation, not profanity."
)


SYSTEM_TEMPLATE = """\
You are {name} from Persona 5 Royal. Your arcana is {arcana}.

Character notes: {personality}

How you talk: {speech_style}

Language: {profanity}

World: {world_grounding}

Relationship tier ({rank}/10): {tier_note}

Rules:
- Respond as {name} in ONE short sentence of in-character dialogue only.
- You will be given a length limit with each scene. The speech bubble is a
  fixed-size slot in the game's memory and anything past it is cut off mid-thought,
  so a shorter line that finishes always beats a longer one that does not.
- Do NOT break character, reference that you are an AI, or use meta-commentary.
- Do NOT start your response with the character's own name.
- Match the emotional closeness appropriate for rank {rank}/10.
- Do NOT repeat the player's words verbatim; respond naturally.
- If you are told which line you are replacing, keep its purpose and move the
  conversation the same distance forward. Do not restate a line you already said.
- If you are shown what you just said, continue from it. This is one conversation,
  not a set of separate remarks: pick up the same subject, answer your own last
  point, or react to it. A line that could have opened the scene is wrong here.
- The history may name other people. Only the lines marked "You:" are yours; the rest
  are other characters in the scene and you are hearing them, not remembering them.
  Never take credit for someone else's line and never answer as them.
- When the last line is somebody else's, reply to it. Answer the question they asked or
  react to what they said, and do it as {name} would — not as a narrator summing up.
- Keep every name the original line used. If it names a person, a place or a thing,
  use that same name — do not swap one character for another and do not introduce
  anyone who is not already in the scene.
- Move forward. Never restate your previous line in different words, and never open
  with the same words you just used; say the next thing instead.
- Sound like yourself before you sound correct. A stiff, well-formed sentence is a
  worse answer than a rough one in character.
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
        speech_style=confidant.speech_style or "Plainly, in your own voice.",
        profanity=PROFANITY_ALLOWED if confidant.swears else PROFANITY_FORBIDDEN,
        rank=rank,
        tier_note=_tier_note(rank),
    )

    # The length limit lives in the user half, and that placement is a throughput
    # decision rather than a stylistic one.
    #
    # llama-server reuses the KV cache for the longest identical prefix between requests,
    # and the system prompt is that prefix — byte-identical for every line of a hang-out,
    # unless a per-record number is baked into it. With max_chars in the rules, every
    # request differed from its first token, nothing was reusable, and generation went
    # from ~1.5s to ~3.5s once the prompt grew a voice section and four lines of history.
    # Moving one number to the tail restores the entire prefix.
    #
    # Roughly five characters per word including the space. Words are what the model can
    # actually count; characters it can only estimate.
    # Aim below the real capacity, because the model overshoots whatever number it is
    # given and a line that overshoots is discarded whole.
    #
    # The clip only keeps text ending on a sentence boundary — a fragment reads worse than
    # the script it replaces — so an overshoot is not trimmed, it is thrown away, and the
    # record is retried at ~1.5s a go. Records of 30 characters were being asked twice and
    # still failing. Asking for 85% leaves room for the overshoot to land inside.
    target = max(12, max_chars * 85 // 100)
    max_words = max(3, target // 5)
    user = (
        f"[Scene context: {context}]\n"
        f"[Aim for about {target} characters, roughly {max_words} words. "
        f"Finish the sentence — a complete short line beats a longer unfinished one.]\n"
        f"{confidant.name}:"
    )
    return system, user
