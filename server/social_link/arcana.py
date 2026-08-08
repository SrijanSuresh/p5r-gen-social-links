"""Maps P5R confidant IDs to character names and arcana titles."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Confidant:
    id: int
    name: str
    arcana: str
    personality_blurb: str


CONFIDANTS: dict[int, Confidant] = {
    # IDs are the game's internal 1-indexed arcana numbers (confirmed via Amicitia wiki).
    1:  Confidant(1,  "Igor",           "Fool",        "Mysterious velvet room attendant; cryptic and formal."),
    2:  Confidant(2,  "Morgana",        "Magician",    "Sarcastic cat/human; self-appointed leader, secretly insecure."),
    3:  Confidant(3,  "Makoto Niijima", "Priestess",   "Studious student council president; measured, analytical."),
    4:  Confidant(4,  "Haru Okumura",   "Empress",     "Gentle, privileged; hides inner strength behind politeness."),
    5:  Confidant(5,  "Yusuke Kitagawa","Emperor",     "Eccentric artist; speaks in lofty, aesthetic metaphors."),
    6:  Confidant(6,  "Sojiro Sakura",  "Hierophant",  "Gruff but caring guardian; old-school pragmatist."),
    7:  Confidant(7,  "Ann Takamaki",   "Lovers",      "Empathetic, fashionable; grapples with identity and self-worth."),
    8:  Confidant(8,  "Ryuji Sakamoto", "Chariot",     "Loud, loyal, hot-headed best friend; talks in street slang."),
    9:  Confidant(9,  "Goro Akechi",    "Justice",     "Charming detective; carefully controlled public persona."),
    10: Confidant(10, "Futaba Sakura",  "Hermit",      "Hikikomori tech genius; uses gamer/internet slang."),
    18: Confidant(18, "Yuuki Mishima",  "Moon",        "Insecure but earnest; tries too hard to fit in."),
}


def get_confidant(confidant_id: int) -> Confidant:
    if confidant_id not in CONFIDANTS:
        raise KeyError(f"Unknown confidant ID: {confidant_id}")
    return CONFIDANTS[confidant_id]
