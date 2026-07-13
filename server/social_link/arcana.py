"""Maps P5R confidant IDs to character names and arcana titles."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Confidant:
    id: int
    name: str
    arcana: str
    personality_blurb: str


CONFIDANTS: dict[int, Confidant] = {
    0:  Confidant(0,  "Igor",           "Fool",        "Mysterious velvet room attendant; cryptic and formal."),
    1:  Confidant(1,  "Ryuji Sakamoto", "Chariot",     "Loud, loyal, hot-headed best friend; talks in street slang."),
    2:  Confidant(2,  "Ann Takamaki",   "Lovers",      "Empathetic, fashionable; grapples with identity and self-worth."),
    3:  Confidant(3,  "Yusuke Kitagawa","Emperor",     "Eccentric artist; speaks in lofty, aesthetic metaphors."),
    4:  Confidant(4,  "Makoto Niijima", "Priestess",   "Studious student council president; measured, analytical."),
    5:  Confidant(5,  "Futaba Sakura",  "Hermit",      "Hikikomori tech genius; uses gamer/internet slang."),
    6:  Confidant(6,  "Haru Okumura",   "Empress",     "Gentle, privileged; hides inner strength behind politeness."),
    7:  Confidant(7,  "Goro Akechi",    "Justice",     "Charming detective; carefully controlled public persona."),
    8:  Confidant(8,  "Sojiro Sakura",  "Hierophant",  "Gruff but caring guardian; old-school pragmatist."),
    9:  Confidant(9,  "Tae Takemi",     "Death",       "Sardonic underground doctor; dry dark humor."),
    10: Confidant(10, "Yuuki Mishima",  "Moon",        "Insecure but earnest; tries too hard to fit in."),
}


def get_confidant(confidant_id: int) -> Confidant:
    if confidant_id not in CONFIDANTS:
        raise KeyError(f"Unknown confidant ID: {confidant_id}")
    return CONFIDANTS[confidant_id]
