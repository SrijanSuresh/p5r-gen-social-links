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
    1:  Confidant(1,  "Igor",              "Fool",        "Mysterious velvet room attendant; cryptic and formal."),
    2:  Confidant(2,  "Morgana",           "Magician",    "Sarcastic cat/human; self-appointed leader, secretly insecure."),
    3:  Confidant(3,  "Makoto Niijima",    "Priestess",   "Studious student council president; measured, analytical."),
    4:  Confidant(4,  "Haru Okumura",      "Empress",     "Gentle, privileged; hides inner strength behind politeness."),
    5:  Confidant(5,  "Yusuke Kitagawa",   "Emperor",     "Eccentric artist; speaks in lofty, aesthetic metaphors."),
    6:  Confidant(6,  "Sojiro Sakura",     "Hierophant",  "Gruff but caring guardian; old-school pragmatist."),
    7:  Confidant(7,  "Ann Takamaki",      "Lovers",      "Empathetic, fashionable; grapples with identity and self-worth."),
    8:  Confidant(8,  "Ryuji Sakamoto",    "Chariot",     "Loud, loyal, hot-headed best friend; talks in street slang."),
    9:  Confidant(9,  "Goro Akechi",       "Justice",     "Charming detective; carefully controlled public persona."),
    10: Confidant(10, "Futaba Sakura",     "Hermit",      "Hikikomori tech genius; uses gamer/internet slang."),
    11: Confidant(11, "Chihaya Mifune",    "Fortune",     "Earnest fortune teller struggling against a manipulative organization."),
    12: Confidant(12, "Munehisa Iwai",     "Hanged Man",  "Taciturn arms dealer with a hidden paternal side."),
    13: Confidant(13, "Tae Takemi",        "Death",       "Unconventional doctor; sardonic, tests experimental medicine on Joker."),
    14: Confidant(14, "Sadayo Kawakami",   "Temperance",  "Overworked homeroom teacher with a moonlighting secret."),
    15: Confidant(15, "Ichiko Ohya",       "Devil",       "Hard-drinking journalist doggedly chasing a story about the Phantom Thieves."),
    16: Confidant(16, "Shinya Oda",        "Tower",       "Competitive boy genius at Akihabara's arcade; hides deep insecurity."),
    17: Confidant(17, "Hifumi Togo",       "Star",        "Shogi prodigy; disciplined, learns to think beyond the board."),
    18: Confidant(18, "Yuuki Mishima",     "Moon",        "Insecure but earnest; tries too hard to fit in."),
    19: Confidant(19, "Toranosuke Yoshida","Sun",         "Disgraced politician working to rebuild trust through honest speech."),
    20: Confidant(20, "Sae Niijima",       "Judgement",   "Ruthless prosecutor who begins to question the justice system."),
    # P5R exclusive: Kasumi Yoshizawa
    22: Confidant(22, "Kasumi Yoshizawa",  "Faith",       "Dedicated gymnast driven by guilt; hides a painful truth about herself."),
    # P5R exclusive: Dr. Maruki
    23: Confidant(23, "Takuto Maruki",     "Councillor",  "Compassionate counsellor whose desire to help others becomes dangerous."),
}


def get_confidant(confidant_id: int) -> Confidant:
    if confidant_id not in CONFIDANTS:
        raise KeyError(f"Unknown confidant ID: {confidant_id}")
    return CONFIDANTS[confidant_id]
