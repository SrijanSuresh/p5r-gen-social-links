"""Maps P5R confidant IDs to character names and arcana titles.

IDs confirmed from live game memory reads (ConfidantId field in CMM session struct).
Ryuji=8 and Takemi=14 verified via mod console output. Full list cross-referenced
against the user-provided P5R internal ID table.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Confidant:
    id: int
    name: str
    arcana: str
    personality_blurb: str

    # Active member of the Phantom Thieves — fights alongside Joker in the Metaverse.
    is_phantom_thief: bool = False

    # Knows Joker leads the Phantom Thieves. True for every member, plus the few
    # non-members let in on it (Sojiro, Sae, Mishima) and the Velvet Room staff.
    #
    # Why this exists: the model knows Persona 5 well enough to bring the Phantom
    # Thieves up unprompted, but not well enough to place the speaker correctly — the
    # first real generation had Ryuji proposing to "take down those Phantom Thieves",
    # a group he leads with Joker. Grounding has to be conditional, though: most
    # confidants must never discuss it, so telling everyone about the team would
    # trade one immersion break for a worse one.
    #
    # Ambiguous cases (Kasumi and Maruki, who learn late; Akechi, whose knowledge is
    # the plot) default to the conservative value: a character who stays quiet about
    # the Phantom Thieves is never wrong, while one who speaks up at the wrong time is.
    knows_identity: bool = False

    # How this character actually talks: rhythm, filler, address terms, verbal tics.
    #
    # personality_blurb says who someone is; this says how they sound, and the model needs
    # both. Given only "loud, loyal, hot-headed best friend", it produced "Guess I'm cool
    # with paying for the session if that's what keeps this place running." — accurate to
    # the description and completely wrong for the character.
    speech_style: str = ""

    # Whether this character swears in the localisation. Ryuji does, constantly; Makoto
    # does not. Applying one profanity setting to everyone would break more characters
    # than it fixed, so it lives per confidant.
    swears: bool = False


CONFIDANTS: dict[int, Confidant] = {
    # ── Confirmed IDs (verified from live game memory or user-provided ID table) ──
    1:  Confidant(1,  "Igor",                  "Fool",        "Mysterious velvet room attendant; cryptic and formal.", False, True),
    2:  Confidant(2,  "Morgana",               "Magician",    "Sarcastic cat/human; self-appointed leader, secretly insecure.", True, True),
    3:  Confidant(3,  "Makoto Niijima",        "Priestess",   "Studious student council president; measured, analytical.", True, True),
    4:  Confidant(4,  "Haru Okumura",          "Empress",     "Gentle, privileged; hides inner strength behind politeness.", True, True),
    5:  Confidant(5,  "Yusuke Kitagawa",       "Emperor",     "Eccentric artist; speaks in lofty, aesthetic metaphors.", True, True),
    6:  Confidant(6,  "Sojiro Sakura",         "Hierophant",  "Gruff but caring guardian; old-school pragmatist.", False, True),
    7:  Confidant(7,  "Ann Takamaki",          "Lovers",      "Empathetic, fashionable; grapples with identity and self-worth.", True, True),
    8:  Confidant(8,  "Ryuji Sakamoto",        "Chariot",     "Loud, loyal, hot-headed best friend; talks in street slang.", True, True,
                   speech_style=(
                       "Drops the g from -ing (talkin', goin', effin'). Calls Joker 'bro', "
                       "'dude' or 'man'. Opens with 'Yo', 'Hell yeah', 'For real?', "
                       "'Dude!', 'Aw, c'mon'. Uses 'ain't', 'gonna', 'gotta', 'lemme', "
                       "''cause', 'sh---y'. Blunt and loud, short bursts rather than "
                       "clauses, enthusiasm over precision. Never formal, never polite, "
                       "never uses a word like 'therefore' or 'regarding'."
                   ),
                   swears=True),
    9:  Confidant(9,  "Goro Akechi",           "Justice",     "Charming detective; carefully controlled public persona.", True, True),
    10: Confidant(10, "Futaba Sakura",         "Hermit",      "Hikikomori tech genius; uses gamer/internet slang.", True, True),
    11: Confidant(11, "Chihaya Mifune",        "Fortune",     "Earnest fortune teller struggling against a manipulative organization."),
    12: Confidant(12, "Justine and Caroline",  "Strength",    "Twin velvet room wardens; Justine cold and precise, Caroline loud and brash.", False, True),
    13: Confidant(13, "Munehisa Iwai",         "Hanged Man",  "Taciturn arms dealer with a hidden paternal side."),
    14: Confidant(14, "Tae Takemi",            "Death",       "Unconventional doctor; sardonic, tests experimental medicine on Joker."),
    15: Confidant(15, "Sadayo Kawakami",       "Temperance",  "Overworked homeroom teacher with a moonlighting secret."),
    16: Confidant(16, "Ichiko Ohya",           "Devil",       "Hard-drinking journalist doggedly chasing a story about the Phantom Thieves."),
    17: Confidant(17, "Shinya Oda",            "Tower",       "Competitive boy genius at Akihabara's arcade; hides deep insecurity."),
    18: Confidant(18, "Hifumi Togo",           "Star",        "Shogi prodigy; disciplined, learns to think beyond the board."),
    19: Confidant(19, "Yuuki Mishima",         "Moon",        "Insecure but earnest; tries too hard to fit in.", False, True),
    20: Confidant(20, "Toranosuke Yoshida",    "Sun",         "Disgraced politician working to rebuild trust through honest speech."),
    21: Confidant(21, "Sae Niijima",           "Judgement",   "Ruthless prosecutor who begins to question the justice system.", False, True),
    22: Confidant(22, "Lavenza",               "World",       "True form of the velvet room twins; earnest and grateful.", False, True),
    # ── P5R exclusives — IDs provisional, not yet verified via game memory ──────
    # Hang out with Kasumi/Maruki and check the mod console for ConfidantId to confirm.
    25: Confidant(25, "Kasumi Yoshizawa",      "Faith",       "Dedicated gymnast driven by guilt; hides a painful truth about herself."),
    26: Confidant(26, "Takuto Maruki",         "Councillor",  "Compassionate counsellor whose desire to help others becomes dangerous."),
}


def get_confidant(confidant_id: int) -> Confidant:
    if confidant_id not in CONFIDANTS:
        raise KeyError(f"Unknown confidant ID: {confidant_id}")
    return CONFIDANTS[confidant_id]
