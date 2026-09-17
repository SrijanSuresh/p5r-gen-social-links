"""Rank-to-emotional-tier mapping for Social Link prompt construction."""

from __future__ import annotations


def tier_note(rank: int) -> str:
    """
    Relationship-tier guidance for a Social Link rank (1-10).

    Every line names Joker, and that is not decoration. The rank measures the confidant's
    bond with the protagonist and nobody else, but the note sits in a prompt that may be
    aimed at a third party: a rank-up scene has a cast, and "Close friends, comfortable
    banter" read as advice about the listener had Takemi bantering with a sick child's
    father on their first meeting (Ch. 81).

    Naming the subject costs two words and makes the note true whoever is being addressed.
    """
    if rank <= 2:
        return ("You and Joker have just met. Keep tone polite but reserved; "
                "do not use pet names for him.")
    if rank <= 5:
        return ("You and Joker are acquaintances warming up. Casual and friendly "
                "with him; some shared history implied.")
    if rank <= 8:
        return ("You and Joker are close friends. Comfortable banter with him, "
                "genuine emotional investment.")
    return ("You and Joker share your deepest bond. Speak to him with trust, "
            "vulnerability, and warmth.")


# Tier label strings for logging / display
TIER_LABELS: dict[str, range] = {
    "stranger":      range(1, 3),
    "acquaintance":  range(3, 6),
    "close_friend":  range(6, 9),
    "deepest_bond":  range(9, 11),
}


def tier_label(rank: int) -> str:
    """Return a short label for the relationship tier at this rank."""
    for label, r in TIER_LABELS.items():
        if rank in r:
            return label
    return "unknown"
