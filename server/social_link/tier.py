"""Rank-to-emotional-tier mapping for Social Link prompt construction."""

from __future__ import annotations


def tier_note(rank: int) -> str:
    """Return relationship-tier guidance text for the given Social Link rank (1-10)."""
    if rank <= 2:
        return "You have just met. Keep tone polite but reserved; do not use pet names."
    if rank <= 5:
        return "Acquaintances warming up. Casual, friendly; some shared history implied."
    if rank <= 8:
        return "Close friends. Comfortable banter, genuine emotional investment."
    return "Deepest bond. Speak with trust, vulnerability, and warmth."


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
