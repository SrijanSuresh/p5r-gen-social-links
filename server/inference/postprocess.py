"""Post-process raw LLM output before it goes into the game buffer."""

from __future__ import annotations

import re

# Patterns that indicate the model broke character or added meta-commentary
_OOC_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\(OOC:.*?\)", re.IGNORECASE),
    re.compile(r"\[Note:.*?\]", re.IGNORECASE),
    re.compile(r"As an AI.*?[\.\!]", re.IGNORECASE),
    re.compile(r"I am an AI.*?[\.\!]", re.IGNORECASE),
]


def clean_response(raw: str, max_chars: int = 200) -> str:
    """
    Strip out-of-character commentary and truncate to buffer limit.
    Returns the cleaned string, or empty string if nothing remains.
    """
    text = raw.strip()

    for pattern in _OOC_PATTERNS:
        text = pattern.sub("", text)

    # Collapse multiple spaces left by removals
    text = re.sub(r"  +", " ", text).strip()

    return text[:max_chars]