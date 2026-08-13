"""Post-process raw LLM output before it goes into the game buffer."""

from __future__ import annotations

import re

# Patterns that indicate the model broke character or added meta-commentary
_OOC_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\(OOC:.*?\)", re.IGNORECASE),
    re.compile(r"\[Note:.*?\]", re.IGNORECASE),
    re.compile(r"As an AI.*?[\.\!\?]", re.IGNORECASE),
    re.compile(r"I am an AI.*?[\.\!\?]", re.IGNORECASE),
    re.compile(r"I'm an AI.*?[\.\!\?]", re.IGNORECASE),
]

# "Name: " or "Name Surname: " prefix the model sometimes adds despite the rule
_NAME_PREFIX = re.compile(r"^[A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)?:\s*")

# Roleplay stage directions: *pumps fist*, *grins*. Observed in live output. The game
# renders the buffer literally, so these reach the speech bubble as asterisks.
_STAGE_DIRECTION = re.compile(r"\*[^*]{1,60}\*")

# The model wraps its line in quotes roughly half the time. P5R's bubble never shows
# them, and the inconsistency is more jarring than either choice — so they always go.
_WRAPPING_QUOTES = re.compile(r'^"(.*)"$', re.DOTALL)

# The mod writes the buffer with Encoding.ASCII, which turns any character above 0x7F
# into a literal '?'. Folding the punctuation an instruct model reaches for keeps a
# stray em-dash from surfacing in-game as "You've been slackin? off".
_ASCII_FOLD = str.maketrans({
    "‘": "'", "’": "'",              # single curly quotes
    "“": '"', "”": '"',              # double curly quotes
    "–": "-", "—": "-", "−": "-",  # en/em dash, minus
    "…": "...",                            # ellipsis
    " ": " ", " ": " ", " ": " ",  # non-breaking/thin spaces
    "•": "-",                              # bullet
    "´": "'", "ʼ": "'",              # acute accent, modifier apostrophe
})


def _truncate_at_sentence(text: str, max_chars: int) -> str:
    """Truncate at the last sentence boundary before max_chars."""
    if len(text) <= max_chars:
        return text
    chunk = text[:max_chars]
    # Walk backwards to find the last sentence-ending punctuation
    for i in range(len(chunk) - 1, -1, -1):
        if chunk[i] in ".!?":
            return chunk[: i + 1]
    # No sentence boundary found — fall back to word boundary
    last_space = chunk.rfind(" ")
    return chunk[:last_space] if last_space > 0 else chunk


def clean_response(raw: str, max_chars: int = 200) -> str:
    """
    Strip out-of-character commentary, remove name-prefix, and truncate at a
    sentence boundary. Returns the cleaned string, or empty string if nothing remains.
    """
    text = raw.strip()

    # Fold before anything else, so later patterns match on plain ASCII quotes.
    text = text.translate(_ASCII_FOLD)

    # Remove "CharacterName: " prefix if model ignored the rule
    text = _NAME_PREFIX.sub("", text, count=1)

    for pattern in _OOC_PATTERNS:
        text = pattern.sub("", text)

    text = _STAGE_DIRECTION.sub("", text)

    # Collapse multiple spaces left by removals
    text = re.sub(r"  +", " ", text).strip()

    # After removals, so a line that was *action* "speech" still unwraps.
    wrapped = _WRAPPING_QUOTES.match(text)
    if wrapped:
        inner = wrapped.group(1)
        # Only unwrap a quote enclosing the whole line. '"Yo," he said, "go"' has a
        # leading and trailing quote too, and stripping those would corrupt it.
        if '"' not in inner:
            text = inner.strip()

    # Last line of defence: drop anything the fold missed rather than let the mod's
    # ASCII encoder turn it into '?' inside the speech bubble.
    text = text.encode("ascii", "ignore").decode("ascii")

    text = re.sub(r"  +", " ", text).strip()

    return _truncate_at_sentence(text, max_chars)