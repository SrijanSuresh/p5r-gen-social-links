"""
The line the player typed, held until the mod comes to collect it.

The mod polls; it does not listen. There is no socket into P5R's process and adding one
means another thread inside somebody else's address space for no gain. So the browser page
posts a line here, this holds it, and the mod picks it up on a poll it already makes.

Two decisions worth stating, both in :class:`PlayerInputStore`:

* **One line, not a queue.** If the player types twice before anything collects it, the
  second line is the one they meant. A queue would make the confidant answer a line the
  player had already replaced.
* **It expires.** A line typed and then abandoned — the tab closed, the player walked away
  — must not surface in a scene half an hour later.

See learning.md Ch. 87.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

MAX_LINE_CHARS = 280
"""
Longest line accepted.

Generous next to the ~56 characters a bubble holds, because this is the player's input
rather than the model's output: it is context for a reply, not text to be written into
game memory, so it is bounded to keep the prompt inside Pydantic's 1024-character context
limit rather than to fit a slot.
"""

DEFAULT_TTL_SECONDS = 120.0
"""
How long a typed line stays collectable.

Two minutes is long enough to cover typing, a mistake, and a retype, and short enough that
a line abandoned at the end of one session is gone before the next one starts.
"""


class PlayerInputError(ValueError):
    """The submitted line is empty, too long, or otherwise not usable."""


@dataclass(frozen=True)
class PlayerLine:
    """A line the player typed, and when."""

    text: str
    submitted_at: float

    def age_seconds(self, now: float) -> float:
        return now - self.submitted_at


def normalise(text: str) -> str:
    """
    Clean a submitted line into something that can go in a prompt.

    Newlines collapse to spaces: the prompt is line-oriented and a submitted paragraph
    would otherwise break the `[...]` blocks the user half is built from. Control
    characters go entirely — they cannot be typed on purpose and they survive into the
    prompt as mojibake.

    Raises:
        PlayerInputError: nothing usable survived, or the line is too long.
    """
    if not isinstance(text, str):
        raise PlayerInputError("a line must be text")

    cleaned = "".join(" " if ch in "\r\n\t" else ch for ch in text)
    cleaned = "".join(ch for ch in cleaned if ch.isprintable() or ch == " ")
    cleaned = " ".join(cleaned.split())

    if not cleaned:
        raise PlayerInputError("nothing to say")
    if len(cleaned) > MAX_LINE_CHARS:
        raise PlayerInputError(
            f"line is {len(cleaned)} characters; the limit is {MAX_LINE_CHARS}"
        )
    return cleaned


class PlayerInputStore:
    """
    Holds at most one pending player line.

    Locked rather than lock-free: the web route writes it from the event loop and the
    mod's poll reads it from a worker thread, and `take()` has to be atomic against
    `submit()` or two collectors could both see the same line.
    """

    def __init__(self, ttl_seconds: float = DEFAULT_TTL_SECONDS) -> None:
        if ttl_seconds <= 0:
            raise PlayerInputError(f"ttl_seconds must be positive, got {ttl_seconds}")
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._pending: PlayerLine | None = None
        self._submitted = 0
        self._collected = 0
        self._expired = 0

    @property
    def ttl_seconds(self) -> float:
        return self._ttl

    @property
    def submitted_count(self) -> int:
        return self._submitted

    @property
    def collected_count(self) -> int:
        return self._collected

    @property
    def expired_count(self) -> int:
        return self._expired

    def submit(self, text: str, now: float | None = None) -> PlayerLine:
        """
        Accept a line, replacing any line not yet collected.

        Raises:
            PlayerInputError: the line is empty or too long.
        """
        cleaned = normalise(text)
        stamp = time.monotonic() if now is None else now

        line = PlayerLine(text=cleaned, submitted_at=stamp)
        with self._lock:
            self._pending = line
            self._submitted += 1
        return line

    def peek(self, now: float | None = None) -> PlayerLine | None:
        """The pending line without consuming it, or None. Expired lines are dropped."""
        stamp = time.monotonic() if now is None else now
        with self._lock:
            return self._drop_if_stale(stamp)

    def take(self, now: float | None = None) -> PlayerLine | None:
        """
        Consume and return the pending line, or None if there is nothing fresh.

        Consuming rather than peeking is what stops one typed line being answered by
        several records: the first generation that asks for it gets it, and the rest see
        an empty store.
        """
        stamp = time.monotonic() if now is None else now
        with self._lock:
            line = self._drop_if_stale(stamp)
            if line is None:
                return None
            self._pending = None
            self._collected += 1
            return line

    def clear(self) -> None:
        """Forget any pending line. Called when a hang-out ends."""
        with self._lock:
            self._pending = None

    def _drop_if_stale(self, now: float) -> PlayerLine | None:
        """Caller holds the lock."""
        if self._pending is None:
            return None
        if self._pending.age_seconds(now) > self._ttl:
            self._pending = None
            self._expired += 1
            return None
        return self._pending
