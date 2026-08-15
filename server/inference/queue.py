"""
Bounded request queue: one inference in flight, a short line behind it.

The original policy dropped anything arriving while a call was running, because the mod
generated reactively and a late answer was a wrong answer -- it would have overwritten
the line after the one the player was reading.

Pre-generation inverts that. A request is for a record the player has not reached, the
mod refuses to write behind its own cursor, and a record freezes once the interpreter
reads it. Lateness is handled on the client, so dropping is pure waste: one scene
recorded 78 requests, 57 drops and 14 completions.

The queue stays shallow anyway. Depth is a promise that work will be done, and a deep
queue promises lines for records a branching scene may never reach.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Awaitable

log = logging.getLogger(__name__)


class InferenceQueue:
    #: Requests allowed to wait behind the running one. Two covers the gap between a
    #: 500ms poll tick and a 2-3s generation without promising work for records the
    #: player may never see.
    MAX_WAITING = 2

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._waiting = 0
        self.total_requests: int = 0
        self.total_drops: int = 0
        self.total_completions: int = 0
        self._latencies: list[float] = []

    @property
    def avg_latency_ms(self) -> float | None:
        if not self._latencies:
            return None
        return sum(self._latencies) / len(self._latencies) * 1000

    def clear_stats(self) -> None:
        """Reset all counters and the latency buffer. Thread-safe for async use."""
        self.total_requests = 0
        self.total_drops = 0
        self.total_completions = 0
        self._latencies.clear()

    async def run_if_idle(
        self,
        task: Callable[[], Awaitable[str]],
    ) -> str | None:
        """Run task, waiting briefly if one is already in flight; None if the line is full."""
        self.total_requests += 1
        if self._lock.locked() and self._waiting >= self.MAX_WAITING:
            self.total_drops += 1
            log.debug("Inference queue full (%d waiting); dropping request.", self._waiting)
            return None

        self._waiting += 1
        try:
            async with self._lock:
                return await self._run_locked(task)
        finally:
            self._waiting -= 1

    async def _run_locked(self, task: Callable[[], Awaitable[str]]) -> str:
        """Run the task with the lock held, recording how long it took."""
        t0 = time.perf_counter()
        result = await task()
        elapsed = time.perf_counter() - t0

        self._latencies.append(elapsed)
        if len(self._latencies) > 100:
            self._latencies.pop(0)
        self.total_completions += 1
        return result