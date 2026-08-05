"""
Single-item request queue: ensures only one inference is in-flight at a time.
P5R conversations move fast; if a second hook fires while the first LLM call
is still running we drop the new request rather than queueing a backlog that
would cause stale dialogue to appear seconds after the conversation moved on.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Awaitable

log = logging.getLogger(__name__)


class InferenceQueue:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
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
        """Run task only if no inference is currently in-flight; else return None."""
        self.total_requests += 1
        if self._lock.locked():
            self.total_drops += 1
            log.debug("Inference already in-flight; dropping request.")
            return None
        async with self._lock:
            t0 = time.perf_counter()
            result = await task()
            elapsed = time.perf_counter() - t0
            self._latencies.append(elapsed)
            if len(self._latencies) > 100:
                self._latencies.pop(0)
            self.total_completions += 1
            return result