"""
Single-item request queue: ensures only one inference is in-flight at a time.
P5R conversations move fast; if a second hook fires while the first LLM call
is still running we drop the new request rather than queueing a backlog that
would cause stale dialogue to appear seconds after the conversation moved on.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Awaitable

log = logging.getLogger(__name__)


class InferenceQueue:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    async def run_if_idle(
        self,
        task: Callable[[], Awaitable[str]],
    ) -> str | None:
        """Run task only if no inference is currently in-flight; else return None."""
        if self._lock.locked():
            log.debug("Inference already in-flight; dropping request.")
            return None
        async with self._lock:
            return await task()