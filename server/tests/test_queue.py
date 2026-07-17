"""Test InferenceQueue drop-on-busy policy."""

import asyncio
import pytest
from inference.queue import InferenceQueue


@pytest.mark.asyncio
async def test_second_request_is_dropped() -> None:
    queue = InferenceQueue()
    results: list[str | None] = []

    async def slow_task() -> str:
        await asyncio.sleep(0.05)
        return "done"

    t1 = asyncio.create_task(queue.run_if_idle(slow_task))
    await asyncio.sleep(0.01)  # let t1 acquire the lock
    t2 = asyncio.create_task(queue.run_if_idle(slow_task))

    r1, r2 = await asyncio.gather(t1, t2)
    assert r1 == "done"
    assert r2 is None  # second request was dropped