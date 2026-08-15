"""Test InferenceQueue drop-on-busy policy and stats tracking."""

import asyncio
import pytest
from inference.queue import InferenceQueue


@pytest.mark.asyncio
async def test_second_request_waits_rather_than_dropping() -> None:
    """
    Reverses the original policy.

    Dropping was right while generation was reactive: a late answer would have
    overwritten the line after the one being read. Pre-generation asks for records the
    player has not reached, refuses to write behind its own cursor, and freezes a record
    once the interpreter reads it -- so lateness is handled on the client and dropping is
    pure waste. One scene recorded 78 requests, 57 drops and 14 completions.
    """
    queue = InferenceQueue()

    async def slow_task() -> str:
        await asyncio.sleep(0.05)
        return "done"

    t1 = asyncio.create_task(queue.run_if_idle(slow_task))
    await asyncio.sleep(0.01)  # let t1 acquire the lock
    t2 = asyncio.create_task(queue.run_if_idle(slow_task))

    r1, r2 = await asyncio.gather(t1, t2)
    assert r1 == "done"
    assert r2 == "done"
    assert queue.total_drops == 0


@pytest.mark.asyncio
async def test_requests_beyond_the_line_are_still_dropped() -> None:
    """Depth is a promise of work; a deep queue promises lines nobody will reach."""
    queue = InferenceQueue()

    async def slow_task() -> str:
        await asyncio.sleep(0.05)
        return "done"

    tasks = [asyncio.create_task(queue.run_if_idle(slow_task))]
    await asyncio.sleep(0.01)
    tasks += [
        asyncio.create_task(queue.run_if_idle(slow_task))
        for _ in range(InferenceQueue.MAX_WAITING + 2)
    ]

    results = await asyncio.gather(*tasks)
    assert results.count(None) >= 1
    assert queue.total_drops >= 1


@pytest.mark.asyncio
async def test_queued_requests_run_one_at_a_time() -> None:
    """The whole point of the lock: llama-server handles a single slot."""
    queue = InferenceQueue()
    concurrent = 0
    peak = 0

    async def tracked() -> str:
        nonlocal concurrent, peak
        concurrent += 1
        peak = max(peak, concurrent)
        await asyncio.sleep(0.02)
        concurrent -= 1
        return "done"

    await asyncio.gather(*[queue.run_if_idle(tracked) for _ in range(3)])
    assert peak == 1


@pytest.mark.asyncio
async def test_stats_track_requests_and_drops() -> None:
    queue = InferenceQueue()

    async def instant_task() -> str:
        return "ok"

    async def slow_task() -> str:
        await asyncio.sleep(0.05)
        return "done"

    # One successful completion
    result = await queue.run_if_idle(instant_task)
    assert result == "ok"
    assert queue.total_requests == 1
    assert queue.total_completions == 1
    assert queue.total_drops == 0
    assert queue.avg_latency_ms is not None and queue.avg_latency_ms >= 0

    # A concurrent call now waits its turn instead of being dropped, so the drop
    # counter only moves once the short line behind the running request is full.
    t1 = asyncio.create_task(queue.run_if_idle(slow_task))
    await asyncio.sleep(0.01)
    t2 = asyncio.create_task(queue.run_if_idle(slow_task))
    await asyncio.gather(t1, t2)

    assert queue.total_requests == 3
    assert queue.total_drops == 0        # it waited rather than being dropped
    assert queue.total_completions == 3


@pytest.mark.asyncio
async def test_avg_latency_none_before_first_completion() -> None:
    queue = InferenceQueue()
    assert queue.avg_latency_ms is None


@pytest.mark.asyncio
async def test_clear_stats_resets_all_counters() -> None:
    queue = InferenceQueue()

    async def instant_task() -> str:
        return "ok"

    await queue.run_if_idle(instant_task)
    assert queue.total_completions == 1

    queue.clear_stats()
    assert queue.total_requests == 0
    assert queue.total_drops == 0
    assert queue.total_completions == 0
    assert queue.avg_latency_ms is None


@pytest.mark.asyncio
async def test_clear_stats_then_normal_tracking_resumes() -> None:
    """Counters should increment normally after a clear."""
    queue = InferenceQueue()

    async def instant_task() -> str:
        return "ok"

    await queue.run_if_idle(instant_task)
    queue.clear_stats()
    await queue.run_if_idle(instant_task)
    assert queue.total_requests == 1
    assert queue.total_completions == 1