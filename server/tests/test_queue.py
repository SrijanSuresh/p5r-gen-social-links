"""Test InferenceQueue drop-on-busy policy and stats tracking."""

import asyncio
import pytest
from inference.queue import InferenceQueue


@pytest.mark.asyncio
async def test_second_request_is_dropped() -> None:
    queue = InferenceQueue()

    async def slow_task() -> str:
        await asyncio.sleep(0.05)
        return "done"

    t1 = asyncio.create_task(queue.run_if_idle(slow_task))
    await asyncio.sleep(0.01)  # let t1 acquire the lock
    t2 = asyncio.create_task(queue.run_if_idle(slow_task))

    r1, r2 = await asyncio.gather(t1, t2)
    assert r1 == "done"
    assert r2 is None  # second request was dropped


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

    # Drop on concurrent call
    t1 = asyncio.create_task(queue.run_if_idle(slow_task))
    await asyncio.sleep(0.01)
    t2 = asyncio.create_task(queue.run_if_idle(slow_task))
    await asyncio.gather(t1, t2)

    assert queue.total_requests == 3
    assert queue.total_drops == 1
    assert queue.total_completions == 2


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