"""Async surface on WebSearchProvider."""
from __future__ import annotations

import asyncio

import pytest

from src.tools.web_search import (
    DDGSearchProvider,
    FakeWebSearchProvider,
    WebResult,
)


@pytest.mark.asyncio
async def test_async_search_default_dispatches_to_sync():
    fake = FakeWebSearchProvider(canned=[
        WebResult(title="t", url="u", snippet="s"),
    ])
    results = await fake.search_async("anything", max_results=3)
    assert results == [WebResult(title="t", url="u", snippet="s")]
    assert fake.calls == ["anything"]


@pytest.mark.asyncio
async def test_async_search_does_not_block_event_loop():
    """search_async should run on a worker thread, leaving the loop free."""

    class SlowFake(FakeWebSearchProvider):
        def search(self, query, max_results=5):  # type: ignore[override]
            import time
            time.sleep(0.05)
            return list(self.canned)

    fake = SlowFake(canned=[WebResult(title="t", url="u", snippet="s")])

    async def tick():
        await asyncio.sleep(0)
        return "tick"

    # Run the slow search and a tick concurrently — tick should complete first.
    search_task = asyncio.create_task(fake.search_async("q"))
    tick_result = await tick()
    assert tick_result == "tick"
    results = await search_task
    assert len(results) == 1


@pytest.mark.asyncio
async def test_async_search_blank_query():
    provider = DDGSearchProvider(sleep_seconds=0.0)
    assert await provider.search_async("   ") == []
