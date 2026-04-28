"""DuckDuckGo web search behind a swappable WebSearchProvider interface."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


@dataclass
class WebResult:
    title: str
    url: str
    snippet: str

    def as_dict(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


class WebSearchProvider(ABC):
    """Abstract web search provider."""

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[WebResult]:
        ...


class DDGSearchProvider(WebSearchProvider):
    """DuckDuckGo via the `ddgs` package. No API key required."""

    def __init__(self, sleep_seconds: float = 1.0) -> None:
        self.sleep_seconds = sleep_seconds
        self._last_call_at: float = 0.0

    def search(self, query: str, max_results: int = 5) -> list[WebResult]:
        if not query.strip():
            return []
        self._rate_limit()
        return self._do_search(query=query, max_results=max_results)

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_call_at
        if elapsed < self.sleep_seconds:
            time.sleep(self.sleep_seconds - elapsed)
        self._last_call_at = time.monotonic()

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
        retry=retry_if_exception_type(Exception),
    )
    def _do_search(self, query: str, max_results: int) -> list[WebResult]:
        from ddgs import DDGS

        out: list[WebResult] = []
        with DDGS() as client:
            for r in client.text(query, max_results=max_results):
                out.append(
                    WebResult(
                        title=str(r.get("title", "")).strip() or "(untitled)",
                        url=str(r.get("href") or r.get("url") or "").strip(),
                        snippet=str(r.get("body") or r.get("snippet") or "").strip(),
                    )
                )
                if len(out) >= max_results:
                    break
        return out


@dataclass
class FakeWebSearchProvider(WebSearchProvider):
    """In-memory provider for tests."""

    canned: list[WebResult] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)

    def search(self, query: str, max_results: int = 5) -> list[WebResult]:
        self.calls.append(query)
        return list(self.canned)[:max_results]


def get_web_search_provider(**kwargs: Any) -> WebSearchProvider:
    return DDGSearchProvider(**kwargs)
