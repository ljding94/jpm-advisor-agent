"""Web search tool tests with mocked DDGS client."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.tools.web_search import (
    DDGSearchProvider,
    FakeWebSearchProvider,
    WebResult,
)


def _ddgs_context_manager(rows: list[dict]):
    """Build a fake DDGS class whose context manager yields a client with .text()."""
    fake_client = MagicMock()
    fake_client.text.return_value = iter(rows)
    fake_ddgs_cls = MagicMock()
    fake_ddgs_cls.return_value.__enter__.return_value = fake_client
    fake_ddgs_cls.return_value.__exit__.return_value = False
    return fake_ddgs_cls, fake_client


def test_ddg_search_returns_top_results():
    rows = [
        {"title": "Asset allocation 101", "href": "https://example.com/a", "body": "intro"},
        {"title": "Risk and return", "href": "https://example.com/b", "body": "tradeoff"},
        {"title": "Diversification", "href": "https://example.com/c", "body": "free lunch"},
    ]
    fake_cls, fake_client = _ddgs_context_manager(rows)
    with patch("ddgs.DDGS", fake_cls):
        provider = DDGSearchProvider(sleep_seconds=0.0)
        results = provider.search("asset allocation", max_results=3)
    assert len(results) == 3
    assert results[0].title == "Asset allocation 101"
    assert results[0].url == "https://example.com/a"
    fake_client.text.assert_called_once()


def test_ddg_search_handles_missing_fields():
    rows = [{"title": "", "href": None, "body": None}]
    fake_cls, _ = _ddgs_context_manager(rows)
    with patch("ddgs.DDGS", fake_cls):
        provider = DDGSearchProvider(sleep_seconds=0.0)
        results = provider.search("anything")
    assert results == [WebResult(title="(untitled)", url="", snippet="")]


def test_ddg_search_blank_query_returns_empty():
    provider = DDGSearchProvider(sleep_seconds=0.0)
    assert provider.search("   ") == []


def test_ddg_search_retries_then_succeeds():
    """First call raises, second succeeds."""
    call_count = {"n": 0}

    def fake_text(*_a, **_kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("transient")
        return iter([{"title": "ok", "href": "https://x", "body": "y"}])

    fake_client = MagicMock()
    fake_client.text.side_effect = fake_text
    fake_ddgs_cls = MagicMock()
    fake_ddgs_cls.return_value.__enter__.return_value = fake_client

    with patch("ddgs.DDGS", fake_ddgs_cls):
        provider = DDGSearchProvider(sleep_seconds=0.0)
        results = provider.search("retry me", max_results=1)
    assert call_count["n"] == 2
    assert results[0].title == "ok"


def test_fake_web_search_records_calls():
    fake = FakeWebSearchProvider(
        canned=[WebResult(title="t", url="u", snippet="s")]
    )
    assert fake.search("q1", max_results=1) == [WebResult(title="t", url="u", snippet="s")]
    assert fake.calls == ["q1"]


def test_web_result_as_dict():
    r = WebResult(title="t", url="u", snippet="s")
    assert r.as_dict() == {"title": "t", "url": "u", "snippet": "s"}
