"""LLM and embedding provider tests with mocks."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.providers.embeddings import (
    LocalEmbeddings,
    OpenRouterEmbeddings,
    get_embedding_provider,
)
from src.providers.llm import LLMProvider, OpenRouterLLM, get_llm_provider


# --------------------- LLM provider ---------------------

def _fake_openai_completion(content: str = '{"hello": "world"}'):
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content=content))]
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response
    return fake_client


def test_openrouter_llm_requires_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        OpenRouterLLM()


def test_openrouter_llm_complete_passes_args(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    fake_client = _fake_openai_completion('{"ok": true}')
    with patch("openai.OpenAI", return_value=fake_client):
        llm = OpenRouterLLM(model="x/y")
    out = llm.complete(
        [{"role": "user", "content": "hello"}],
        max_tokens=42,
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    assert out == '{"ok": true}'
    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "x/y"
    assert call_kwargs["max_tokens"] == 42
    assert call_kwargs["response_format"] == {"type": "json_object"}


def test_openrouter_llm_returns_empty_string_when_content_none(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    fake_client = _fake_openai_completion(content=None)  # type: ignore[arg-type]
    with patch("openai.OpenAI", return_value=fake_client):
        llm = OpenRouterLLM()
    assert llm.complete([{"role": "user", "content": "hi"}]) == ""


def test_get_llm_provider_constructs_openrouter(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    with patch("openai.OpenAI"):
        provider = get_llm_provider()
    assert isinstance(provider, LLMProvider)


# --------------------- Embedding provider ---------------------

def test_openrouter_embeddings_requires_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        OpenRouterEmbeddings()


def test_openrouter_embeddings_calls_api(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    fake_client = MagicMock()
    fake_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1, 0.2, 0.3]), MagicMock(embedding=[0.4, 0.5, 0.6])]
    )
    with patch("openai.OpenAI", return_value=fake_client):
        emb = OpenRouterEmbeddings()
    vectors = emb.embed(["hello", "world"])
    assert len(vectors) == 2
    assert vectors[0] == [0.1, 0.2, 0.3]
    fake_client.embeddings.create.assert_called_once()


def test_openrouter_embeddings_empty_input(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    fake_client = MagicMock()
    with patch("openai.OpenAI", return_value=fake_client):
        emb = OpenRouterEmbeddings()
    assert emb.embed([]) == []
    fake_client.embeddings.create.assert_not_called()


def test_get_embedding_provider_returns_local_without_key(
    monkeypatch: pytest.MonkeyPatch,
):
    """Without OPENROUTER_API_KEY, get_embedding_provider() falls back to local."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    fake_local = MagicMock(spec=LocalEmbeddings)
    with patch("src.providers.embeddings.LocalEmbeddings", return_value=fake_local):
        result = get_embedding_provider()
    assert result is fake_local


def test_get_embedding_provider_returns_openrouter_with_key(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    with patch("openai.OpenAI"):
        provider = get_embedding_provider()
    assert isinstance(provider, OpenRouterEmbeddings)
