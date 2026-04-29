"""LLM and embedding provider tests with mocks."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.providers.embeddings import (
    EmbeddingProvider,
    LocalEmbeddings,
    OpenAIEmbeddings,
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
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    with patch("openai.OpenAI"):
        provider = get_embedding_provider()
    assert isinstance(provider, EmbeddingProvider)
    assert isinstance(provider, OpenRouterEmbeddings)


# --------------------- Multi-provider dispatch ---------------------

def test_get_llm_provider_dispatches_openai(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with patch("openai.OpenAI"):
        provider = get_llm_provider()
    from src.providers.openai_llm import OpenAILLM

    assert isinstance(provider, OpenAILLM)
    assert isinstance(provider, LLMProvider)


def test_get_llm_provider_dispatches_anthropic(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("anthropic.Anthropic"):
        provider = get_llm_provider()
    from src.providers.anthropic_llm import AnthropicLLM

    assert isinstance(provider, AnthropicLLM)


def test_get_llm_provider_dispatches_ollama(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    provider = get_llm_provider()
    from src.providers.ollama_llm import OllamaLLM

    assert isinstance(provider, OllamaLLM)


def test_get_llm_provider_rejects_unknown(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LLM_PROVIDER", "bogus")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        get_llm_provider()


def test_openai_llm_complete_records_usage(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock(message=MagicMock(content='{"ok": true}'))]
    fake_resp.usage = MagicMock(prompt_tokens=11, completion_tokens=22)
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_resp
    with patch("openai.OpenAI", return_value=fake_client):
        from src.providers.openai_llm import OpenAILLM

        llm = OpenAILLM(model="gpt-4o-mini")
    out = llm.complete([{"role": "user", "content": "hi"}])
    assert out == '{"ok": true}'
    assert llm.last_usage.prompt_tokens == 11
    assert llm.last_usage.completion_tokens == 22
    assert llm.cumulative_prompt_tokens == 11
    assert llm.cumulative_cost_usd > 0


def test_anthropic_llm_complete_records_usage(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    block = MagicMock()
    block.text = "Hello there."
    fake_resp = MagicMock()
    fake_resp.content = [block]
    fake_resp.usage = MagicMock(input_tokens=7, output_tokens=4)
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_resp
    with patch("anthropic.Anthropic", return_value=fake_client):
        from src.providers.anthropic_llm import AnthropicLLM

        llm = AnthropicLLM(model="claude-sonnet-4-6")
    out = llm.complete(
        [
            {"role": "system", "content": "you are X"},
            {"role": "user", "content": "hi"},
        ],
        response_format={"type": "json_object"},
    )
    assert out == "Hello there."
    assert llm.last_usage.prompt_tokens == 7
    assert llm.last_usage.completion_tokens == 4
    # System messages must be split out and passed as a top-level kwarg.
    create_kwargs = fake_client.messages.create.call_args.kwargs
    assert "system" in create_kwargs
    assert "you are X" in create_kwargs["system"]
    # JSON response format triggers a system-level nudge.
    assert "JSON" in create_kwargs["system"] or "json" in create_kwargs["system"]
    # The system message itself should NOT appear in the message list.
    assert all(m["role"] != "system" for m in create_kwargs["messages"])


def test_ollama_llm_complete_no_cost(monkeypatch: pytest.MonkeyPatch):
    fake_payload = {
        "message": {"role": "assistant", "content": "Local response."},
        "prompt_eval_count": 5,
        "eval_count": 3,
    }
    import io
    import json as _json

    class _FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return _json.dumps(self._payload).encode("utf-8")

    captured: dict = {}

    def _fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = _json.loads(req.data.decode("utf-8"))
        return _FakeResponse(fake_payload)

    with patch("urllib.request.urlopen", _fake_urlopen):
        from src.providers.ollama_llm import OllamaLLM

        llm = OllamaLLM(base_url="http://localhost:11434", model="llama3.1:8b")
        out = llm.complete([{"role": "user", "content": "hi"}], response_format={"type": "json_object"})
    assert out == "Local response."
    assert llm.last_usage.cost_usd == 0.0
    assert llm.cumulative_prompt_tokens == 5
    assert llm.cumulative_completion_tokens == 3
    assert captured["body"]["format"] == "json"
    assert captured["body"]["options"]["temperature"] == 0.2


def test_openai_embeddings_dispatch(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    fake_client = MagicMock()
    fake_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.0] * 4)]
    )
    with patch("openai.OpenAI", return_value=fake_client):
        provider = get_embedding_provider()
        vecs = provider.embed(["hi"])
    assert isinstance(provider, OpenAIEmbeddings)
    assert provider.dimension == 1536
    assert len(vecs) == 1


def test_get_embedding_provider_rejects_unknown(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "nope")
    with pytest.raises(ValueError, match="Unknown EMBEDDING_PROVIDER"):
        get_embedding_provider()
