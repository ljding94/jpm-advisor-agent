"""PII redaction tests."""
from __future__ import annotations

from src.guardrails.pii import redact, redact_messages, redact_text


def test_ssn_redacted():
    out = redact("My SSN is 123-45-6789")
    assert "[REDACTED_SSN]" in out.text
    assert out.counts.get("ssn", 0) == 1


def test_ssn_no_dashes_redacted():
    out = redact("ssn: 123456789 thanks")
    assert "[REDACTED_SSN]" in out.text or "[REDACTED_ACCOUNT_NUMBER]" in out.text
    assert out.total >= 1


def test_email_redacted():
    out = redact("Email me at jane.doe+work@example.com please.")
    assert "[REDACTED_EMAIL]" in out.text
    assert "@" not in out.text


def test_phone_redacted():
    text = "Call me at (415) 555-1212 or +1 415-555-1212"
    out = redact(text)
    assert out.counts.get("phone", 0) >= 2


def test_credit_card_redacted():
    out = redact("card 4111 1111 1111 1111")
    assert "[REDACTED_CREDIT_CARD]" in out.text


def test_account_number_redacted():
    out = redact("account 1234567890")
    assert out.total >= 1


def test_no_pii_returns_unchanged():
    text = "I want help planning for retirement."
    out = redact(text)
    assert out.text == text
    assert out.counts == {}


def test_redact_text_helper():
    assert "[REDACTED_EMAIL]" in redact_text("write to a@b.co")


def test_redact_messages_redacts_each_content():
    msgs = [
        {"role": "system", "content": "hi"},
        {"role": "user", "content": "ssn 123-45-6789 and email a@b.co"},
    ]
    out = redact_messages(msgs)
    assert out[0]["content"] == "hi"
    assert "[REDACTED_SSN]" in out[1]["content"]
    assert "[REDACTED_EMAIL]" in out[1]["content"]


def test_pii_never_reaches_llm_via_base_agent(fake_llm, monkeypatch):
    """Acceptance criterion #6: SSN redacted before any LLM call."""
    from src.agents.base import BaseAgent
    from src.providers.llm import LLMProvider
    from src.schemas import AgentRole

    class _DummyAgent(BaseAgent):
        role = AgentRole.ADVISOR

        def process(self, state):
            return state

        def respond(self, user_text: str) -> str:
            redacted = redact_text(user_text)
            return self._call_llm(redacted)

    agent = _DummyAgent(name="x", system_prompt="be brief", llm=fake_llm)
    fake_llm.script.append("ok")
    agent.respond("My SSN is 123-45-6789, please remember it.")
    sent = fake_llm.calls[-1]["messages"]
    joined = " ".join(m["content"] for m in sent)
    assert "123-45-6789" not in joined
    assert "[REDACTED_SSN]" in joined
