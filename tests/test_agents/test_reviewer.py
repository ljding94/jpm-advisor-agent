"""ReviewerAgent tests with FakeLLM."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agents.reviewer import (
    MAX_REVIEWER_RETRIES,
    ReviewerAgent,
    ReviewVerdict,
)
from src.graph.state import ConversationStatus, initial_state
from src.schemas import (
    STANDARD_DISCLAIMER,
    AgentMessage,
    AgentRole,
    MessageType,
)


# ---------- helpers ----------

def _advice_msg(content: str) -> AgentMessage:
    return AgentMessage(
        sender=AgentRole.ADVISOR,
        recipient=AgentRole.REVIEWER,
        content=content,
        message_type=MessageType.ADVICE,
    )


def _question_msg(content: str) -> AgentMessage:
    return AgentMessage(
        sender=AgentRole.ADVISOR,
        recipient=AgentRole.REVIEWER,
        content=content,
        message_type=MessageType.QUESTION,
    )


def _verdict_response(
    verdict: str = "pass",
    reasons: list[str] | None = None,
    matched_rule_ids: list[str] | None = None,
    revised: str = "",
) -> str:
    return json.dumps(
        {
            "verdict": verdict,
            "reasons": reasons or [],
            "matched_rule_ids": matched_rule_ids or [],
            "revised_content": revised,
        }
    )


# ---------- policy loading ----------

def test_policy_loads_from_default_file(fake_llm):
    reviewer = ReviewerAgent(llm=fake_llm)
    assert reviewer.rules, "expected at least one rule loaded from data/reviewer/policy.yaml"
    ids = {r.id for r in reviewer.rules}
    # Some critical rules must exist by id.
    assert "disclosure.standard_disclaimer" in ids
    assert "claims.no_guarantees" in ids


def test_policy_loads_from_custom_path(fake_llm, tmp_path: Path):
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        "rules:\n"
        "  - id: only_rule\n"
        "    severity: revise\n"
        "    applies_to: [advice]\n"
        "    description: only rule\n"
    )
    reviewer = ReviewerAgent(llm=fake_llm, policy_path=policy)
    assert [r.id for r in reviewer.rules] == ["only_rule"]


def test_missing_policy_file_yields_empty_rules(fake_llm, tmp_path: Path):
    reviewer = ReviewerAgent(llm=fake_llm, policy_path=tmp_path / "missing.yaml")
    assert reviewer.rules == ()


# ---------- pass path ----------

def test_pass_forwards_message_to_client_unchanged(david_profile, fake_llm):
    fake_llm.script.append(_verdict_response("pass"))
    reviewer = ReviewerAgent(llm=fake_llm)
    state = initial_state(david_profile)
    original = _advice_msg(
        "Maintain a diversified portfolio across stocks, bonds, and cash. "
        f"Disclaimer: {STANDARD_DISCLAIMER}"
    )
    state["conversation_history"].append(original)

    new_state = reviewer.process(state)
    last = new_state["conversation_history"][-1]
    assert last.sender is AgentRole.REVIEWER
    assert last.recipient is AgentRole.CLIENT
    assert last.message_type is MessageType.ADVICE
    assert last.content == original.content
    assert last.metadata["review"]["verdict"] == "pass"
    assert new_state["reviewer_retries"] == 0
    assert new_state["status"] is ConversationStatus.CONFIRM


# ---------- revise path ----------

def test_revise_uses_llm_revised_content(david_profile, fake_llm):
    revised = (
        "Maintain a diversified portfolio. "
        f"Disclaimer: {STANDARD_DISCLAIMER}"
    )
    fake_llm.script.append(
        _verdict_response("revise", reasons=["missing disclaimer"], revised=revised)
    )
    reviewer = ReviewerAgent(llm=fake_llm)
    state = initial_state(david_profile)
    state["conversation_history"].append(_advice_msg("Maintain diversification."))

    new_state = reviewer.process(state)
    last = new_state["conversation_history"][-1]
    assert last.recipient is AgentRole.CLIENT
    assert last.content == revised
    assert last.metadata["review"]["verdict"] == "revise"


def test_deterministic_revise_strips_ticker_and_adds_disclaimer(david_profile, fake_llm):
    """Deterministic guardrail trips on ticker + missing disclaimer; reviewer rewrites."""
    # LLM cannot help — return pass; deterministic check should still force a revise.
    fake_llm.script.append(_verdict_response("pass"))
    reviewer = ReviewerAgent(llm=fake_llm)
    state = initial_state(david_profile)
    state["conversation_history"].append(_advice_msg("Buy MSFT for tech exposure."))

    new_state = reviewer.process(state)
    last = new_state["conversation_history"][-1]
    assert last.recipient is AgentRole.CLIENT
    assert "MSFT" not in last.content
    assert "[asset class]" in last.content
    assert "not financial advice" in last.content.lower()
    assert last.metadata["review"]["verdict"] == "revise"


# ---------- block path ----------

def test_block_bounces_to_advisor_below_retry_limit(david_profile, fake_llm):
    fake_llm.script.append(
        _verdict_response(
            "block",
            reasons=["promises guaranteed returns"],
            matched_rule_ids=["claims.no_guarantees"],
        )
    )
    reviewer = ReviewerAgent(llm=fake_llm)
    state = initial_state(david_profile)
    state["conversation_history"].append(_advice_msg("This portfolio is risk-free."))

    new_state = reviewer.process(state)
    last = new_state["conversation_history"][-1]
    assert last.sender is AgentRole.REVIEWER
    assert last.recipient is AgentRole.ADVISOR
    assert last.message_type is MessageType.REVIEW
    assert "BLOCKED" in last.content
    assert "claims.no_guarantees" in last.content
    assert new_state["reviewer_retries"] == 1
    # Conversation should NOT move to CONFIRM — there is no client-visible advice yet.
    assert new_state["status"] is not ConversationStatus.CONFIRM


def test_block_falls_back_to_revise_after_retries_exhausted(david_profile, fake_llm):
    fake_llm.script.append(
        _verdict_response(
            "block",
            reasons=["promises guaranteed returns"],
            matched_rule_ids=["claims.no_guarantees"],
        )
    )
    reviewer = ReviewerAgent(llm=fake_llm)
    state = initial_state(david_profile)
    state["reviewer_retries"] = MAX_REVIEWER_RETRIES  # already at cap
    original = _advice_msg(
        "This portfolio offers guaranteed returns. "
        f"Disclaimer: {STANDARD_DISCLAIMER}"
    )
    state["conversation_history"].append(original)

    new_state = reviewer.process(state)
    last = new_state["conversation_history"][-1]
    # Fallback path emits a sanitized message to the client so the loop can complete.
    assert last.recipient is AgentRole.CLIENT
    assert last.message_type is MessageType.ADVICE
    assert "guaranteed returns" not in last.content.lower()
    assert "historical tendency" in last.content.lower()
    assert new_state["reviewer_retries"] == 0
    # Sanitized fallback annotation present.
    assert "fallback_after_retries" in last.metadata["review"]
    # An error was appended noting the exhaustion.
    assert any("retries exhausted" in e for e in new_state.get("errors", []))


# ---------- routing-tier asserts ----------

def test_reviewer_no_op_when_last_message_not_for_reviewer(david_profile, fake_llm):
    reviewer = ReviewerAgent(llm=fake_llm)
    state = initial_state(david_profile)
    state["conversation_history"].append(
        AgentMessage(
            sender=AgentRole.ADVISOR,
            recipient=AgentRole.ANALYST,
            content="research please",
            message_type=MessageType.TASK,
        )
    )
    new_state = reviewer.process(state)
    assert new_state == state  # unchanged


def test_reviewer_no_op_when_sender_not_advisor(david_profile, fake_llm):
    """Reviewer only reviews advisor-authored content; ignores anything else."""
    reviewer = ReviewerAgent(llm=fake_llm)
    state = initial_state(david_profile)
    state["conversation_history"].append(
        AgentMessage(
            sender=AgentRole.SYSTEM,
            recipient=AgentRole.REVIEWER,
            content="system note",
            message_type=MessageType.SYSTEM,
        )
    )
    new_state = reviewer.process(state)
    assert new_state == state


# ---------- JSON parsing robustness ----------

def test_verdict_json_in_code_fence_parses(david_profile, fake_llm):
    fenced = "```json\n" + _verdict_response("pass") + "\n```"
    fake_llm.script.append(fenced)
    reviewer = ReviewerAgent(llm=fake_llm)
    state = initial_state(david_profile)
    state["conversation_history"].append(
        _advice_msg(f"Diversify. Disclaimer: {STANDARD_DISCLAIMER}")
    )
    new_state = reviewer.process(state)
    assert new_state["conversation_history"][-1].recipient is AgentRole.CLIENT
    assert new_state["last_review"]["verdict"] == "pass"


def test_malformed_verdict_fails_open(david_profile, fake_llm):
    """A garbage reviewer response must default to pass (fail-open, never deadlock)."""
    fake_llm.script.append("not a json at all")
    reviewer = ReviewerAgent(llm=fake_llm)
    state = initial_state(david_profile)
    state["conversation_history"].append(
        _advice_msg(f"Diversify. Disclaimer: {STANDARD_DISCLAIMER}")
    )
    new_state = reviewer.process(state)
    last = new_state["conversation_history"][-1]
    assert last.recipient is AgentRole.CLIENT
    assert last.metadata["review"]["verdict"] == "pass"


# ---------- factory wiring ----------

def test_reviewer_factory_via_agent_factory(fake_llm):
    from src.factories.agent_factory import AgentFactory

    reviewer = AgentFactory.create(AgentRole.REVIEWER, llm=fake_llm)
    assert isinstance(reviewer, ReviewerAgent)


# ---------- direct `review()` API ----------

@pytest.mark.parametrize(
    "content,expected_verdict",
    [
        (
            f"Diversify across asset classes. Disclaimer: {STANDARD_DISCLAIMER}",
            "pass",
        ),
        (
            "Buy MSFT for growth.",
            "revise",  # ticker + missing disclaimer → deterministic revise
        ),
    ],
)
def test_review_direct_verdicts(david_profile, fake_llm, content, expected_verdict):
    fake_llm.script.append(_verdict_response("pass"))  # let determinism dominate
    reviewer = ReviewerAgent(llm=fake_llm)
    verdict: ReviewVerdict = reviewer.review(_advice_msg(content))
    assert verdict.verdict == expected_verdict
