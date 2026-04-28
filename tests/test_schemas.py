"""Schema tests including the hard routing constraint."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas import (
    STANDARD_DISCLAIMER,
    AdviceOutput,
    AgentMessage,
    AgentRole,
    AnalystReport,
    ClientProfile,
    Investment,
    MessageType,
    Source,
)


# ----------------------- AgentMessage routing -----------------------

@pytest.mark.parametrize(
    "sender,recipient",
    [
        (AgentRole.CLIENT, AgentRole.ADVISOR),
        (AgentRole.ADVISOR, AgentRole.CLIENT),
        (AgentRole.ADVISOR, AgentRole.ANALYST),
        (AgentRole.ANALYST, AgentRole.ADVISOR),
    ],
)
def test_legal_routes_accepted(sender, recipient):
    msg = AgentMessage(
        sender=sender,
        recipient=recipient,
        content="ping",
        message_type=MessageType.QUESTION,
    )
    assert msg.sender is sender
    assert msg.recipient is recipient


def test_analyst_to_client_rejected():
    """Acceptance criterion #5: Analyst→Client must raise validation error."""
    with pytest.raises(ValidationError) as exc_info:
        AgentMessage(
            sender=AgentRole.ANALYST,
            recipient=AgentRole.CLIENT,
            content="hi client",
        )
    assert "Illegal route" in str(exc_info.value)


def test_client_to_analyst_rejected():
    with pytest.raises(ValidationError):
        AgentMessage(
            sender=AgentRole.CLIENT,
            recipient=AgentRole.ANALYST,
            content="hi analyst",
        )


def test_self_loop_rejected():
    with pytest.raises(ValidationError):
        AgentMessage(
            sender=AgentRole.ADVISOR,
            recipient=AgentRole.ADVISOR,
            content="self",
        )


# ----------------------- ClientProfile -----------------------

def test_client_profile_validates_age_bounds():
    with pytest.raises(ValidationError):
        ClientProfile(
            name="Too Young",
            age=10,
            risk_tolerance="moderate",
            time_horizon_years=10,
            annual_income=50_000,
        )


def test_client_profile_total_assets():
    p = ClientProfile(
        name="Test",
        age=40,
        risk_tolerance="moderate",
        assets={"cash": 10_000.0, "stocks": 90_000.0},
        time_horizon_years=20,
        annual_income=120_000,
    )
    assert p.total_assets == pytest.approx(100_000.0)


def test_client_profile_rejects_negative_asset():
    with pytest.raises(ValidationError):
        ClientProfile(
            name="Bad",
            age=40,
            risk_tolerance="moderate",
            assets={"cash": -1.0},
            time_horizon_years=20,
            annual_income=120_000,
        )


def test_investment_negative_value_rejected():
    with pytest.raises(ValidationError):
        Investment(name="X", asset_class="equity", value_usd=-5)


# ----------------------- AnalystReport -----------------------

def test_analyst_report_requires_sources():
    with pytest.raises(ValidationError):
        AnalystReport(query="q", findings="f", sources=[], confidence=0.5)


def test_analyst_report_confidence_bounds():
    with pytest.raises(ValidationError):
        AnalystReport(
            query="q",
            findings="f",
            sources=[Source(title="s")],
            confidence=1.5,
        )


# ----------------------- AdviceOutput -----------------------

def test_advice_output_force_appends_disclaimer():
    advice = AdviceOutput(
        recommendations=["Diversify."],
        rationale="Risk reduction.",
        sources=[],
        disclaimers=[],
    )
    assert any("not financial advice" in d.lower() for d in advice.disclaimers)
    assert STANDARD_DISCLAIMER in advice.disclaimers


def test_advice_output_keeps_existing_disclaimer():
    advice = AdviceOutput(
        recommendations=["X"],
        rationale="Y",
        disclaimers=["This is not financial advice — see a pro."],
    )
    assert len(advice.disclaimers) == 1


def test_advice_output_requires_at_least_one_recommendation():
    with pytest.raises(ValidationError):
        AdviceOutput(recommendations=[], rationale="r")


def test_source_title_required():
    with pytest.raises(ValidationError):
        Source(title="   ")
