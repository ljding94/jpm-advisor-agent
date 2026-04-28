"""All three persona JSONs load and validate as ClientProfile."""
from __future__ import annotations

import pytest

from src.schemas import ClientProfile


@pytest.mark.parametrize("key", ["margaret", "david", "priya"])
def test_persona_loads_and_validates(all_personas: dict[str, ClientProfile], key: str):
    profile = all_personas[key]
    assert profile.name
    assert profile.total_assets > 0
    assert profile.goals
    assert profile.risk_tolerance in ("conservative", "moderate", "aggressive")


def test_persona_risk_tolerances_match_names(all_personas: dict[str, ClientProfile]):
    assert all_personas["margaret"].risk_tolerance == "conservative"
    assert all_personas["david"].risk_tolerance == "moderate"
    assert all_personas["priya"].risk_tolerance == "aggressive"
