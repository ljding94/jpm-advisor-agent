"""Tests for src/ui/persona_editor.py — round-trip and IO helpers."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.schemas import ClientProfile
from src.ui.persona_editor import (
    form_dict_to_profile,
    load_persona,
    profile_to_form_dict,
    save_persona,
)


def test_load_persona_by_key():
    p = load_persona("david")
    assert isinstance(p, ClientProfile)
    assert "David" in p.name


def test_load_persona_by_path(tmp_path):
    profile = ClientProfile(
        name="Test", age=30, risk_tolerance="moderate",
        assets={"cash": 1000.0}, investments=[],
        goals=["retire eventually"],
        time_horizon_years=20, annual_income=50000.0,
    )
    p_path = tmp_path / "custom.json"
    p_path.write_text(profile.model_dump_json())
    loaded = load_persona(p_path)
    assert loaded.name == "Test"


def test_round_trip_form_dict():
    original = load_persona("priya")
    form = profile_to_form_dict(original)
    rebuilt = form_dict_to_profile(form)
    assert rebuilt.name == original.name
    assert rebuilt.age == original.age
    assert rebuilt.risk_tolerance == original.risk_tolerance
    assert rebuilt.assets == original.assets
    # investments may have field reordering — compare via model_dump
    assert [i.model_dump() for i in rebuilt.investments] == \
           [i.model_dump() for i in original.investments]
    assert rebuilt.goals == original.goals
    assert rebuilt.time_horizon_years == original.time_horizon_years
    assert rebuilt.annual_income == original.annual_income


def test_form_dict_invalid_json():
    form = {
        "name": "X", "age": 30, "risk_tolerance": "moderate",
        "annual_income": 50000.0, "time_horizon_years": 10,
        "goals": "g1\ng2",
        "notes": "",
        "assets_json": "{not valid json",
        "investments_json": "[]",
    }
    with pytest.raises(ValueError, match="invalid JSON"):
        form_dict_to_profile(form)


def test_form_dict_invalid_age():
    form = {
        "name": "X", "age": 200, "risk_tolerance": "moderate",
        "annual_income": 50000.0, "time_horizon_years": 10,
        "goals": "",
        "notes": "",
        "assets_json": "{}",
        "investments_json": "[]",
    }
    with pytest.raises(Exception):  # pydantic ValidationError
        form_dict_to_profile(form)


def test_save_persona_writes_file(tmp_path):
    profile = ClientProfile(
        name="Saver", age=45, risk_tolerance="moderate",
        assets={"savings": 100.0}, investments=[],
        goals=["save more"],
        time_horizon_years=15, annual_income=80000.0,
    )
    out = save_persona(profile, "saver_test", dir_=tmp_path)
    assert out.exists()
    parsed = ClientProfile(**json.loads(out.read_text()))
    assert parsed.name == "Saver"
    assert parsed.assets == {"savings": 100.0}
