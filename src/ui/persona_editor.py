"""Helpers for the Streamlit persona-editor mode.

Pure data-layer functions so they're testable without Streamlit imported.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.schemas import ClientProfile, Investment

PERSONA_DIR = Path("data/personas")
PERSONA_FILES = {
    "margaret": "margaret_conservative.json",
    "david": "david_moderate.json",
    "priya": "priya_aggressive.json",
}


def load_persona(key_or_path: str | Path) -> ClientProfile:
    """Load by short key (margaret/david/priya) or by file path."""
    if isinstance(key_or_path, str) and key_or_path in PERSONA_FILES:
        path = PERSONA_DIR / PERSONA_FILES[key_or_path]
    else:
        path = Path(key_or_path)
    return ClientProfile(**json.loads(path.read_text(encoding="utf-8")))


def profile_to_form_dict(p: ClientProfile) -> dict:
    """Flat dict suitable for binding to Streamlit input widgets."""
    return {
        "name": p.name,
        "age": p.age,
        "risk_tolerance": p.risk_tolerance,
        "annual_income": p.annual_income,
        "time_horizon_years": p.time_horizon_years,
        "goals": "\n".join(p.goals),
        "notes": p.notes or "",
        "assets_json": json.dumps(p.assets, indent=2),
        "investments_json": json.dumps(
            [inv.model_dump() for inv in p.investments], indent=2
        ),
    }


def form_dict_to_profile(d: dict) -> ClientProfile:
    """Inverse of `profile_to_form_dict`. Raises ValueError on bad JSON or schema violations."""
    try:
        assets = json.loads(d["assets_json"]) if d.get("assets_json") else {}
        investments_raw = json.loads(d["investments_json"]) if d.get("investments_json") else []
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in assets/investments: {exc}") from exc

    investments = [Investment(**inv) for inv in investments_raw]
    goals = [g.strip() for g in (d.get("goals") or "").splitlines() if g.strip()]

    return ClientProfile(
        name=d["name"],
        age=int(d["age"]),
        risk_tolerance=d["risk_tolerance"],
        assets=assets,
        investments=investments,
        goals=goals,
        time_horizon_years=int(d["time_horizon_years"]),
        annual_income=float(d["annual_income"]),
        notes=d.get("notes") or "",
    )


def save_persona(p: ClientProfile, key: str, *, dir_: Path = PERSONA_DIR) -> Path:
    """Write `data/personas/<key>.json`. Creates the directory if needed."""
    dir_.mkdir(parents=True, exist_ok=True)
    out = dir_ / f"{key}.json"
    out.write_text(p.model_dump_json(indent=2), encoding="utf-8")
    return out


__all__ = [
    "PERSONA_DIR",
    "PERSONA_FILES",
    "form_dict_to_profile",
    "load_persona",
    "profile_to_form_dict",
    "save_persona",
]
