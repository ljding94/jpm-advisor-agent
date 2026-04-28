"""Client profile schema."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

RiskTolerance = Literal["conservative", "moderate", "aggressive"]


class Investment(BaseModel):
    name: str
    asset_class: str
    value_usd: float = Field(ge=0)


class ClientProfile(BaseModel):
    name: str
    age: int = Field(ge=18, le=100)
    risk_tolerance: RiskTolerance
    assets: dict[str, float] = Field(default_factory=dict)
    investments: list[Investment] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    time_horizon_years: int = Field(ge=0, le=80)
    annual_income: float = Field(ge=0)
    notes: str = ""

    @field_validator("assets")
    @classmethod
    def _non_negative_assets(cls, v: dict[str, float]) -> dict[str, float]:
        for k, val in v.items():
            if val < 0:
                raise ValueError(f"asset {k!r} must be non-negative, got {val}")
        return v

    @property
    def total_assets(self) -> float:
        return sum(self.assets.values())
