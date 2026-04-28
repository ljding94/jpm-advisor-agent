"""Analyst report and final advice schemas."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

STANDARD_DISCLAIMER = (
    "This is not financial advice. Consult a licensed financial professional "
    "before making investment decisions."
)


class Source(BaseModel):
    title: str
    url: str = ""
    snippet: str = ""

    @field_validator("title")
    @classmethod
    def _non_empty_title(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("source title must be non-empty")
        return v


class AnalystReport(BaseModel):
    query: str
    findings: str
    sources: list[Source] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class AdviceOutput(BaseModel):
    recommendations: list[str] = Field(min_length=1)
    rationale: str
    sources: list[Source] = Field(default_factory=list)
    disclaimers: list[str] = Field(default_factory=list)

    @field_validator("disclaimers")
    @classmethod
    def _ensure_standard_disclaimer(cls, v: list[str]) -> list[str]:
        if not any("not financial advice" in d.lower() for d in v):
            v = [*v, STANDARD_DISCLAIMER]
        return v
