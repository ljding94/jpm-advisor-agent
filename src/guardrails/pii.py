"""PII redaction. Run on every string before it reaches the LLM."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Patterns are intentionally broader than strict definitions to err on the side
# of redaction. Order matters — more specific patterns first.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # SSN: 123-45-6789 or 123 45 6789 (also matches with no separators if surrounded by non-digits)
    ("ssn", re.compile(r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b")),
    ("ssn", re.compile(r"(?<!\d)\d{9}(?!\d)")),
    # Credit-card: 13–19 digits with optional spaces or dashes
    ("credit_card", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    # Bank/account number heuristic: 8–17 digit run not adjacent to other digits
    ("account_number", re.compile(r"(?<!\d)\d{8,17}(?!\d)")),
    # Email
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    # US phone: (123) 456-7890, 123-456-7890, +1 123 456 7890
    ("phone", re.compile(r"(?:\+?1[-\s.]?)?(?:\(\d{3}\)|\d{3})[-\s.]\d{3}[-\s.]\d{4}")),
]


@dataclass
class RedactionResult:
    text: str
    counts: dict[str, int]

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def redact(text: str) -> RedactionResult:
    """Redact PII in `text`. Returns the redacted text + counts per category."""
    counts: dict[str, int] = {}
    redacted = text
    for label, pattern in _PATTERNS:
        def _replace(_m: re.Match[str], _label: str = label) -> str:
            counts[_label] = counts.get(_label, 0) + 1
            return f"[REDACTED_{_label.upper()}]"

        redacted = pattern.sub(_replace, redacted)

    if counts:
        logger.info("pii.redaction", extra={"counts": counts})
    return RedactionResult(text=redacted, counts=counts)


def redact_text(text: str) -> str:
    """Convenience wrapper returning only the redacted string."""
    return redact(text).text


def redact_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Redact PII in every message's `content` field."""
    return [
        {**m, "content": redact_text(m.get("content", ""))}
        for m in messages
    ]
