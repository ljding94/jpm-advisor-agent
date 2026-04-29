"""Tests for src/ui/streaming.py — UITurnLogger callback firing."""
from __future__ import annotations

import time

from src.ui.streaming import UITurnLogger


def test_callback_fires_per_turn():
    received = []
    log = UITurnLogger(on_turn=lambda r: received.append(r))
    log.record_turn(agent="advisor", action="ask",
                    started_at=time.monotonic(),
                    input_tokens=10, output_tokens=5, cost_usd=0.001)
    log.record_turn(agent="analyst", action="report",
                    started_at=time.monotonic(),
                    input_tokens=20, output_tokens=8, cost_usd=0.002)
    assert len(received) == 2
    assert received[0].agent == "advisor"
    assert received[1].agent == "analyst"
    # Underlying TurnLogger contract still holds.
    assert log.summary()["turns"] == 2
    assert log.total_cost_usd == 0.003


def test_callback_can_be_swapped():
    log = UITurnLogger()
    bucket = []
    log.set_callback(lambda r: bucket.append(r.agent))
    log.record_turn(agent="client", action="answer", started_at=time.monotonic())
    assert bucket == ["client"]


def test_callback_exception_does_not_propagate():
    """A buggy UI callback must not blow up the graph thread."""
    def boom(_rec):
        raise RuntimeError("ui exploded")

    log = UITurnLogger(on_turn=boom)
    rec = log.record_turn(agent="advisor", action="x", started_at=time.monotonic())
    assert rec.agent == "advisor"
    assert log.summary()["turns"] == 1
