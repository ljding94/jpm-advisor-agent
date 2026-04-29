"""TurnLogger subclass that fires a callback after each recorded turn.

The graph wrapper calls `turn_logger.record_turn(...)` once per agent turn.
By subclassing TurnLogger we get a stream-like notification surface for the UI
without changing the wrapper or any agent code.
"""
from __future__ import annotations

from typing import Callable

from src.observability.logger import TurnLogger, TurnRecord


class UITurnLogger(TurnLogger):
    """TurnLogger that invokes `on_turn(record)` after each recorded turn.

    Callbacks must be cheap and non-blocking — they run inline on the graph
    thread. For Streamlit we just push the record onto a thread-safe queue
    and let the main UI thread re-render.
    """

    def __init__(self, on_turn: Callable[[TurnRecord], None] | None = None) -> None:
        super().__init__()
        self._on_turn = on_turn

    def set_callback(self, on_turn: Callable[[TurnRecord], None] | None) -> None:
        self._on_turn = on_turn

    def record_turn(self, **kwargs) -> TurnRecord:
        rec = super().record_turn(**kwargs)
        if self._on_turn is not None:
            try:
                self._on_turn(rec)
            except Exception:  # pragma: no cover - never let UI errors kill the graph
                pass
        return rec
