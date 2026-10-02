"""Causal event selection for asynchronous sensor and image channels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ObservationEvent:
    capture_tick: int
    receive_tick: int
    payload_ref: str


def latest_available(events: Iterable[ObservationEvent], decision_tick: int) -> ObservationEvent | None:
    """Return newest event captured and received by decision time; never backfill late data."""
    available = [e for e in events if e.capture_tick <= decision_tick and e.receive_tick <= decision_tick]
    if not available:
        return None
    return max(available, key=lambda e: (e.capture_tick, e.receive_tick))
