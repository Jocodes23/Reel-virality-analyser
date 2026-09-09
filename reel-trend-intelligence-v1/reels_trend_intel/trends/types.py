"""Shared trend types consumed by the modelling layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class AdoptionCurve:
    """N(t): new adopting reels per time bucket, plus aggregate engagement."""

    bucket_s: int
    start: datetime
    counts: list[int] = field(default_factory=list)          # new reels per bucket
    plays: list[float] = field(default_factory=list)         # aggregate plays per bucket

    @property
    def total(self) -> int:
        return int(sum(self.counts))


@dataclass
class BuiltTrend:
    trend_id: str
    type: str                       # audio|visual|caption|format
    label: str
    member_ids: list[str]
    first_seen: datetime
    size: int
    curve: AdoptionCurve
    # Posting times (hours since first_seen) — the event stream for the Hawkes model.
    event_times_h: list[float] = field(default_factory=list)
    top_terms: list[str] = field(default_factory=list)
