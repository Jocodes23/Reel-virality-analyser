"""Ranking keys for the report (default = Trend Health Score; alt = opportunity)."""

from __future__ import annotations

from reels_trend_intel.report.schema import TrendRecord


def _health_key(r: TrendRecord) -> tuple[float, int, float]:
    # most-famous current trends first
    return (r.health_score, r.identity.size, r.dynamics.r_t)


def _opportunity_key(r: TrendRecord) -> tuple[float, float, int]:
    return (r.opportunity_score, r.health_score, r.identity.size)


def sort_records(records: list[TrendRecord], ranking_key: str) -> list[TrendRecord]:
    key = _opportunity_key if ranking_key == "opportunity" else _health_key
    return sorted(records, key=key, reverse=True)
