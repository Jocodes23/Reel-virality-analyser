"""Adaptive engagement re-sampling — the single biggest efficiency lever.

We NEVER poll everything on a fixed interval. Each tracked reel gets its own
next-sample time derived from its age and recent acceleration:

    interval = clamp(
        base * 2**(age_hours / age_halflife_h) / (1 + accel_weight * |accel_norm|),
        min_interval, max_interval)

Young/accelerating reels -> short interval (sampled often). Old/decaying reels
-> long interval. Dead reels (no growth across the retirement window) are RETIRED
and never polled again. Under budget pressure only HOT reels are sampled.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from reels_trend_intel.config.settings import SamplingConfig
from reels_trend_intel.storage.rows import EngagementSample


def tier_for_age(age_h: float, cfg: SamplingConfig) -> str:
    if age_h <= cfg.hot_max_age_h:
        return "hot"
    if age_h <= cfg.warm_max_age_h:
        return "warm"
    if age_h <= cfg.cold_max_age_h:
        return "cold"
    return "cold"


def normalized_acceleration(series: list[EngagementSample]) -> float:
    """Unitless acceleration of plays from the last three samples.

    accel_norm = (v_last - v_prev) / (level_per_hour + 1), where v_* are
    plays-per-hour velocities. Positive => accelerating; negative => decelerating.
    """
    if len(series) < 3:
        return 0.0
    a, b, c = series[-3], series[-2], series[-1]
    dt1 = max((b.sampled_at - a.sampled_at).total_seconds() / 3600.0, 1e-6)
    dt2 = max((c.sampled_at - b.sampled_at).total_seconds() / 3600.0, 1e-6)
    v1 = (b.plays - a.plays) / dt1
    v2 = (c.plays - b.plays) / dt2
    level = max(1.0, (c.plays - a.plays) / max(dt1 + dt2, 1e-6))
    return (v2 - v1) / level


def relative_growth(series: list[EngagementSample], window_h: float) -> float:
    """Fractional plays growth over the trailing `window_h` hours."""
    if len(series) < 2:
        return 1.0
    end = series[-1]
    cutoff = end.sampled_at - timedelta(hours=window_h)
    base = None
    for s in series:
        if s.sampled_at <= cutoff:
            base = s
    if base is None:
        base = series[0]
    denom = max(1, base.plays)
    return (end.plays - base.plays) / denom


@dataclass(frozen=True)
class SamplingDecision:
    next_at: datetime
    tier: str
    retire: bool
    interval_s: float


def plan_next(
    posted_at: datetime,
    series: list[EngagementSample],
    now: datetime,
    cfg: SamplingConfig,
) -> SamplingDecision:
    age_h = max(0.0, (now - posted_at).total_seconds() / 3600.0)
    tier = tier_for_age(age_h, cfg)

    # Retirement: only consider once the reel is old enough to have a full window
    # and we have enough samples to judge growth.
    if age_h >= cfg.retire_after_h and len(series) >= 3:
        if relative_growth(series, cfg.retire_after_h) < cfg.retire_min_rel_growth:
            return SamplingDecision(now, "dead", True, cfg.max_interval_s)

    accel = abs(normalized_acceleration(series))
    interval = cfg.base_interval_s * (2.0 ** (age_h / cfg.age_halflife_h))
    interval = interval / (1.0 + cfg.accel_weight * accel)
    interval = max(cfg.min_interval_s, min(cfg.max_interval_s, interval))
    return SamplingDecision(now + timedelta(seconds=interval), tier, False, interval)


def initial_schedule_time(posted_at: datetime, now: datetime, cfg: SamplingConfig) -> datetime:
    """First sample is due immediately (fresh reels are the hottest)."""
    return now if now >= posted_at else posted_at


def utcnow() -> datetime:
    return datetime.now(UTC)
