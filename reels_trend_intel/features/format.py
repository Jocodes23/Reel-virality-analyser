"""Format/timing features.

Timing (hour-of-day, day-of-week) is computed from the post timestamp and is
fully real offline. Duration / cut-rate / has-face come from provider-supplied
metadata when available (Graph API returns duration; the sample adapter supplies
the synthetic clip profile), else are left unknown.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from reels_trend_intel.features.types import FormatFeatures


def extract_format(
    posted_at: datetime, metadata: dict[str, Any] | None = None,
    tz_offset_hours: float = 0.0,
) -> FormatFeatures:
    meta = metadata or {}
    local = posted_at.astimezone(UTC)
    # TZ-normalize to the author/local timezone if an offset is supplied.
    hour = int((local.hour + tz_offset_hours) % 24)
    dow = local.weekday()
    return FormatFeatures(
        duration_s=_as_float(meta.get("duration_s")),
        cut_rate=_as_float(meta.get("cut_rate")),
        has_face=bool(meta.get("has_face", False)),
        hour_of_day=hour,
        day_of_week=dow,
    )


def _as_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
