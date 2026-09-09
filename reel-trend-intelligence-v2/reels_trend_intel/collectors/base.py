"""Pluggable collection adapters.

Every adapter sits behind the single polite, rate-limited, budgeted scheduler
(see scheduler.py). Adapters return canonical domain objects (Reel, Audio) and
provide media bytes for the feature stage. The `at` argument on fetch_engagement
lets the OFFLINE sample adapter serve a point on a simulated engagement curve;
live adapters ignore it and read "now".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from reels_trend_intel.storage.rows import Audio, EngagementSample, Reel


class CollectorHalted(Exception):
    """Raised when an adapter hits a login challenge/checkpoint and HALTS.

    The scheduler never routes around this; it stops the adapter and surfaces it.
    """


class CollectorAdapter(ABC):
    name: str = "base"

    @property
    def halted(self) -> bool:
        return getattr(self, "_halted", False)

    def halt(self, reason: str) -> None:
        self._halted = True
        self._halt_reason = reason

    @abstractmethod
    async def discover(self, limit: int) -> list[tuple[Reel, Audio | None]]:
        """Return newly-seen public reels (+ audio) to begin tracking."""

    @abstractmethod
    async def fetch_engagement(
        self, reel: Reel, at: datetime | None = None
    ) -> EngagementSample | None:
        """Return the current (or simulated-at-`at`) engagement snapshot."""

    @abstractmethod
    async def fetch_media(self, reel: Reel) -> bytes | None:
        """Return a representative keyframe (PNG/JPEG bytes) for feature extraction."""

    async def fetch_metadata(self, reel: Reel) -> dict[str, object]:
        """Provider-supplied clip metadata (duration_s, cut_rate, has_face, ...).

        Optional; defaults to empty. Graph API fills duration from media metadata;
        the sample adapter supplies the synthetic clip profile.
        """
        return {}

    async def close(self) -> None:  # pragma: no cover - default no-op
        return None
