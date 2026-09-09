"""Offline 'sample' adapter — wraps the synthetic dataset as a CollectorAdapter.

Reveals reels as they are "posted" relative to a simulated clock and serves
engagement at any simulated time, so the adaptive resampler and the whole
collect->extract->cluster->model->report pipeline run with ZERO network.
"""

from __future__ import annotations

from datetime import datetime

from reels_trend_intel.collectors.base import CollectorAdapter
from reels_trend_intel.fixtures.generate import SyntheticDataset
from reels_trend_intel.storage.rows import Audio, EngagementSample, Reel


class SampleAdapter(CollectorAdapter):
    name = "sample"

    def __init__(self, dataset: SyntheticDataset | None = None) -> None:
        self.dataset = dataset or SyntheticDataset()
        self._all = self.dataset.reels_sorted()
        self._revealed = 0
        self.clock: datetime = self.dataset.sim_start

    def set_clock(self, now: datetime) -> None:
        self.clock = now

    async def discover(self, limit: int) -> list[tuple[Reel, Audio | None]]:
        out: list[tuple[Reel, Audio | None]] = []
        while self._revealed < len(self._all) and len(out) < limit:
            reel = self._all[self._revealed]
            if reel.posted_at > self.clock:
                break  # not yet posted in simulated time
            audio = self.dataset.audio_for(reel.reel_id)
            out.append((reel, audio))
            self._revealed += 1
        return out

    async def fetch_engagement(
        self, reel: Reel, at: datetime | None = None
    ) -> EngagementSample | None:
        return self.dataset.engagement_at(reel.reel_id, at or self.clock)

    async def fetch_media(self, reel: Reel) -> bytes | None:
        return self.dataset.render_image(reel.reel_id)

    async def fetch_metadata(self, reel: Reel) -> dict[str, object]:
        return self.dataset.format_hints(reel.reel_id)
