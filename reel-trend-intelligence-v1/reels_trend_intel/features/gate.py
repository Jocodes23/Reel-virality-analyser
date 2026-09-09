"""Cheap gate (tiered compute): run BEFORE any expensive CV/audio/OCR model.

Filters out duplicates (perceptual-hash near-match) and low-relevance/low-quality
reels, so the GPU models only ever touch survivors. This is the second efficiency
lever after adaptive resampling.
"""

from __future__ import annotations

from dataclasses import dataclass

from reels_trend_intel.config.settings import FeatureConfig
from reels_trend_intel.storage.base import StorageBackend
from reels_trend_intel.storage.rows import EngagementSample


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reason: str
    duplicate_of: str | None = None


class CheapGate:
    def __init__(self, cfg: FeatureConfig) -> None:
        self.cfg = cfg

    async def check(
        self,
        reel_id: str,
        phash: int,
        latest: EngagementSample | None,
        storage: StorageBackend,
    ) -> GateResult:
        # 1) quality / relevance floor
        plays = latest.plays if latest else 0
        if plays < self.cfg.gate_min_plays:
            return GateResult(False, "below_quality_floor")
        # 2) perceptual-hash dedupe (skip re-processing near-identical media)
        dup = await storage.seen_phash(phash, self.cfg.phash_hamming_dup)
        if dup is not None and dup != reel_id:
            return GateResult(False, "duplicate", duplicate_of=dup)
        return GateResult(True, "ok")
