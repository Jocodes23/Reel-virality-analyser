"""Domain entities persisted by the StorageBackend.

Canonical URL shapes (collected/persisted for EVERY reel and audio):
    reel permalink : https://www.instagram.com/reel/{shortcode}/
    audio page     : https://www.instagram.com/reels/audio/{audio_id}/
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


def canonical_permalink(shortcode: str) -> str:
    return f"https://www.instagram.com/reel/{shortcode}/"


def canonical_audio_url(audio_id: str) -> str:
    return f"https://www.instagram.com/reels/audio/{audio_id}/"


class RawPayload(BaseModel):
    """Verbatim payload landed BEFORE any transform (replayable)."""

    source: str
    kind: str                      # "reel" | "engagement" | "audio" | ...
    fetched_at: datetime
    content_hash: str
    payload: dict[str, Any]


class Reel(BaseModel):
    reel_id: str                   # canonical id (== shortcode for IG reels)
    shortcode: str
    permalink: str
    author_handle: str
    author_url: str
    posted_at: datetime
    audio_id: str | None = None
    caption: str = ""
    source: str = "sample"
    media_url: str | None = None   # transient fetch URL (not the permalink)
    phash: int | None = None       # perceptual hash for dedupe
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    tracking_tier: str = "hot"     # hot|warm|cold|dead
    retired: bool = False

    @classmethod
    def from_shortcode(cls, shortcode: str, **kw: Any) -> Reel:
        return cls(
            reel_id=shortcode,
            shortcode=shortcode,
            permalink=canonical_permalink(shortcode),
            **kw,
        )


class Audio(BaseModel):
    audio_id: str
    instagram_audio_url: str
    song_title: str | None = None
    artist: str | None = None
    is_original_audio: bool = False
    usage_count: int | None = None
    external_links: dict[str, str] = Field(default_factory=dict)  # spotify/apple_music/youtube
    fingerprint_cluster_id: int | None = None

    @classmethod
    def from_id(cls, audio_id: str, **kw: Any) -> Audio:
        return cls(
            audio_id=audio_id,
            instagram_audio_url=canonical_audio_url(audio_id),
            **kw,
        )


class EngagementSample(BaseModel):
    """One point in a reel's engagement TIME SERIES (hypertable row)."""

    reel_id: str
    sampled_at: datetime
    plays: int = 0
    likes: int = 0
    comments: int = 0
    shares: int = 0
    saves: int = 0

    def engagement_rate(self) -> float:
        if self.plays <= 0:
            return 0.0
        return (self.likes + self.comments + self.shares + self.saves) / self.plays


class EmbeddingRow(BaseModel):
    content_hash: str
    kind: str                      # "image" | "audio" | "caption"
    vector: list[float]


class TrendRow(BaseModel):
    trend_id: str
    type: str                      # audio|visual|caption|format
    label: str
    size: int
    first_seen: datetime
    status: str = "active"         # active|dead


class TrendModelRow(BaseModel):
    trend_id: str
    fitted_at: datetime
    payload: dict[str, Any]        # full fitted model snapshot (hawkes/survival/spikem/...)
    health_score: float
    persistence_prob: float
    persistence_ci_low: float
    persistence_ci_high: float
    phase: str                     # pre-peak|peak|decay
    model_versions: dict[str, str]


class ReelSchedule(BaseModel):
    """Adaptive-resampler state for one tracked reel."""

    reel_id: str
    posted_at: datetime
    next_sample_at: datetime
    tier: str = "hot"
    last_plays: int = 0
    retired: bool = False


class BudgetState(BaseModel):
    adapter: str
    day: str                       # YYYY-MM-DD (UTC)
    requests_used: int = 0
    cap: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.cap - self.requests_used)

    @property
    def remaining_fraction(self) -> float:
        return self.remaining / self.cap if self.cap > 0 else 0.0
