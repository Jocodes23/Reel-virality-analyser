"""The versioned, downstream-agnostic TrendReport schema (sections A-F).

Stable wire format: a ranked list of TrendRecord. JSONL/Parquet exports carry the
FULL record incl. member drill-down; the human CSV/Excel carries one row per trend
with the two clickable links. Nothing collected is dropped from the full export.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


# ---------------------------------------------------------------- A. headline
class ExternalLinks(BaseModel):
    spotify: str | None = None
    apple_music: str | None = None
    youtube: str | None = None


class TopReel(BaseModel):
    url: str                       # canonical permalink
    reel_id: str
    author_handle: str
    author_url: str
    plays: int
    likes: int
    comments: int
    shares: int
    saves: int
    posted_at: datetime
    why_top: str                   # the metric/value that won


class AudioInfo(BaseModel):
    audio_id: str
    instagram_audio_url: str
    song_title: str | None = None
    artist: str | None = None
    is_original_audio: bool = False
    usage_count: int | None = None
    external_links: ExternalLinks = Field(default_factory=ExternalLinks)


class HeadlineLinks(BaseModel):
    top_reel: TopReel
    audio: AudioInfo


# ---------------------------------------------------------------- B. identity
class Identity(BaseModel):
    trend_id: str
    type: str                      # audio|visual|caption|format
    label: str
    member_reel_ids: list[str]
    size: int
    first_seen: datetime


# ---------------------------------------------------------------- C. dynamics
class ConfidenceInterval(BaseModel):
    low: float
    high: float


class Dynamics(BaseModel):
    phase: str                     # pre-peak|peak|decay
    velocity: float
    acceleration: float
    r_t: float                     # effective reproduction number R(t)
    trend_health_score: float
    persistence_probability: float
    persistence_ci: ConfidenceInterval
    persistence_horizon_days: float
    forecast_peak_time: datetime | None = None
    post_before: datetime | None = None
    supercritical: bool = False


# ---------------------------------------------------------------- D. creative
class PaletteColor(BaseModel):
    hex: str
    proportion: float


class CreativeSignal(BaseModel):
    palette: list[PaletteColor]
    mood: str                      # high-key/low-key etc.
    lighting: str
    composition: str
    warm_cool_ratio: float
    best_posting_hour: int
    best_posting_day: int
    caption_template: str
    hashtags: list[str]
    example_reel_ids: list[str]


# ---------------------------------------------------------------- E. metadata
class EngagementPoint(BaseModel):
    sampled_at: datetime
    plays: int
    likes: int
    comments: int
    shares: int
    saves: int


class MemberReel(BaseModel):
    reel_id: str
    permalink: str
    author_handle: str
    author_url: str
    posted_at: datetime
    audio_id: str | None = None
    plays: int = 0
    likes: int = 0
    comments: int = 0
    shares: int = 0
    saves: int = 0
    engagement_series: list[EngagementPoint] = Field(default_factory=list)
    color: dict = Field(default_factory=dict)
    audio: dict = Field(default_factory=dict)
    caption: dict = Field(default_factory=dict)
    fmt: dict = Field(default_factory=dict)


class TrendMetadata(BaseModel):
    aggregated_features: dict = Field(default_factory=dict)
    members: list[MemberReel] = Field(default_factory=list)


# ---------------------------------------------------------------- F. provenance
class Provenance(BaseModel):
    model_versions: dict[str, str]
    feature_schema_version: str
    report_version: str
    sample_size: int
    data_last_updated: datetime
    calibration: dict = Field(default_factory=dict)


# ---------------------------------------------------------------- record + report
class TrendRecord(BaseModel):
    headline_links: HeadlineLinks
    identity: Identity
    dynamics: Dynamics
    creative_signal: CreativeSignal
    metadata: TrendMetadata
    provenance: Provenance
    # ranking scores carried for transport (sorting key chosen at query time)
    health_score: float
    opportunity_score: float
    is_whitespace: bool


class TrendReport(BaseModel):
    report_version: str
    generated_at: datetime
    ranking_key: str               # "health" | "opportunity"
    niche: str
    provenance: Provenance
    trends: list[TrendRecord]      # ranked, most-famous / best-opportunity first

    def to_json(self) -> dict:
        return self.model_dump(mode="json")
