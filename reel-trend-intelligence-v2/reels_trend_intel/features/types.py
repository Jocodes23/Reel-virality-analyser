"""Typed feature payloads produced by the extraction stage.

These are pure data (pydantic v2) with no dependency on storage, so both the
storage layer and the report layer can import them without cycles. Embedding
*vectors* are NOT stored here; they live in the content-hash-keyed embedding
cache. This struct keeps the scalar/structured features plus the content-hash
keys that point at the cached vectors.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ColorFeatures(BaseModel):
    dominant_hex: str
    accent_hex: str
    background_hex: str
    # (hex, proportion) sorted by proportion desc; proportions sum ~1.0.
    palette: list[tuple[str, float]]
    luminance: float            # 0..1 mean perceived luminance
    saturation: float           # 0..1 mean saturation
    warm_cool_ratio: float      # warm pixels / cool pixels
    contrast: float             # 0..1 std of luminance
    key: str                    # "high-key" | "low-key" | "mid-key"


class SceneFeatures(BaseModel):
    label: str                  # top zero-shot label
    scores: dict[str, float]    # label -> probability
    is_niche: bool = False      # matches the configured niche


class AudioFeatures(BaseModel):
    audio_id: str | None = None
    fingerprint: str | None = None     # hex acoustic fingerprint (for song clustering)
    tempo: float | None = None
    energy: float | None = None
    is_original_audio: bool = False


class CaptionFeatures(BaseModel):
    text: str = ""
    hashtags: list[str] = Field(default_factory=list)
    length: int = 0
    emoji_density: float = 0.0
    has_cta: bool = False
    language: str = "und"
    ocr_text: str | None = None        # only present if OCR sampled this reel


class FormatFeatures(BaseModel):
    duration_s: float | None = None
    cut_rate: float | None = None      # cuts per second (scene-change estimate)
    has_face: bool = False
    hour_of_day: int = 0               # TZ-normalized 0..23
    day_of_week: int = 0               # 0=Mon .. 6=Sun


class ReelFeatures(BaseModel):
    reel_id: str
    feature_schema_version: str
    color: ColorFeatures
    scene: SceneFeatures
    audio: AudioFeatures
    caption: CaptionFeatures
    fmt: FormatFeatures
    # Content-hash keys pointing into the embedding cache (never recompute).
    image_hash: str
    audio_hash: str | None = None
    caption_hash: str | None = None

    def to_json(self) -> dict[str, object]:
        return self.model_dump(mode="json")
