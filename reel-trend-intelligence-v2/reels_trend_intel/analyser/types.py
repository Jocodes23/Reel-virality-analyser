"""Typed analysis schema shared by every VLM provider.

One VLM call per reel returns this whole object as strict JSON, validated against
the pydantic schema below. It covers the *judgement* dimensions only:

    D1  cinematography / shot analysis
    D3  colour-grade LABEL (the numeric proxies are computed with OpenCV/numpy,
        never asked of the model)
    D5  format archetype
    D7  on-screen text template
    D8  structure (is_loopable is computed numerically from perceptual hashes)

Enums are deliberately CLOSED and versioned — `ANALYSER_SCHEMA_VERSION` is stamped
onto every row so older analyses stay identifiable and re-analysable when the
enums grow. Frequency of `other` / `unknown` is logged so the enums are only
extended on evidence.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

# Bump when the schema or enum members change.
ANALYSER_SCHEMA_VERSION = "1.0.0"


class ShotClass(StrEnum):
    CINEMATIC = "cinematic"
    POLISHED_HANDHELD = "polished_handheld"
    AMATEUR_HANDHELD = "amateur_handheld"
    HOMEMADE_STATIC = "homemade_static"
    SCREEN_RECORDING = "screen_recording"
    SLIDESHOW = "slideshow"
    MIXED = "mixed"


class CameraMovement(StrEnum):
    STATIC = "static"
    HANDHELD = "handheld"
    GIMBAL = "gimbal"
    DRONE = "drone"
    WHIP_PAN = "whip_pan"
    ORBIT = "orbit"


class Framing(StrEnum):
    CLOSE = "close"
    MEDIUM = "medium"
    WIDE = "wide"
    MIXED = "mixed"


class Lighting(StrEnum):
    NATURAL = "natural"
    RING_LIGHT = "ring_light"
    GOLDEN_HOUR = "golden_hour"
    LOW_LIGHT = "low_light"
    STUDIO = "studio"


class GradeClass(StrEnum):
    GRADED_STRONG = "graded_strong"
    GRADED_SUBTLE = "graded_subtle"
    UNGRADED = "ungraded"
    OVER_FILTERED = "over_filtered"


class Archetype(StrEnum):
    COMEDY_SKIT = "comedy_skit"
    DANCE = "dance"
    PRODUCT_SHOWCASE = "product_showcase"
    CAR_EDIT = "car_edit"
    TALKING_HEAD = "talking_head"
    POV = "pov"
    TRANSITION = "transition"
    TUTORIAL = "tutorial"
    TRANSFORMATION = "transformation"
    TRAVEL_MONTAGE = "travel_montage"
    FOOD = "food"
    PET = "pet"
    MEME_REPOST = "meme_repost"
    OTHER = "other"


class TextPosition(StrEnum):
    TOP = "top"
    CENTRE = "centre"
    BOTTOM = "bottom"
    MIXED = "mixed"
    NONE = "none"


class TextStyle(StrEnum):
    CAPTION_DEFAULT = "caption_default"
    BOLD_OUTLINE = "bold_outline"
    HANDWRITTEN = "handwritten"
    KINETIC = "kinetic"
    NONE = "none"


class Structure(StrEnum):
    SINGLE_TAKE = "single_take"
    MONTAGE = "montage"
    BEFORE_AFTER = "before_after"
    LIST = "list"
    NARRATIVE = "narrative"


class ShotAttributes(BaseModel):
    """D1 structured sub-fields."""

    model_config = ConfigDict(extra="forbid")

    camera_movement: CameraMovement
    framing: Framing
    lighting: Lighting
    stabilisation_quality: float = Field(ge=0.0, le=1.0)
    angle_variety: float = Field(ge=0.0, le=1.0)


class VLMAnalysis(BaseModel):
    """The single strict-JSON response returned by any VLM provider.

    `extra="forbid"` makes pydantic emit `additionalProperties: false`, which is
    what strict structured-output modes require.
    """

    model_config = ConfigDict(extra="forbid")

    # --- D1 cinematography -------------------------------------------------
    shot_class: ShotClass
    shot_description: str = Field(
        min_length=1,
        description="2-4 sentences describing HOW the reel was shot — camera "
        "movement, framing, lens feel, lighting, stabilisation, transitions. "
        "Never what the reel is about.",
    )
    shot_attributes: ShotAttributes
    shot_confidence: float = Field(ge=0.0, le=1.0)

    # --- D3 colour grade (judgement only; metrics computed numerically) ----
    grade_class: GradeClass
    grade_description: str = Field(min_length=1)
    grade_confidence: float = Field(ge=0.0, le=1.0)

    # --- D5 format archetype ----------------------------------------------
    archetype: Archetype
    archetype_confidence: float = Field(ge=0.0, le=1.0)

    # --- D7 on-screen text template ---------------------------------------
    text_present: bool
    text_position: TextPosition
    text_style: TextStyle
    text_density: float = Field(
        ge=0.0, le=1.0, description="Share of sampled frames carrying on-screen text."
    )

    # --- D8 structure (is_loopable computed numerically) -------------------
    structure: Structure


# Marker for a free-text field the model did not answer. Visible on purpose — an
# empty string would read as "the model said nothing about the craft", which is a
# different claim from "the model never answered".
NOT_SUPPLIED = "(not supplied by model)"

# Neutral values used when a model omits a field. They are deliberately the
# "don't know" member of each enum, and any confidence belonging to a defaulted
# dimension is forced to 0.0 — we degrade, we never fabricate.
_PARTIAL_DEFAULTS: dict[str, object] = {
    "shot_class": ShotClass.MIXED,
    "shot_description": NOT_SUPPLIED,
    "shot_attributes": {
        "camera_movement": CameraMovement.STATIC, "framing": Framing.MIXED,
        "lighting": Lighting.NATURAL, "stabilisation_quality": 0.0, "angle_variety": 0.0,
    },
    "shot_confidence": 0.0,
    "grade_class": GradeClass.UNGRADED,
    "grade_description": NOT_SUPPLIED,
    "grade_confidence": 0.0,
    "archetype": Archetype.OTHER,
    "archetype_confidence": 0.0,
    "text_present": False,
    "text_position": TextPosition.NONE,
    "text_style": TextStyle.NONE,
    "text_density": 0.0,
    "structure": Structure.SINGLE_TAKE,
}
# Confidence that must be zeroed when its dimension was defaulted.
_CONFIDENCE_OF: dict[str, str] = {
    "shot_class": "shot_confidence",
    "grade_class": "grade_confidence",
    "archetype": "archetype_confidence",
}


def parse_lenient(raw: dict) -> tuple[VLMAnalysis, list[str]]:
    """Build an analysis from a partial model response (graceful degradation).

    Small local models routinely emit a subset of the schema. Rather than losing
    the reel entirely we keep what was answered, fill the rest with neutral
    defaults, zero the confidence of every defaulted dimension, and return the
    list of missing fields so downstream can weight the row honestly.

    Raises if the response is so degenerate that nothing usable was supplied.
    """
    payload = dict(raw)
    missing: list[str] = []
    for field, default in _PARTIAL_DEFAULTS.items():
        if payload.get(field) is None:
            payload[field] = default
            missing.append(field)
    for dimension, conf_field in _CONFIDENCE_OF.items():
        if dimension in missing:
            payload[conf_field] = 0.0
    supplied = len(_PARTIAL_DEFAULTS) - len(missing)
    if supplied == 0:
        raise ValueError("model supplied none of the analysis fields")
    # Drop keys the model invented so extra="forbid" still holds.
    payload = {k: v for k, v in payload.items() if k in _PARTIAL_DEFAULTS}
    return VLMAnalysis.model_validate(payload), missing


def completeness(missing: list[str]) -> float:
    """Share of analysis fields the model actually supplied (1.0 = complete)."""
    return round(1.0 - len(missing) / len(_PARTIAL_DEFAULTS), 3)


class AnalysisStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"
