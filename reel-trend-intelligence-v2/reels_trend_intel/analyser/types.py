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


class AnalysisStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"
