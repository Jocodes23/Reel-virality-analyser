"""Per-reel video analyser (V2).

Public surface: the analysis schema and the swappable VLM layer. Nothing here
imports a model provider directly — see `analyser.vlm` for the toggle.
"""

from __future__ import annotations

from reels_trend_intel.analyser.types import (
    ANALYSER_SCHEMA_VERSION,
    AnalysisStatus,
    Archetype,
    GradeClass,
    ShotAttributes,
    ShotClass,
    Structure,
    VLMAnalysis,
)
from reels_trend_intel.analyser.vlm import (
    PROVIDERS,
    VLMAdapter,
    VLMResult,
    analyse_with_retry,
    availability_report,
    make_vlm_adapter,
)

__all__ = [
    "ANALYSER_SCHEMA_VERSION", "AnalysisStatus", "VLMAnalysis", "ShotClass",
    "ShotAttributes", "GradeClass", "Archetype", "Structure",
    "VLMAdapter", "VLMResult", "analyse_with_retry", "make_vlm_adapter",
    "availability_report", "PROVIDERS",
]
