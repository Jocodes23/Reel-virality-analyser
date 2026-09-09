"""reels-trend-intel: Instagram Reels trend-intelligence system.

Version stamps below are written onto every persisted record (features, trend
models, reports) for reproducibility and downstream schema stability.
"""

from __future__ import annotations

__version__ = "0.1.0"

# --- Versioning stamped onto records (see report.schema.Provenance) -----------
# Bump FEATURE_SCHEMA_VERSION when the feature extraction layout changes.
# Bump MODEL_VERSIONS entries when a model's math / hyperparameters change.
# Bump REPORT_VERSION when the TrendReport wire schema changes.
FEATURE_SCHEMA_VERSION = "1.0.0"
REPORT_VERSION = "1.0.0"

MODEL_VERSIONS: dict[str, str] = {
    "hawkes": "1.0.0",
    "survival": "1.0.0",
    "spikem": "1.0.0",
    "fusion": "1.0.0",
    "calibration": "1.0.0",
    "emerging": "1.0.0",
    "embeddings.deterministic": "1.0.0",
    "embeddings.clip": "clip-ViT-B-32",
    "embeddings.caption": "all-MiniLM-L6-v2",
}

__all__ = [
    "__version__",
    "FEATURE_SCHEMA_VERSION",
    "REPORT_VERSION",
    "MODEL_VERSIONS",
]
