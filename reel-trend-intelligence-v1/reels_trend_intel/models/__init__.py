from __future__ import annotations

from reels_trend_intel.models.calibration import Calibrator, backtest
from reels_trend_intel.models.emerging import EarlyClassifier, EmergingSignals, compute_emerging
from reels_trend_intel.models.fusion import FusionModel, FusionResult, component_votes
from reels_trend_intel.models.hawkes import HawkesResult, fit_hawkes
from reels_trend_intel.models.health import health_score
from reels_trend_intel.models.spikem import SpikeMResult, fit_spikem
from reels_trend_intel.models.survival import CoxPH, detect_death, fit_cox, kaplan_meier
from reels_trend_intel.models.whitespace import OpportunityResult, opportunity_score

__all__ = [
    "fit_hawkes", "HawkesResult",
    "fit_spikem", "SpikeMResult",
    "fit_cox", "CoxPH", "detect_death", "kaplan_meier",
    "compute_emerging", "EmergingSignals", "EarlyClassifier",
    "health_score", "opportunity_score", "OpportunityResult",
    "FusionModel", "FusionResult", "component_votes",
    "Calibrator", "backtest",
]
