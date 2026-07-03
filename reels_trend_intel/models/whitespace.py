"""Whitespace detection: proven-but-unsaturated opportunities for a niche.

A trend that is accelerating GLOBALLY but UNDER-adopted in the configured niche
(e.g. food/restaurant) is the best opportunity: proven demand, low local
competition. opportunity = momentum * (1 - niche_saturation), gated by global
acceleration and a niche-saturation cap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reels_trend_intel.config.settings import EmergingConfig


@dataclass
class OpportunityResult:
    opportunity_score: float
    niche_saturation: float
    global_momentum: float
    is_whitespace: bool


def opportunity_score(
    health: float, acceleration: float, niche_share: float, cfg: EmergingConfig
) -> OpportunityResult:
    # momentum blends current health with positive acceleration
    momentum = float(np.clip(0.6 * health + 0.4 * _sigmoid(acceleration), 0.0, 1.0))
    score = momentum * (1.0 - niche_share)
    is_ws = (
        acceleration >= cfg.opportunity_min_global_accel
        and niche_share <= cfg.opportunity_niche_saturation_cap
        and momentum > 0.3
    )
    return OpportunityResult(
        opportunity_score=float(np.clip(score, 0.0, 1.0)),
        niche_saturation=float(niche_share),
        global_momentum=momentum,
        is_whitespace=is_ws,
    )


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))
