"""Trend Health Score — a 0..1 summary of current momentum + phase."""

from __future__ import annotations

import numpy as np


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def health_score(
    r_t: float, velocity_norm: float, phase: str, size: int, supercritical: bool
) -> float:
    """Weighted blend of reproduction, velocity, phase and (log) size."""
    r_term = _sigmoid(2.5 * (r_t - 1.0))               # R>1 => healthy
    v_term = _sigmoid(velocity_norm)                   # rising adoption
    phase_term = {"pre-peak": 0.9, "peak": 1.0, "decay": 0.3}.get(phase, 0.5)
    size_term = float(np.clip(np.log1p(size) / np.log1p(50), 0.0, 1.0))
    score = 0.40 * r_term + 0.25 * v_term + 0.25 * phase_term + 0.10 * size_term
    if supercritical:
        score = min(1.0, score + 0.05)
    return float(np.clip(score, 0.0, 1.0))
