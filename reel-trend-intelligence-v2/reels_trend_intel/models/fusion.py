"""Fuse the three trend models + calibrate into a persistence probability + CI.

Inputs per trend are the model signals (Hawkes R(t), survival conditional S, SpikeM
phase, emerging velocity/acceleration, early P(viral), size). A logistic meta-model
maps them to a raw score; a Calibrator (isotonic/Platt) turns that into a probability
fitted against realized persistence. The confidence interval combines:
  * sampling uncertainty (Wald, widening as the calibration set shrinks), and
  * model disagreement (spread of the three components' individual votes),
so the interval is honestly wide when the data is thin or the models conflict.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reels_trend_intel.models.calibration import Calibrator

FEATURE_NAMES = [
    "r_t", "survival_cond", "phase_score", "velocity_norm", "acceleration",
    "p_viral", "log_size",
]


@dataclass
class FusionResult:
    persistence_probability: float
    ci_low: float
    ci_high: float
    raw_score: float
    components: dict[str, float]


def phase_score(phase: str) -> float:
    return {"pre-peak": 0.75, "peak": 0.6, "decay": 0.2}.get(phase, 0.5)


def component_votes(r_t: float, survival_cond: float, phase: str) -> dict[str, float]:
    return {
        "hawkes": float(_sigmoid(2.5 * (r_t - 1.0))),
        "survival": float(np.clip(survival_cond, 0.0, 1.0)),
        "spikem": phase_score(phase),
    }


def raw_persistence_score(
    r_t: float, survival_vote: float, phase: str, velocity: float
) -> float:
    """A MONOTONE aliveness score in [0,1] (higher = more likely to persist).

    Fixed positive weights guarantee sensible ordering (a rising R>1 pre-peak trend
    always scores above a dead R≈0 decaying one); isotonic/Platt calibration then
    turns this ordered score into an honest probability.
    """
    hv = float(_sigmoid(2.5 * (r_t - 1.0)))
    pv = phase_score(phase)
    vv = float(_sigmoid(np.tanh(velocity)))
    return float(0.35 * hv + 0.30 * np.clip(survival_vote, 0.0, 1.0) + 0.20 * pv + 0.15 * vv)


class FusionModel:
    def __init__(self, method: str = "isotonic") -> None:
        self._lr: object | None = None
        self._mean = np.zeros(len(FEATURE_NAMES))
        self._std = np.ones(len(FEATURE_NAMES))
        self.calibrator = Calibrator(method=method)
        self.fitted = False

    def _raw(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self._mean) / self._std
        if self._lr is not None:
            return np.asarray(self._lr.predict_proba(Z)[:, 1])  # type: ignore[attr-defined]
        # fallback: mean of survival_cond + hawkes-ish column
        return np.clip(0.5 * X[:, 1] + 0.5 * _sigmoid(2.5 * (X[:, 0] - 1.0)), 0, 1)

    def fit(self, X: np.ndarray, labels: np.ndarray) -> FusionModel:
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._std[self._std < 1e-6] = 1.0
        Z = (X - self._mean) / self._std
        if len(np.unique(labels)) >= 2 and len(labels) >= 4:
            try:
                from sklearn.linear_model import LogisticRegression

                lr = LogisticRegression(C=1.0, max_iter=500)
                lr.fit(Z, labels)
                self._lr = lr
            except Exception:
                self._lr = None
        raw = self._raw(X)
        self.calibrator.fit(raw, labels)
        self.fitted = True
        return self

    def fit_scores(self, raw_scores: np.ndarray, labels: np.ndarray) -> FusionModel:
        """Calibrate directly on raw persistence scores (bucket-level backtest set)."""
        self.calibrator.fit(np.asarray(raw_scores), np.asarray(labels))
        self.fitted = True
        return self

    def predict_raw(self, raw: float, votes: dict[str, float]) -> FusionResult:
        p = self.calibrator.transform(raw)
        n_eff = max(self.calibrator.n_samples, 1)
        wald = 1.96 * float(np.sqrt(max(p * (1 - p), 1e-4) / n_eff))
        disagreement = float(np.std(list(votes.values())))
        half = float(np.clip(max(wald, 0.6 * disagreement), 0.03, 0.45))
        return FusionResult(
            persistence_probability=float(np.clip(p, 0.0, 1.0)),
            ci_low=float(np.clip(p - half, 0.0, 1.0)),
            ci_high=float(np.clip(p + half, 0.0, 1.0)),
            raw_score=raw, components=votes,
        )

    def predict(self, x: np.ndarray, votes: dict[str, float]) -> FusionResult:
        raw = float(self._raw(x.reshape(1, -1))[0])
        p = self.calibrator.transform(raw)
        n_eff = max(self.calibrator.n_samples, 1)
        wald = 1.96 * float(np.sqrt(max(p * (1 - p), 1e-4) / n_eff))
        disagreement = float(np.std(list(votes.values())))
        half = float(np.clip(max(wald, 0.6 * disagreement), 0.03, 0.45))
        return FusionResult(
            persistence_probability=float(np.clip(p, 0.0, 1.0)),
            ci_low=float(np.clip(p - half, 0.0, 1.0)),
            ci_high=float(np.clip(p + half, 0.0, 1.0)),
            raw_score=raw, components=votes,
        )


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))
