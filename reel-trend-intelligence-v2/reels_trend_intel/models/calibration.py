"""Probability calibration + backtest metrics.

Calibrated honesty is the whole point: a raw model score becomes a probability
only after calibration against realized outcomes (isotonic or Platt/sigmoid). We
also report the standard reliability diagnostics (Brier score, ECE) so the
calibration quality is auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Calibrator:
    method: str = "isotonic"
    _iso_x: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
    _iso_y: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
    _platt_a: float = 1.0
    _platt_b: float = 0.0
    fitted: bool = False
    n_samples: int = 0

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> Calibrator:
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)
        self.n_samples = len(labels)
        if self.method == "none" or len(np.unique(labels)) < 2 or len(labels) < 4:
            self.fitted = False
            return self
        if self.method == "isotonic":
            try:
                from sklearn.isotonic import IsotonicRegression

                ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                ir.fit(scores, labels)
                grid = np.linspace(scores.min(), scores.max(), 50)
                self._iso_x = grid
                self._iso_y = np.clip(ir.predict(grid), 0.0, 1.0)
                self.fitted = True
            except Exception:
                self.fitted = False
        else:  # platt / sigmoid
            from scipy.optimize import minimize

            def nll(ab: np.ndarray) -> float:
                a, b = ab
                z = np.clip(a * scores + b, -30, 30)
                p = 1.0 / (1.0 + np.exp(-z))
                return float(-(labels * np.log(p + 1e-9)
                              + (1 - labels) * np.log(1 - p + 1e-9)).mean())

            res = minimize(nll, np.array([1.0, 0.0]), method="Nelder-Mead")
            self._platt_a, self._platt_b = float(res.x[0]), float(res.x[1])
            self.fitted = True
        return self

    def transform(self, score: float) -> float:
        if not self.fitted:
            return float(np.clip(score, 0.0, 1.0))
        if self.method == "isotonic":
            return float(np.interp(score, self._iso_x, self._iso_y))
        z = np.clip(self._platt_a * score + self._platt_b, -30, 30)
        return float(1.0 / (1.0 + np.exp(-z)))


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(bins):
        m = (probs >= edges[i]) & (probs < edges[i + 1] if i < bins - 1 else probs <= 1.0)
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(probs[m].mean() - labels[m].mean())
    return float(ece)


@dataclass
class BacktestReport:
    n: int
    brier: float
    ece: float
    base_rate: float
    reliability: list[tuple[float, float, int]]  # (mean_pred, empirical, count)


def backtest(probs: np.ndarray, labels: np.ndarray, bins: int = 5) -> BacktestReport:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0, 1, bins + 1)
    rel: list[tuple[float, float, int]] = []
    for i in range(bins):
        m = (probs >= edges[i]) & (probs <= edges[i + 1] if i == bins - 1 else probs < edges[i + 1])
        if m.sum() > 0:
            rel.append((float(probs[m].mean()), float(labels[m].mean()), int(m.sum())))
    return BacktestReport(
        n=len(labels), brier=brier_score(probs, labels),
        ece=expected_calibration_error(probs, labels),
        base_rate=float(labels.mean()) if len(labels) else 0.0, reliability=rel,
    )
