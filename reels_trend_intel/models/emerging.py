"""Early / emerging-trend detection.

Bundles several complementary early-warning signals over the adoption curve N(t):
  * velocity + acceleration (seasonally normalised)
  * Kleinberg 2-state burst detection (onset bucket)
  * CUSUM change alarm
  * Bayesian Online Change-Point Detection (BOCPD) most-recent change
  * Bass diffusion fit (p,q,M) -> peak size + time-to-peak from the early curve
  * an early gradient-boosted classifier: P(crosses virality threshold) from the
    first-T-hours features (early acceleration, adopter breadth, cross-niche spread)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit

from reels_trend_intel.config.settings import EmergingConfig
from reels_trend_intel.trends.types import AdoptionCurve


def deseasonalize(counts: np.ndarray, period: int = 24) -> np.ndarray:
    if len(counts) < 2 * period:
        return counts.astype(np.float64)
    idx = np.arange(len(counts)) % period
    seasonal = np.array([counts[idx == p].mean() for p in range(period)])
    seasonal = seasonal / (seasonal.mean() + 1e-9)
    return counts / (seasonal[idx] + 1e-9)


def velocity_acceleration(counts: np.ndarray, w_h: float) -> tuple[float, float]:
    cum = np.cumsum(counts)
    if len(cum) < 3:
        return 0.0, 0.0
    v = np.gradient(cum) / w_h
    a = np.gradient(v) / w_h
    return float(v[-1]), float(a[-1])


def kleinberg_burst(counts: np.ndarray, gamma: float = 1.0) -> int | None:
    """Two-state burst automaton; returns the onset bucket index, else None."""
    x = counts.astype(np.float64)
    n = len(x)
    if n < 4 or x.sum() < 4:
        return None
    base = max(x.mean(), 1e-3)
    lam = [base, base * 2.0]  # base vs burst rate
    log_n = np.log(max(n, 2))

    def cost(state: int, k: float) -> float:
        rate = lam[state]
        return rate - k * np.log(rate + 1e-9)  # neg Poisson log-lik (k counts)

    dp = [[0.0, 0.0] for _ in range(n)]
    back = [[0, 0] for _ in range(n)]
    dp[0] = [cost(0, x[0]), cost(1, x[0]) + gamma * log_n]
    for i in range(1, n):
        for s in (0, 1):
            best, arg = 1e18, 0
            for sp in (0, 1):
                trans = gamma * log_n if (s == 1 and sp == 0) else 0.0
                val = dp[i - 1][sp] + trans + cost(s, x[i])
                if val < best:
                    best, arg = val, sp
            dp[i][s], back[i][s] = best, arg
    state = int(np.argmin(dp[-1]))
    seq = [0] * n
    for i in range(n - 1, -1, -1):
        seq[i] = state
        state = back[i][state]
    for i, s in enumerate(seq):
        if s == 1:
            return i
    return None


def cusum(counts: np.ndarray, threshold: float, drift: float) -> int | None:
    x = counts.astype(np.float64)
    mu = x.mean()
    sd = x.std() + 1e-9
    s = 0.0
    for i, v in enumerate(x):
        s = max(0.0, s + (v - mu) / sd - drift)
        if s > threshold:
            return i
    return None


def bocpd(counts: np.ndarray, hazard: float) -> int | None:
    """Compact Gaussian BOCPD; returns index of the most recent change-point."""
    x = counts.astype(np.float64)
    n = len(x)
    if n < 4:
        return None
    mu0, kappa0, alpha0, beta0 = x.mean(), 1.0, 1.0, max(x.var(), 1.0)
    R = np.zeros(n + 1)
    R[0] = 1.0
    mu = np.array([mu0])
    kappa = np.array([kappa0])
    alpha = np.array([alpha0])
    beta = np.array([beta0])
    changes: list[int] = []
    for t in range(n):
        xt = x[t]
        scale = beta * (kappa + 1) / (alpha * kappa)
        pred = _student_t_pdf(xt, 2 * alpha, mu, np.sqrt(np.maximum(scale, 1e-9)))
        growth = R[: t + 1] * pred * (1 - hazard)
        cp = float((R[: t + 1] * pred * hazard).sum())
        newR = np.empty(t + 2)
        newR[1:] = growth
        newR[0] = cp
        newR /= newR.sum() + 1e-12
        R[: t + 2] = newR
        # posterior updates (prepend prior)
        kappa_n = kappa + 1
        mu_n = (kappa * mu + xt) / kappa_n
        alpha_n = alpha + 0.5
        beta_n = beta + (kappa * (xt - mu) ** 2) / (2 * kappa_n)
        mu = np.concatenate([[mu0], mu_n])
        kappa = np.concatenate([[kappa0], kappa_n])
        alpha = np.concatenate([[alpha0], alpha_n])
        beta = np.concatenate([[beta0], beta_n])
        if t > 0 and int(np.argmax(R[: t + 2])) < 2:
            changes.append(t)
    return changes[-1] if changes else None


def _student_t_pdf(x: float, nu: np.ndarray, loc: np.ndarray, scale: np.ndarray) -> np.ndarray:
    from scipy.special import gammaln

    z = (x - loc) / scale
    log_c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi) - np.log(scale)
    return np.exp(log_c - (nu + 1) / 2 * np.log(1 + z**2 / nu))


def _bass_cum(t: np.ndarray, p: float, q: float, m: float) -> np.ndarray:
    e = np.exp(-(p + q) * t)
    return m * (1 - e) / (1 + (q / max(p, 1e-6)) * e)


def bass_fit(counts: np.ndarray, w_h: float) -> tuple[float, float, float, float] | None:
    y = np.cumsum(counts.astype(np.float64))
    t = (np.arange(len(y)) + 1) * w_h
    if len(y) < 4 or y[-1] < 4:
        return None
    try:
        popt, _ = curve_fit(
            _bass_cum, t, y, p0=[0.01, 0.2, max(y[-1] * 1.5, 5)],
            bounds=([1e-5, 1e-4, y[-1]], [0.8, 2.0, y[-1] * 50]), maxfev=8000,
        )
        p, q, m = (float(popt[0]), float(popt[1]), float(popt[2]))
        peak_h = float(np.log(max(q / p, 1e-6)) / (p + q)) if q > p else 0.0
        return p, q, m, peak_h
    except Exception:
        return None


@dataclass
class EmergingSignals:
    velocity: float
    acceleration: float
    burst_onset_bucket: int | None
    cusum_alarm_bucket: int | None
    changepoint_bucket: int | None
    bass_p: float | None = None
    bass_q: float | None = None
    bass_m: float | None = None
    bass_peak_time_h: float | None = None
    p_viral: float = 0.0
    early_features: list[float] = field(default_factory=list)


def compute_emerging(curve: AdoptionCurve, cfg: EmergingConfig) -> EmergingSignals:
    counts = np.asarray(curve.counts, dtype=np.float64)
    w_h = curve.bucket_s / 3600.0
    des = deseasonalize(counts)
    v, a = velocity_acceleration(des, w_h)
    burst = kleinberg_burst(counts, cfg.kleinberg_gamma)
    cu = cusum(counts, cfg.cusum_threshold, cfg.cusum_drift)
    cp = bocpd(counts, cfg.bocpd_hazard)
    bass = bass_fit(counts, w_h)
    sig = EmergingSignals(
        velocity=v, acceleration=a, burst_onset_bucket=burst, cusum_alarm_bucket=cu,
        changepoint_bucket=cp,
    )
    if bass:
        sig.bass_p, sig.bass_q, sig.bass_m, sig.bass_peak_time_h = bass
    return sig


def early_window_features(
    curve: AdoptionCurve, early_window_h: float, niche_share: float, n_authors: int
) -> list[float]:
    counts = np.asarray(curve.counts, dtype=np.float64)
    w_h = curve.bucket_s / 3600.0
    k = max(2, int(early_window_h / w_h))
    early = counts[:k]
    v, a = velocity_acceleration(early if len(early) >= 3 else counts, w_h)
    breadth = float(n_authors) / max(1.0, float(curve.total))   # adopter diversity
    return [float(early.sum()), v, a, breadth, niche_share]


class EarlyClassifier:
    """P(crosses virality threshold) from first-T-hours features."""

    def __init__(self) -> None:
        self.model: object | None = None
        self._mean = 0.0
        self._std = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(np.unique(y)) < 2 or len(y) < 4:
            # Fallback: calibrate a logistic on the early-acceleration column.
            self.model = None
            col = X[:, 2] if X.shape[1] > 2 else X[:, 0]
            self._mean, self._std = float(col.mean()), float(col.std() + 1e-9)
            return
        try:
            from sklearn.ensemble import GradientBoostingClassifier

            gb = GradientBoostingClassifier(n_estimators=60, max_depth=2, random_state=1729)
            gb.fit(X, y)
            self.model = gb
        except Exception:
            self.model = None

    def predict_proba(self, x: np.ndarray) -> float:
        if self.model is not None:
            return float(self.model.predict_proba(x.reshape(1, -1))[0, 1])  # type: ignore
        z = (float(x[2] if len(x) > 2 else x[0]) - self._mean) / self._std
        return float(1.0 / (1.0 + np.exp(-z)))
