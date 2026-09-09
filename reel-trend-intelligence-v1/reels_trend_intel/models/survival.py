"""Survival analysis: Kaplan-Meier + Cox proportional hazards.

DEATH of a trend = adoption rate drops > death_drop_pct from its peak for
death_consecutive_buckets buckets. We fit:
  * Kaplan-Meier for the marginal S(t) across trends, and
  * a ridge-regularised Cox PH  h(t|x) = h0(t)·exp(beta·x)  over trend covariates.
"Persists >= N more days" is the conditional survival  S(t0+N|x) / S(t0|x).

Pure NumPy/SciPy so it runs on the lean box; with only a few resolved trends the
estimates are wide — they sharpen as trend history accumulates across runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def detect_death(counts: list[int], bucket_s: int, drop_pct: float, k: int) -> tuple[float, int]:
    """Return (duration_days, event) where event=1 if the trend died, else censored."""
    y = np.asarray(counts, dtype=np.float64)
    bucket_days = bucket_s / 86400.0
    total_days = max(len(y) * bucket_days, bucket_days)
    if len(y) < k + 2:
        return total_days, 0
    peak_idx = int(np.argmax(y))
    peak = float(y[peak_idx])
    if peak <= 0:
        return total_days, 0
    threshold = (1.0 - drop_pct) * peak
    run = 0
    for i in range(peak_idx + 1, len(y)):
        run = run + 1 if y[i] < threshold else 0
        if run >= k:
            return (i - k + 1) * bucket_days, 1
    return total_days, 0


def kaplan_meier(durations: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(durations)
    d = durations[order]
    e = events[order]
    n = len(d)
    times: list[float] = []
    surv: list[float] = []
    s = 1.0
    for i in range(n):
        if e[i] == 1:
            at_risk = n - i
            s *= 1.0 - 1.0 / at_risk
            times.append(float(d[i]))
            surv.append(s)
    if not times:
        return np.array([0.0]), np.array([1.0])
    return np.asarray(times), np.asarray(surv)


def km_at(times: np.ndarray, surv: np.ndarray, t: float) -> float:
    idx = int(np.searchsorted(times, t, side="right")) - 1
    if idx < 0:
        return 1.0
    return float(surv[min(idx, len(surv) - 1)])


@dataclass
class CoxPH:
    beta: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    base_times: np.ndarray
    base_cumhaz: np.ndarray
    fitted: bool

    def _z(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def cumhaz_at(self, t: float) -> float:
        idx = int(np.searchsorted(self.base_times, t, side="right")) - 1
        if idx < 0:
            return 0.0
        return float(self.base_cumhaz[min(idx, len(self.base_cumhaz) - 1)])

    def survival_at(self, t: float, x: np.ndarray) -> float:
        risk = float(np.exp(np.clip(self.beta @ self._z(x), -10, 10)))
        return float(np.exp(-self.cumhaz_at(t) * risk))

    def conditional_persistence(self, t0: float, n_days: float, x: np.ndarray) -> float:
        s0 = self.survival_at(t0, x)
        s1 = self.survival_at(t0 + n_days, x)
        if s0 <= 1e-9:
            return 0.0
        return float(np.clip(s1 / s0, 0.0, 1.0))


def fit_cox(
    X: np.ndarray, durations: np.ndarray, events: np.ndarray, ridge: float = 1.0,
    iters: int = 25,
) -> CoxPH:
    n, p = X.shape
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    Z = (X - mean) / std
    beta = np.zeros(p)

    n_events = int(events.sum())
    if n_events >= 2 and n >= p + 1:
        order = np.argsort(-durations)  # descending so risk set is a prefix
        Zs, ds, es = Z[order], durations[order], events[order]
        for _ in range(iters):
            eta = np.clip(Zs @ beta, -10, 10)
            exp_eta = np.exp(eta)
            grad = np.zeros(p)
            hess = np.zeros((p, p))
            for i in range(n):
                if es[i] != 1:
                    continue
                risk = exp_eta[: i + 1]  # descending durations -> risk set is prefix
                Zr = Zs[: i + 1]
                denom = risk.sum()
                if denom <= 1e-9:
                    continue
                wbar = (risk[:, None] * Zr).sum(axis=0) / denom
                grad += Zs[i] - wbar
                cov = (risk[:, None, None] * (Zr[:, :, None] * Zr[:, None, :])).sum(axis=0) / denom
                cov -= np.outer(wbar, wbar)
                hess -= cov
            grad -= ridge * beta
            hess -= ridge * np.eye(p)
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                break
            beta_new = beta - step
            if not np.all(np.isfinite(beta_new)):
                break
            if np.linalg.norm(beta_new - beta) < 1e-6:
                beta = beta_new
                break
            beta = beta_new
        fitted = True
    else:
        fitted = False

    # Breslow baseline cumulative hazard.
    order = np.argsort(durations)
    ds, es = durations[order], events[order]
    risk_score = np.exp(np.clip(Z[order] @ beta, -10, 10))
    times: list[float] = []
    cumhaz: list[float] = []
    H = 0.0
    for i in range(n):
        if es[i] == 1:
            denom = risk_score[i:].sum()
            if denom > 1e-9:
                H += 1.0 / denom
            times.append(float(ds[i]))
            cumhaz.append(H)
    if not times:
        times, cumhaz = [0.0], [0.0]
    return CoxPH(beta=beta, mean=mean, std=std, base_times=np.asarray(times),
                 base_cumhaz=np.asarray(cumhaz), fitted=fitted)
