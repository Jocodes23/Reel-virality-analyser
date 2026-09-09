"""Self-exciting point process (Hawkes / SEISMIC) with a power-law memory kernel.

We model trend adoption as a self-exciting process: each adopting reel raises the
intensity of further adoption. Using a normalised power-law kernel (human
reaction-time tail, as in SEISMIC, Zhao et al. KDD 2015):

    phi(s) = theta * c^theta / (s + c)^(1+theta),   s > 0,   ∫phi = 1

We estimate a NONPARAMETRIC branching ratio (effective reproduction number)

    R(t) = observed_adoption_rate(t) / Σ_{t_j < t} phi(t - t_j)

R>1 => supercritical (growing); R<1 => subcritical (decaying). For a subcritical
process the expected final size follows the branching-process geometric sum
    N_inf ≈ N_now / (1 - R_eff).
Infectiousness p(t) = R(t) / n*. Honest about irreducible variance: these are
estimates with wide uncertainty in the heavy tail, not guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from reels_trend_intel.config.settings import ModelConfig


def power_law_kernel(s: np.ndarray, theta: float, c_h: float) -> np.ndarray:
    out = np.zeros_like(s, dtype=np.float64)
    pos = s > 0
    out[pos] = theta * (c_h**theta) / np.power(s[pos] + c_h, 1.0 + theta)
    return out


@dataclass
class HawkesResult:
    r_t: float
    p_t: float
    n_star: float
    r_series: list[float] = field(default_factory=list)
    final_size_estimate: float = 0.0
    supercritical: bool = False
    n_now: int = 0


def fit_hawkes(
    event_times_h: list[float], bucket_s: int, cfg: ModelConfig, now_h: float | None = None
) -> HawkesResult:
    n_now = len(event_times_h)
    n_star = cfg.hawkes_n_star
    if n_now < 3:
        return HawkesResult(r_t=0.0, p_t=0.0, n_star=n_star, n_now=n_now,
                            final_size_estimate=float(n_now))

    events = np.asarray(sorted(event_times_h), dtype=np.float64)
    c_h = max(cfg.hawkes_kernel_c_s / 3600.0, 1e-3)
    theta = cfg.hawkes_theta
    w = bucket_s / 3600.0
    horizon = float(now_h if now_h is not None else events[-1])

    centers = np.arange(events[0] + w, horizon + w, w)
    r_series: list[float] = []
    for t in centers:
        prior = events[events < t]
        if prior.size == 0:
            continue
        parent_rate = float(power_law_kernel(t - prior, theta, c_h).sum())
        observed = float(((events >= t - w) & (events < t)).sum()) / w
        if parent_rate <= 1e-9:
            continue
        r_series.append(observed / parent_rate)

    if not r_series:
        return HawkesResult(r_t=0.0, p_t=0.0, n_star=n_star, n_now=n_now,
                            final_size_estimate=float(n_now))

    # Recent R = weighted average of the last few buckets (more weight = more recent).
    tail = np.asarray(r_series[-5:])
    weights = np.linspace(0.5, 1.0, len(tail))
    r_t = float(np.average(tail, weights=weights))
    p_t = r_t / n_star if n_star > 0 else 0.0

    supercritical = r_t >= 1.0
    if supercritical:
        # Project bounded growth over the persistence horizon (not "unbounded").
        final = float(n_now) * (1.0 + min(r_t - 1.0, 1.0))
    else:
        r_eff = min(max(r_t, 0.0), 0.98)
        final = float(n_now) / (1.0 - r_eff)
    return HawkesResult(
        r_t=r_t, p_t=p_t, n_star=n_star, r_series=r_series,
        final_size_estimate=final, supercritical=supercritical, n_now=n_now,
    )
