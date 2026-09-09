"""SpikeM-style rise / peak / decay fit (Matsubara et al., KDD 2012).

SpikeM models the rise-and-fall of online activity (a SIR-with-power-law shape).
We fit a parametric rise-peak-decay curve to the adoption series N(t) and read
off the current PHASE (pre-peak / peak / decay) and TIME-TO-PEAK. We use a Gamma
shape, which captures a power-law-ish rise with an exponential decay and a single
interior peak at t* = (alpha - 1) * tau:

    f(t) = A * (t/tau)^(alpha-1) * exp(-(t/tau)) ,  t >= 0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


def _gamma_shape(t: np.ndarray, A: float, alpha: float, tau: float) -> np.ndarray:
    t = np.maximum(t, 1e-6)
    tau = max(tau, 1e-3)
    return A * np.power(t / tau, alpha - 1.0) * np.exp(-(t / tau))


@dataclass
class SpikeMResult:
    phase: str                  # pre-peak | peak | decay
    peak_time_h: float
    time_to_peak_h: float
    alpha: float
    tau: float
    amplitude: float
    fitted: bool


def fit_spikem(counts: list[int], bucket_s: int, now_h: float | None = None) -> SpikeMResult:
    y = np.asarray(counts, dtype=np.float64)
    w = bucket_s / 3600.0
    t = (np.arange(len(y)) + 0.5) * w
    now = float(now_h if now_h is not None else (t[-1] if len(t) else 0.0))

    if len(y) < 4 or y.sum() < 4:
        # Too little signal: call it by the empirical argmax.
        peak_h = float(t[int(np.argmax(y))]) if len(y) else 0.0
        phase = "pre-peak" if now < peak_h else "decay"
        return SpikeMResult(phase, peak_h, peak_h - now, 2.0, max(peak_h, 1.0), float(y.max()),
                            False)

    A0 = max(float(y.max()), 1.0)
    p0 = [A0, 2.0, max(float(t[int(np.argmax(y))]) / 1.0, 1.0)]
    try:
        popt, _ = curve_fit(
            _gamma_shape, t, y, p0=p0,
            bounds=([1e-3, 1.01, 1e-2], [A0 * 50, 50.0, max(t[-1] * 5, 10.0)]),
            maxfev=8000,
        )
        A, alpha, tau = (float(popt[0]), float(popt[1]), float(popt[2]))
        peak_h = (alpha - 1.0) * tau
        fitted = True
    except Exception:
        A, alpha, tau = A0, 2.0, max(float(t[int(np.argmax(y))]), 1.0)
        peak_h = float(t[int(np.argmax(y))])
        fitted = False

    width = max(tau, 1.0)
    if now < peak_h - 0.5 * width:
        phase = "pre-peak"
    elif now <= peak_h + 0.5 * width:
        phase = "peak"
    else:
        phase = "decay"
    return SpikeMResult(phase, peak_h, peak_h - now, alpha, tau, A, fitted)
