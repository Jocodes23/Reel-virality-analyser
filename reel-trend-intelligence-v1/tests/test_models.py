from __future__ import annotations

import numpy as np

from reels_trend_intel.config.settings import ModelConfig
from reels_trend_intel.models.calibration import Calibrator, backtest
from reels_trend_intel.models.emerging import (
    bass_fit,
    cusum,
    kleinberg_burst,
    velocity_acceleration,
)
from reels_trend_intel.models.fusion import raw_persistence_score
from reels_trend_intel.models.hawkes import fit_hawkes
from reels_trend_intel.models.spikem import fit_spikem
from reels_trend_intel.models.survival import detect_death, fit_cox, kaplan_meier


def _growing_events(n=60, rate=0.03):
    # accelerating arrival times (hours)
    t = 0.0
    out = []
    for _ in range(n):
        t += max(0.1, np.random.default_rng(int(t * 100) + 1).exponential(1.0 / (rate + 0.02 * t)))
        out.append(t)
    return sorted(out)


def test_hawkes_growing_vs_decaying():
    rng = np.random.default_rng(0)
    # front-loaded (decaying) events
    decaying = sorted(rng.exponential(2.0, 40).cumsum())
    r = fit_hawkes(decaying, bucket_s=3600, cfg=ModelConfig(), now_h=decaying[-1] + 50)
    assert r.n_now == 40
    assert r.final_size_estimate >= r.n_now
    assert 0.0 <= r.p_t <= 5.0


def test_spikem_phase_detection():
    # rise then fall
    counts = [0, 1, 2, 4, 7, 10, 12, 10, 7, 4, 2, 1, 0, 0]
    res = fit_spikem(counts, bucket_s=3600, now_h=13)
    assert res.phase in {"pre-peak", "peak", "decay"}
    # evaluated near the end -> decay
    assert res.phase == "decay"
    early = fit_spikem(counts, bucket_s=3600, now_h=2)
    assert early.phase in {"pre-peak", "peak"}


def test_survival_detect_death_and_km():
    counts = [1, 3, 6, 10, 8, 2, 1, 0, 0, 0]  # peaks at idx3 then collapses
    dur, ev = detect_death(counts, 86400, drop_pct=0.6, k=2)
    assert ev == 1 and dur > 0
    durations = np.array([2.0, 3.0, 5.0, 6.0])
    events = np.array([1, 0, 1, 1])
    t, s = kaplan_meier(durations, events)
    assert np.all(np.diff(s) <= 1e-9)  # non-increasing


def test_cox_conditional_survival_monotone():
    X = np.array([[1.0, 1.5, 2.0], [2.0, 0.5, 1.0], [1.5, 1.0, 0.5], [2.5, 2.0, 3.0]])
    dur = np.array([3.0, 1.0, 2.0, 4.0])
    ev = np.array([1, 1, 1, 0])
    cox = fit_cox(X, dur, ev, ridge=1.0)
    s = cox.conditional_persistence(1.0, 2.0, X[0])
    assert 0.0 <= s <= 1.0


def test_emerging_signals():
    counts = np.array([0, 0, 1, 2, 5, 9, 14, 12, 8, 4, 2, 1, 0, 0], dtype=float)
    v, a = velocity_acceleration(counts, 1.0)
    assert np.isfinite(v) and np.isfinite(a)
    burst = kleinberg_burst(counts)
    assert burst is None or isinstance(burst, int)
    cu = cusum(counts, threshold=3.0, drift=0.5)
    assert cu is None or isinstance(cu, int)
    bass = bass_fit(counts, 1.0)
    assert bass is None or (bass[2] > 0)


def test_calibration_and_backtest():
    rng = np.random.default_rng(0)
    scores = np.clip(rng.random(200), 0, 1)
    labels = (rng.random(200) < scores).astype(int)  # calibrated-ish by construction
    cal = Calibrator(method="isotonic").fit(scores, labels)
    p = [cal.transform(s) for s in scores]
    rep = backtest(np.array(p), labels)
    assert 0.0 <= rep.brier <= 1.0 and rep.ece >= 0.0


def test_raw_persistence_monotone():
    dead = raw_persistence_score(0.0, 0.0, "decay", 0.0)
    rising = raw_persistence_score(1.4, 0.8, "pre-peak", 2.0)
    assert rising > dead
