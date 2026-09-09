"""Run all trend models + fuse + calibrate.

Per trend: Hawkes R(t), SpikeM phase, emerging signals, Cox conditional survival,
early P(viral), Trend Health Score, opportunity. Persistence probability is the
fused logistic over the model signals, CALIBRATED against a bucket-level TEMPORAL
BACKTEST: for every trend we slide a cutoff across its history, compute the as-of
signal vector, and label whether adoption actually persisted afterwards. That
gives a real (signals -> probability) mapping with auditable Brier/ECE — and, by
fitting a logistic, persistence varies meaningfully across trends (a decayed
R≈0 trend scores far below a rising R>1 one) rather than collapsing to the base
rate.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from reels_trend_intel.config.settings import Settings
from reels_trend_intel.models.calibration import backtest
from reels_trend_intel.models.emerging import (
    EarlyClassifier,
    EmergingSignals,
    compute_emerging,
    early_window_features,
    velocity_acceleration,
)
from reels_trend_intel.models.fusion import (
    FusionModel,
    component_votes,
    raw_persistence_score,
)
from reels_trend_intel.models.hawkes import HawkesResult, fit_hawkes
from reels_trend_intel.models.health import health_score
from reels_trend_intel.models.spikem import fit_spikem
from reels_trend_intel.models.survival import CoxPH, detect_death, fit_cox
from reels_trend_intel.models.whitespace import opportunity_score
from reels_trend_intel.observability.logging import get_logger
from reels_trend_intel.report.builder import TrendModeling
from reels_trend_intel.storage.base import StorageBackend
from reels_trend_intel.trends.types import BuiltTrend

log = get_logger("orchestration.modeling")


def _survival_proxy(counts: np.ndarray) -> float:
    if len(counts) == 0:
        return 0.0
    peak = float(counts.max())
    if peak <= 0:
        return 0.0
    recent = float(counts[-min(3, len(counts)):].mean())
    return float(np.clip(recent / peak, 0.0, 1.0))


def build_calibration_set(
    trends: list[BuiltTrend], settings: Settings
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a cutoff across each trend; emit (monotone raw score, persisted label)."""
    w = settings.trends.bucket_s / 3600.0
    horizon_buckets = max(1, int(settings.models.persistence_horizon_days * 86400
                                 / settings.trends.bucket_s))
    obs_min = max(3, int(settings.emerging.early_window_h / w))
    raws: list[float] = []
    y: list[int] = []
    for bt in trends:
        counts = np.asarray(bt.curve.counts, dtype=np.float64)
        L = len(counts)
        if L < obs_min + 2:
            continue
        step = max(1, L // 12)
        for cutoff in range(obs_min, L - 1, step):
            counts_c = counts[:cutoff]
            events_c = [t for t in bt.event_times_h if t <= cutoff * w]
            h = fit_hawkes(events_c, settings.trends.bucket_s, settings.models, now_h=cutoff * w)
            sp = fit_spikem(list(counts_c.astype(int)), settings.trends.bucket_s, now_h=cutoff * w)
            v, _ = velocity_acceleration(counts_c, w)
            raws.append(raw_persistence_score(h.r_t, _survival_proxy(counts_c), sp.phase, v))
            future = counts[cutoff:cutoff + horizon_buckets].sum()
            base = max(counts[max(0, cutoff - horizon_buckets):cutoff].sum(), 1.0)
            y.append(1 if future >= 0.5 * base else 0)
    return (np.asarray(raws), np.asarray(y))


async def run_models(
    storage: StorageBackend, settings: Settings, trends: list[BuiltTrend], now: datetime,
) -> tuple[list[tuple[BuiltTrend, TrendModeling]], dict]:
    if not trends:
        return [], {}
    feats = {f.reel_id: f for f in await storage.all_features()}

    hawkes_by, spikem_by, emerging_by = {}, {}, {}
    age_days, peak_plays, niche_share = {}, {}, {}
    for bt in trends:
        age_h = max((now - bt.first_seen).total_seconds() / 3600.0, 1e-3)
        age_days[bt.trend_id] = age_h / 24.0
        hawkes_by[bt.trend_id] = fit_hawkes(bt.event_times_h, settings.trends.bucket_s,
                                            settings.models, now_h=age_h)
        spikem_by[bt.trend_id] = fit_spikem(bt.curve.counts, settings.trends.bucket_s, now_h=age_h)
        emerging_by[bt.trend_id] = compute_emerging(bt.curve, settings.emerging)
        pk, niche = 0, 0
        authors: set[str] = set()
        for rid in bt.member_ids:
            le = await storage.latest_engagement(rid)
            pk = max(pk, le.plays if le else 0)
            f = feats.get(rid)
            if f and f.scene.is_niche:
                niche += 1
            r = await storage.get_reel(rid)
            if r:
                authors.add(r.author_handle)
        peak_plays[bt.trend_id] = pk
        niche_share[bt.trend_id] = niche / max(1, bt.size)
        bt._authors = len(authors)  # type: ignore[attr-defined]

    cox = _fit_cohort(trends, hawkes_by, emerging_by, settings)
    early = _fit_early_classifier(trends, emerging_by, niche_share, peak_plays, settings)

    # Fuse + calibrate the MONOTONE persistence score on the temporal backtest.
    raws, labels = build_calibration_set(trends, settings)
    fusion = FusionModel(method=settings.models.calibration)
    calib_info: dict = {"method": settings.models.calibration, "n": int(len(labels))}
    if len(labels) >= 4 and len(np.unique(labels)) >= 2:
        fusion.fit_scores(raws, labels)
        cal = np.array([fusion.calibrator.transform(float(r)) for r in raws])
        rep = backtest(cal, labels)
        calib_info.update({"brier": round(rep.brier, 4), "ece": round(rep.ece, 4),
                           "base_rate": round(rep.base_rate, 4), "reliability": rep.reliability})
    else:
        log.warning("calibration_skipped", n=len(labels))

    items: list[tuple[BuiltTrend, TrendModeling]] = []
    for bt in trends:
        h, sp, em = hawkes_by[bt.trend_id], spikem_by[bt.trend_id], emerging_by[bt.trend_id]
        survival_cond = cox.conditional_persistence(
            age_days[bt.trend_id], settings.models.persistence_horizon_days,
            _covariates(bt, h, em))
        em.p_viral = early.predict_proba(np.asarray(_early_feats(bt, em, niche_share)))
        sv = _survival_proxy(np.asarray(bt.curve.counts, dtype=np.float64))
        raw = raw_persistence_score(h.r_t, sv, sp.phase, em.velocity)
        votes = component_votes(h.r_t, survival_cond, sp.phase)
        persistence = fusion.predict_raw(raw, votes)
        health = health_score(h.r_t, float(np.tanh(em.velocity)), sp.phase, bt.size,
                              h.supercritical)
        opp = opportunity_score(health, em.acceleration, niche_share[bt.trend_id],
                                settings.emerging)
        items.append((bt, TrendModeling(
            hawkes=h, spikem=sp, emerging=em, health=health, persistence=persistence,
            opportunity=opp, survival_cond=survival_cond)))
    log.info("models_run", trends=len(items), brier=calib_info.get("brier"))
    return items, calib_info


def _covariates(bt: BuiltTrend, h: HawkesResult, em: EmergingSignals) -> np.ndarray:
    return np.array([np.log1p(bt.size), h.r_t, em.velocity], dtype=np.float64)


def _early_feats(
    bt: BuiltTrend, em: EmergingSignals, niche_share: dict[str, float]
) -> list[float]:
    authors = getattr(bt, "_authors", bt.size)
    return early_window_features(bt.curve, 24.0, niche_share[bt.trend_id], authors)


def _fit_cohort(
    trends: list[BuiltTrend], hawkes_by: dict[str, HawkesResult],
    emerging_by: dict[str, EmergingSignals], settings: Settings,
) -> CoxPH:
    X, durations, events = [], [], []
    for bt in trends:
        dur, ev = detect_death(bt.curve.counts, settings.trends.bucket_s,
                               settings.models.death_drop_pct,
                               settings.models.death_consecutive_buckets)
        X.append(_covariates(bt, hawkes_by[bt.trend_id], emerging_by[bt.trend_id]))
        durations.append(dur)
        events.append(ev)
    return fit_cox(np.vstack(X), np.asarray(durations), np.asarray(events), ridge=1.0)


def _fit_early_classifier(
    trends: list[BuiltTrend], emerging_by: dict[str, EmergingSignals],
    niche_share: dict[str, float], peak_plays: dict[str, int], settings: Settings,
) -> EarlyClassifier:
    X, y = [], []
    for bt in trends:
        X.append(_early_feats(bt, emerging_by[bt.trend_id], niche_share))
        y.append(1 if peak_plays[bt.trend_id] >= settings.emerging.virality_plays_threshold else 0)
    clf = EarlyClassifier()
    clf.fit(np.asarray(X), np.asarray(y))
    return clf
