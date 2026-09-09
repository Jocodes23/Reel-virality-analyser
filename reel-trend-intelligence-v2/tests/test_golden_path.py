"""Golden-path integration test: collect -> extract -> cluster -> model -> report.

Runs the ENTIRE pipeline on synthetic fixtures with ZERO network. Asserts the
report is well-formed, ranked, calibrated-in-range, and that the efficiency design
(adaptive sampling + tiered gate) actually saved work.
"""

from __future__ import annotations

import json
import os

import pytest

from reels_trend_intel.orchestration import run_offline_pipeline


@pytest.mark.asyncio
async def test_offline_golden_path(settings):
    res = await run_offline_pipeline(settings, sim_step_minutes=120)
    report = res.report

    # --- structure ---
    assert len(report.trends) >= 3
    assert report.report_version and report.ranking_key in {"health", "opportunity"}

    # ranked by health (default), non-increasing
    healths = [r.health_score for r in report.trends]
    assert healths == sorted(healths, reverse=True)

    for rec in report.trends:
        # A. headline links always present + canonical URLs
        assert rec.headline_links.top_reel.url.startswith("https://www.instagram.com/reel/")
        assert "/reels/audio/" in rec.headline_links.audio.instagram_audio_url
        assert rec.headline_links.top_reel.plays >= 0
        # C. calibrated probability in range with a CI
        d = rec.dynamics
        assert 0.0 <= d.persistence_probability <= 1.0
        assert d.persistence_ci.low <= d.persistence_probability <= d.persistence_ci.high
        assert d.phase in {"pre-peak", "peak", "decay"}
        # D. creative recipe present
        assert rec.creative_signal.palette
        # E. member drill-down carries per-reel series + features (nothing dropped)
        assert rec.metadata.members
        m0 = rec.metadata.members[0]
        assert m0.permalink.startswith("https://www.instagram.com/reel/")
        assert "dominant_hex" in m0.color
        # F. provenance stamped
        assert rec.provenance.feature_schema_version and rec.provenance.model_versions

    # --- efficiency: adaptive sampling beat naive fixed-interval ---
    collect = res.stats["collect"]
    assert collect["engagement_samples"] < collect["naive_fixed_interval_samples"]
    assert collect["adaptive_sampling_savings_pct"] > 20.0
    assert collect["retired"] > 0  # dead reels retired

    # tiered gate did some filtering work (dedupe/quality)
    assert res.stats["extract"]["featurized_survivors"] > 0

    # --- exports written, full record in JSONL ---
    assert os.path.exists(res.exports["jsonl"])
    assert os.path.exists(res.exports["parquet"])
    assert os.path.exists(res.exports["csv"])
    with open(res.exports["jsonl"], encoding="utf-8") as f:
        first = json.loads(f.readline())
    assert "headline_links" in first and "metadata" in first
    assert first["metadata"]["members"]  # drill-down present in export


@pytest.mark.asyncio
async def test_reproducible_seed(settings):
    r1 = await run_offline_pipeline(settings, sim_step_minutes=180, seed=123)
    ids1 = [t.identity.trend_id for t in r1.report.trends]
    settings.storage.sqlite_path = settings.storage.sqlite_path + ".2"
    r2 = await run_offline_pipeline(settings, sim_step_minutes=180, seed=123)
    ids2 = [t.identity.trend_id for t in r2.report.trends]
    assert ids1 == ids2  # deterministic given a seed
