from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from reels_trend_intel.config.settings import reset_settings_cache
from reels_trend_intel.report.export import export_all
from reels_trend_intel.report.schema import (
    AudioInfo,
    ConfidenceInterval,
    CreativeSignal,
    Dynamics,
    HeadlineLinks,
    Identity,
    PaletteColor,
    Provenance,
    TopReel,
    TrendMetadata,
    TrendRecord,
    TrendReport,
)
from reels_trend_intel.storage import make_storage
from reels_trend_intel.storage.rows import canonical_audio_url, canonical_permalink


def _sample_report() -> TrendReport:
    now = datetime(2026, 7, 1, tzinfo=UTC)
    prov = Provenance(model_versions={"hawkes": "1.0.0"}, feature_schema_version="1.0.0",
                      report_version="1.0.0", sample_size=10, data_last_updated=now)
    rec = TrendRecord(
        headline_links=HeadlineLinks(
            top_reel=TopReel(url=canonical_permalink("ABC"), reel_id="ABC", author_handle="chef",
                             author_url="u", plays=1000, likes=100, comments=5, shares=3, saves=9,
                             posted_at=now, why_top="peak plays 1,000"),
            audio=AudioInfo(audio_id="aud1", instagram_audio_url=canonical_audio_url("aud1"),
                            song_title="Sizzle", artist="X")),
        identity=Identity(trend_id="tr_food_1", type="visual", label="smash burger #food",
                          member_reel_ids=["ABC"], size=10, first_seen=now),
        dynamics=Dynamics(phase="pre-peak", velocity=1.0, acceleration=0.2, r_t=1.3,
                          trend_health_score=0.8, persistence_probability=0.75,
                          persistence_ci=ConfidenceInterval(low=0.6, high=0.9),
                          persistence_horizon_days=3.0),
        creative_signal=CreativeSignal(
            palette=[PaletteColor(hex="#aa5522", proportion=0.4)], mood="low-key",
            lighting="low-key/warm", composition="food-closeup", warm_cool_ratio=1.5,
            best_posting_hour=18, best_posting_day=4, caption_template="t", hashtags=["#food"],
            example_reel_ids=["ABC"]),
        metadata=TrendMetadata(aggregated_features={"top_terms": ["food"]}, members=[]),
        provenance=prov, health_score=0.8, opportunity_score=0.2, is_whitespace=False)
    return TrendReport(report_version="1.0.0", generated_at=now, ranking_key="health",
                       niche="food/restaurant", provenance=prov, trends=[rec])


def test_export_roundtrip(tmp_path):
    report = _sample_report()
    written = export_all(report, str(tmp_path))
    assert written["jsonl"] and written["parquet"] and written["csv"]
    import polars as pl

    df = pl.read_parquet(written["parquet"])
    assert "top_reel_url" in df.columns and "record_json" in df.columns
    assert df["audio_instagram_url"][0].endswith("/aud1/")


@pytest.fixture
async def api_client(settings, monkeypatch):
    # populate a db with one report, then point the app at it
    monkeypatch.setenv("RTI_STORAGE__SQLITE_PATH", settings.storage.sqlite_path)
    reset_settings_cache()
    db = make_storage(settings)
    await db.connect()
    await db.init_schema()
    rep = _sample_report()
    await db.write_report(rep.report_version, rep.generated_at, rep.ranking_key, rep.to_json())
    await db.close()
    from reels_trend_intel.api.app import app

    with TestClient(app) as client:
        yield client
    reset_settings_cache()


async def test_api_trends_and_filters(api_client):
    r = api_client.get("/trends")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["trends"][0]["headline_links"]["top_reel"]["url"].endswith("/ABC/")

    # phase filter
    assert api_client.get("/trends?phase=decay").json()["count"] == 0
    assert api_client.get("/trends?phase=pre-peak").json()["count"] == 1
    # min_persistence filter
    assert api_client.get("/trends?min_persistence=0.9").json()["count"] == 0
    # niche filter (matches on label/terms)
    assert api_client.get("/trends?niche=food").json()["count"] == 1

    assert api_client.get("/health").json()["status"] == "ok"
    assert "rti_" in api_client.get("/metrics").text or api_client.get("/metrics").text == ""
    assert api_client.get("/").status_code == 200
