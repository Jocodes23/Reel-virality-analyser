from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from reels_trend_intel.collectors.resampler import (
    normalized_acceleration,
    plan_next,
    tier_for_age,
)
from reels_trend_intel.config.settings import SamplingConfig
from reels_trend_intel.storage import make_storage
from reels_trend_intel.storage.rows import (
    Audio,
    EmbeddingRow,
    EngagementSample,
    Reel,
)


@pytest.fixture
async def db(settings):
    backend = make_storage(settings)
    await backend.connect()
    await backend.init_schema()
    yield backend
    await backend.close()


async def test_storage_roundtrip_and_dedupe(db):
    now = datetime.now(UTC)
    r = Reel.from_shortcode("ABC", author_handle="a", author_url="u", posted_at=now,
                            audio_id="aud1", phash=12345, first_seen=now, last_seen=now)
    await db.upsert_reels([r])
    await db.upsert_audio([Audio.from_id("aud1", song_title="S", artist="X")])
    got = await db.get_reel("ABC")
    assert got.permalink == "https://www.instagram.com/reel/ABC/"
    assert got.phash == 12345
    assert (await db.get_audio("aud1")).instagram_audio_url.endswith("/aud1/")
    dup = await db.seen_phash(12345, hamming=2)
    assert dup == "ABC"


async def test_engagement_series_and_knn(db):
    now = datetime.now(UTC)
    await db.append_engagement([
        EngagementSample(reel_id="R", sampled_at=now, plays=100, likes=10),
        EngagementSample(reel_id="R", sampled_at=now + timedelta(hours=1), plays=300, likes=40),
    ])
    ser = await db.engagement_series("R")
    assert [s.plays for s in ser] == [100, 300]
    await db.put_embeddings([EmbeddingRow(content_hash="h1", kind="image", vector=[1, 0, 0]),
                             EmbeddingRow(content_hash="h2", kind="image", vector=[0, 1, 0])])
    nn = await db.knn("image", [1, 0, 0], 1)
    assert nn[0][0] == "h1"


async def test_budget_increments(db):
    st = await db.incr_budget("sample", "2026-06-30", 1, 100)
    st = await db.incr_budget("sample", "2026-06-30", 1, 100)
    assert st.requests_used == 2 and st.remaining == 98


def test_resampler_interval_grows_with_age():
    cfg = SamplingConfig()
    now = datetime(2026, 6, 30, tzinfo=UTC)
    young = plan_next(now - timedelta(hours=1), [], now, cfg)
    old = plan_next(now - timedelta(hours=100), [], now, cfg)
    assert old.interval_s > young.interval_s
    assert tier_for_age(1, cfg) == "hot"


def test_resampler_retires_dead():
    cfg = SamplingConfig()
    now = datetime(2026, 6, 30, tzinfo=UTC)
    posted = now - timedelta(hours=200)
    # flat plays across the retirement window -> retire
    series = [EngagementSample(reel_id="R", sampled_at=posted + timedelta(hours=h),
                               plays=1000, likes=1) for h in (0, 100, 199)]
    decision = plan_next(posted, series, now, cfg)
    assert decision.retire is True


def test_normalized_acceleration_sign():
    now = datetime(2026, 6, 30, tzinfo=UTC)
    accel = [EngagementSample(reel_id="R", sampled_at=now + timedelta(hours=h), plays=p)
             for h, p in [(0, 0), (1, 100), (2, 400)]]  # accelerating
    assert normalized_acceleration(accel) > 0
