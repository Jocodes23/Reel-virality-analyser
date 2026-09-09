"""LIVE public-reels run: real Instagram data -> real trends.

Discovery + engagement come from Instagram Graph API **Business Discovery** over a
configurable list of public Business/Creator accounts (hashtag search additionally
requires Meta App Review). Business Discovery returns each account's media *with*
like/comment counts in one call, so we seed the first engagement sample straight
from the discovery payload instead of spending an API call per reel.

Then the normal pipeline runs unchanged: gated+cached feature extraction (real CLIP
on the GPU over actual reel thumbnails) -> clustering -> Hawkes/survival/SpikeM ->
calibrated persistence -> TrendReport + exports.
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import UTC, datetime

PROJECT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT)


def _load_dotenv() -> None:
    path = os.path.join(PROJECT, ".env")
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v


async def main() -> None:
    _load_dotenv()
    from reels_trend_intel.collectors.graph_api import GraphApiAdapter
    from reels_trend_intel.collectors.scheduler import PoliteScheduler, utc_day
    from reels_trend_intel.config.settings import Settings
    from reels_trend_intel.features import FeatureInput, FeaturePipeline
    from reels_trend_intel.orchestration.modeling import run_models
    from reels_trend_intel.report import build_report, export_all
    from reels_trend_intel.storage import make_storage
    from reels_trend_intel.storage.rows import EngagementSample
    from reels_trend_intel.trends import build_trends

    s = Settings()
    print(f"storage={s.storage.sqlite_path} | embeddings={s.features.embedding_backend}"
          f" | device={s.features.device}")
    db = make_storage(s)
    await db.connect()
    await db.init_schema()
    adapter = GraphApiAdapter(hashtags=s.collection.graph_hashtags,
                              seed_usernames=s.collection.graph_seed_usernames)
    sched = PoliteScheduler(db, s.collection, apply_delays=True)  # real limits on API calls
    timings: dict[str, float] = {}

    # --- 1. DISCOVER real public reels -------------------------------------
    t = time.perf_counter()
    found = await sched.guard("graph_api", lambda: adapter.discover(1000))
    now = datetime.now(UTC)
    seeded = 0
    for reel, audio in found:
        reel.first_seen = reel.first_seen or now
        reel.last_seen = now
        await db.upsert_reels([reel])
        if audio:
            await db.upsert_audio([audio])
        # Seed engagement from the SAME payload (no extra API call).
        for m in await adapter._business_discovery(reel.author_handle):
            if (m.get("permalink") or "").rstrip("/").endswith(reel.shortcode):
                await db.append_engagement([EngagementSample(
                    reel_id=reel.reel_id, sampled_at=now, plays=0,
                    likes=int(m.get("like_count") or 0),
                    comments=int(m.get("comments_count") or 0), shares=0, saves=0)])
                seeded += 1
                break
    timings["collect_s"] = round(time.perf_counter() - t, 1)
    budget = await db.get_budget("graph_api", utc_day())
    print(f"[collect] {len(found)} public reels discovered, {seeded} engagement-seeded "
          f"| API budget used {budget.requests_used}/{s.collection.daily_request_cap} "
          f"| {timings['collect_s']}s")
    if not found:
        print("no reels discovered — nothing to model")
        await adapter.close()
        await db.close()
        return

    # --- 2. EXTRACT features (real CLIP on GPU over real thumbnails) -------
    t = time.perf_counter()
    fp = FeaturePipeline(db, s)
    reels = await db.list_reels()
    batch: list[FeatureInput] = []
    survivors = 0
    for reel in reels:
        try:
            media = await adapter.fetch_media(reel)
        except Exception:
            media = None
        latest = await db.latest_engagement(reel.reel_id)
        audio = await db.get_audio(reel.audio_id) if reel.audio_id else None
        batch.append(FeatureInput(reel=reel, audio=audio, media=media, metadata={},
                                  latest=latest))
        if len(batch) >= s.features.batch_size * 2:
            survivors += len(await fp.extract_batch(batch))
            batch = []
    if batch:
        survivors += len(await fp.extract_batch(batch))
    timings["extract_s"] = round(time.perf_counter() - t, 1)
    print(f"[extract] {survivors}/{len(reels)} featurized "
          f"({len(reels) - survivors} gated by dedupe/quality) | {timings['extract_s']}s")

    # --- 3. CLUSTER + 4. MODEL + 5. REPORT --------------------------------
    t = time.perf_counter()
    trends = await build_trends(db, s)
    timings["cluster_s"] = round(time.perf_counter() - t, 1)
    print(f"[cluster] {len(trends)} trends | {timings['cluster_s']}s")
    if not trends:
        print("no trends formed (need >= hdbscan_min_cluster_size similar reels)")
        await adapter.close()
        await db.close()
        return

    t = time.perf_counter()
    items, calib = await run_models(db, s, trends, now)
    timings["model_s"] = round(time.perf_counter() - t, 1)
    report = await build_report(db, s, items, now, calibration=calib)
    await db.write_report(report.report_version, report.generated_at, report.ranking_key,
                          report.to_json())
    exports = export_all(report, s.report.export_dir)
    print(f"[model] calibration={ {k: v for k, v in calib.items() if k != 'reliability'} } "
          f"| {timings['model_s']}s")

    print("\n=== LIVE TREND REPORT (real Instagram public reels) ===")
    for i, rec in enumerate(report.trends, 1):
        d = rec.dynamics
        h = rec.headline_links
        print(f"{i:2d}. {rec.identity.label[:44]:44s} {rec.identity.type:7s} "
              f"n={rec.identity.size:3d} {d.phase:8s} R={d.r_t:5.2f} "
              f"health={rec.health_score:.2f} persist={d.persistence_probability:.2f}"
              f"[{d.persistence_ci.low:.2f}-{d.persistence_ci.high:.2f}]"
              f"{'  WHITESPACE' if rec.is_whitespace else ''}")
        print(f"     top: {h.top_reel.url}  @{h.top_reel.author_handle}  "
              f"{h.top_reel.likes:,} likes / {h.top_reel.comments:,} comments")
        print(f"     recipe: {rec.creative_signal.mood}/{rec.creative_signal.composition} "
              f"palette={[p.hex for p in rec.creative_signal.palette[:3]]} "
              f"best_hour={rec.creative_signal.best_posting_hour} "
              f"tags={' '.join(rec.creative_signal.hashtags[:4])}")
    print(f"\ntimings: {timings}")
    print("exports:", exports)
    await adapter.close()
    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
