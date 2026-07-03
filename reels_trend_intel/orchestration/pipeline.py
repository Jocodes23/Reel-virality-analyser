"""Offline run-once pipeline: collect (adaptive) -> extract -> cluster -> model -> report.

This is the golden path. Collection runs over a SIMULATED clock so the adaptive
resampler builds genuinely sparse-over-time engagement series (frequent while
young/accelerating, rare when old, retired when dead) — and we measure the result
against a naive fixed-interval baseline to prove the efficiency design.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import timedelta
from functools import partial

from reels_trend_intel.collectors.resampler import plan_next
from reels_trend_intel.collectors.sample_adapter import SampleAdapter
from reels_trend_intel.collectors.scheduler import PoliteScheduler
from reels_trend_intel.config.settings import Settings
from reels_trend_intel.features import FeatureInput, FeaturePipeline
from reels_trend_intel.fixtures import SyntheticDataset
from reels_trend_intel.observability.logging import get_logger
from reels_trend_intel.observability.metrics import snapshot
from reels_trend_intel.orchestration.modeling import run_models
from reels_trend_intel.report import build_report, export_all
from reels_trend_intel.report.schema import TrendReport
from reels_trend_intel.storage import make_storage
from reels_trend_intel.storage.base import StorageBackend
from reels_trend_intel.storage.rows import ReelSchedule

log = get_logger("orchestration.pipeline")


@dataclass
class PipelineResult:
    report: TrendReport
    stats: dict = field(default_factory=dict)
    exports: dict = field(default_factory=dict)


async def _collect_adaptive(
    storage: StorageBackend, settings: Settings, adapter: SampleAdapter,
    scheduler: PoliteScheduler, sim_step_minutes: int,
) -> dict:
    ds = adapter.dataset
    clock = ds.sim_start
    end = ds.sim_start + timedelta(hours=ds.horizon_h)
    step = timedelta(minutes=sim_step_minutes)
    samples = 0
    discovered_total = 0
    retired = 0
    deferred = 0

    while clock <= end:
        adapter.set_clock(clock)
        for reel, audio in await adapter.discover(10_000):
            if audio:
                await storage.upsert_audio([audio])
            reel.first_seen = clock
            reel.last_seen = clock
            await storage.upsert_reels([reel])
            await storage.upsert_schedule(ReelSchedule(
                reel_id=reel.reel_id, posted_at=reel.posted_at, next_sample_at=clock,
                tier="hot", last_plays=0))
            discovered_total += 1

        due = await storage.due_for_resample(clock, limit=100_000)
        frac = await scheduler.remaining_fraction("sample")
        for sched in due:
            # budget-aware degradation: under pressure, only sample HOT reels.
            if frac < settings.sampling.budget_low_fraction and sched.tier != "hot":
                await storage.set_next_sample(sched.reel_id, clock + timedelta(hours=1),
                                              sched.tier, sched.last_plays)
                deferred += 1
                continue
            tracked = await storage.get_reel(sched.reel_id)
            if tracked is None:
                continue
            eng = await scheduler.guard(
                "sample", partial(adapter.fetch_engagement, tracked, clock))
            if eng is not None:
                await storage.append_engagement([eng])
                samples += 1
            series = await storage.engagement_series(sched.reel_id)
            decision = plan_next(sched.posted_at, series, clock, settings.sampling)
            if decision.retire:
                await storage.retire_reel(sched.reel_id)
                retired += 1
            else:
                await storage.set_next_sample(
                    sched.reel_id, decision.next_at, decision.tier,
                    eng.plays if eng else sched.last_plays)
        clock += step

    # naive fixed-interval baseline: poll every tracked reel every step while alive.
    naive = 0
    for reel in await storage.list_reels():
        alive_h = (end - reel.posted_at).total_seconds() / 3600.0
        naive += max(1, int(alive_h * 60 / sim_step_minutes))
    savings = 1.0 - (samples / naive) if naive else 0.0
    return {
        "discovered": discovered_total, "engagement_samples": samples, "retired": retired,
        "deferred_under_budget": deferred, "naive_fixed_interval_samples": naive,
        "adaptive_sampling_savings_pct": round(100 * savings, 1),
        "sim_step_minutes": sim_step_minutes,
    }


async def _extract_all(storage: StorageBackend, settings: Settings,
                       adapter: SampleAdapter) -> dict:
    reels = await storage.list_reels()
    fp = FeaturePipeline(storage, settings)
    survivors = 0
    batch: list[FeatureInput] = []
    for reel in reels:
        media = await adapter.fetch_media(reel)
        meta = await adapter.fetch_metadata(reel)
        latest = await storage.latest_engagement(reel.reel_id)
        audio = await storage.get_audio(reel.audio_id) if reel.audio_id else None
        batch.append(FeatureInput(reel=reel, audio=audio, media=media, metadata=meta,
                                  latest=latest))
        if len(batch) >= settings.features.batch_size * 2:
            survivors += len(await fp.extract_batch(batch))
            batch = []
    if batch:
        survivors += len(await fp.extract_batch(batch))
    gated = len(reels) - survivors
    return {"reels": len(reels), "featurized_survivors": survivors, "gated_out": gated,
            "expensive_calls_saved_by_gate": gated}


async def run_offline_pipeline(
    settings: Settings | None = None, sim_step_minutes: int = 30, seed: int | None = None,
) -> PipelineResult:
    settings = settings or Settings()
    if seed is not None:
        settings.app.seed = seed
    timings: dict[str, float] = {}

    storage = make_storage(settings)
    await storage.connect()
    await storage.init_schema()

    dataset = SyntheticDataset(seed=settings.app.seed)
    adapter = SampleAdapter(dataset)
    # Generous limits for the simulation (no real sleeps); budget/token mechanics
    # still run so the accounting is real.
    sim_cfg = settings.collection.model_copy(update={
        "requests_per_minute": 1_000_000.0, "bucket_capacity": 10_000,
        "daily_request_cap": 100_000_000,
    })
    scheduler = PoliteScheduler(storage, sim_cfg, seed=settings.app.seed, apply_delays=False)

    t = time.perf_counter()
    collect_stats = await _collect_adaptive(storage, settings, adapter, scheduler,
                                            sim_step_minutes)
    timings["collect_s"] = round(time.perf_counter() - t, 2)

    t = time.perf_counter()
    extract_stats = await _extract_all(storage, settings, adapter)
    timings["extract_s"] = round(time.perf_counter() - t, 2)

    from reels_trend_intel.trends import build_trends

    t = time.perf_counter()
    trends = await build_trends(storage, settings)
    timings["cluster_s"] = round(time.perf_counter() - t, 2)

    now = dataset.sim_start + timedelta(hours=dataset.horizon_h)
    t = time.perf_counter()
    items, calib_info = await run_models(storage, settings, trends, now)
    timings["model_s"] = round(time.perf_counter() - t, 2)

    t = time.perf_counter()
    report = await build_report(storage, settings, items, now, calibration=calib_info)
    await storage.write_report(report.report_version, report.generated_at, report.ranking_key,
                               report.to_json())
    exports = export_all(report, settings.report.export_dir)
    timings["report_s"] = round(time.perf_counter() - t, 2)

    stats = {
        "collect": collect_stats, "extract": extract_stats,
        "trends": len(trends), "calibration": calib_info, "timings_s": timings,
        "metrics": snapshot(),
    }
    log.info("pipeline_done", trends=len(trends), **timings)
    await storage.close()
    return PipelineResult(report=report, stats=stats, exports=exports)
