"""Resilient long-running production loop (APScheduler).

Per-stage scheduling with retries, dead-letter handling, graceful shutdown and
backpressure:
  * collect tick   — discover new reels + resample DUE reels (adaptive) across all
                     configured adapters, through the polite scheduler; halts an
                     adapter (never routes around) on a login challenge.
  * rebuild tick   — extract pending features -> build trends -> run models ->
                     write + export the TrendReport.

Offline this idles politely (live adapters are inert without credentials); the
offline golden path (orchestration.pipeline) is the proven path. Production swaps
in real adapters via config.
"""

from __future__ import annotations

import asyncio
import signal
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from functools import partial

from reels_trend_intel.collectors import CollectorHalted, make_adapter
from reels_trend_intel.collectors.resampler import plan_next
from reels_trend_intel.collectors.scheduler import BudgetExhausted, PoliteScheduler
from reels_trend_intel.config.settings import Settings
from reels_trend_intel.features import FeatureInput, FeaturePipeline
from reels_trend_intel.observability.logging import get_logger
from reels_trend_intel.observability.metrics import METRICS
from reels_trend_intel.storage.rows import ReelSchedule

log = get_logger("orchestration.loop")


class ResilientLoop:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        from reels_trend_intel.storage import make_storage

        self.storage = make_storage(settings)
        self.scheduler = PoliteScheduler(self.storage, settings.collection, settings.app.seed)
        self.adapters = {n: make_adapter(n) for n in settings.collection.enabled_adapters}
        self.fp = FeaturePipeline(self.storage, settings)
        self._stop = asyncio.Event()

    async def collect_tick(self) -> None:
        now = datetime.now(UTC)
        for name, adapter in list(self.adapters.items()):
            if adapter.halted:
                continue
            try:
                discovered = await self.scheduler.guard(name, partial(adapter.discover, 50))
                for reel, audio in discovered:
                    if audio:
                        await self.storage.upsert_audio([audio])
                    reel.first_seen = reel.first_seen or now
                    reel.last_seen = now
                    await self.storage.upsert_reels([reel])
                    await self.storage.upsert_schedule(ReelSchedule(
                        reel_id=reel.reel_id, posted_at=reel.posted_at, next_sample_at=now,
                        tier="hot", last_plays=0))
            except CollectorHalted:
                log.error("adapter_halted", adapter=name)
                continue
            except BudgetExhausted:
                log.warning("budget_exhausted", adapter=name)
                continue
            except Exception as exc:  # dead-letter: log and move on
                log.error("collect_failed", adapter=name, error=str(exc))

        # resample DUE reels across the source they came from
        due = await self.storage.due_for_resample(now, limit=200)
        METRICS.queue_depth.labels("resample").set(len(due))
        for sched in due:
            tracked = await self.storage.get_reel(sched.reel_id)
            if tracked is None:
                continue
            src_adapter = self.adapters.get(tracked.source)
            if src_adapter is None or src_adapter.halted:
                continue
            try:
                eng = await self.scheduler.guard(
                    tracked.source, partial(src_adapter.fetch_engagement, tracked, now))
            except CollectorHalted:
                continue
            except Exception as exc:
                log.error("resample_failed", reel=sched.reel_id, error=str(exc))
                continue
            if eng is not None:
                await self.storage.append_engagement([eng])
            series = await self.storage.engagement_series(sched.reel_id)
            decision = plan_next(sched.posted_at, series, now, self.settings.sampling)
            if decision.retire:
                await self.storage.retire_reel(sched.reel_id)
            else:
                await self.storage.set_next_sample(sched.reel_id, decision.next_at,
                                                   decision.tier, eng.plays if eng else 0)

    async def rebuild_tick(self) -> None:
        from reels_trend_intel.orchestration.modeling import run_models
        from reels_trend_intel.report import build_report, export_all
        from reels_trend_intel.trends import build_trends

        now = datetime.now(UTC)
        reels = await self.storage.list_reels()
        batch: list[FeatureInput] = []
        for reel in reels:
            if "featurized" in await self.storage.stages_done(reel.reel_id):
                continue
            adapter = self.adapters.get(reel.source)
            if adapter is None:
                continue
            media = await adapter.fetch_media(reel)
            meta = await adapter.fetch_metadata(reel)
            latest = await self.storage.latest_engagement(reel.reel_id)
            audio = await self.storage.get_audio(reel.audio_id) if reel.audio_id else None
            batch.append(FeatureInput(reel=reel, audio=audio, media=media, metadata=meta,
                                      latest=latest))
        if batch:
            await self.fp.extract_batch(batch)
        trends = await build_trends(self.storage, self.settings)
        if not trends:
            return
        items, calib = await run_models(self.storage, self.settings, trends, now)
        report = await build_report(self.storage, self.settings, items, now, calibration=calib)
        await self.storage.write_report(report.report_version, report.generated_at,
                                        report.ranking_key, report.to_json())
        export_all(report, self.settings.report.export_dir)
        log.info("rebuild_done", trends=len(trends))

    async def run(self) -> None:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler

        await self.storage.connect()
        await self.storage.init_schema()
        sched = AsyncIOScheduler(timezone="UTC")
        sched.add_job(self._guarded(self.collect_tick), "interval", seconds=30,
                      max_instances=1, coalesce=True)
        sched.add_job(self._guarded(self.rebuild_tick), "interval", minutes=15,
                      max_instances=1, coalesce=True)
        sched.start()
        log.info("loop_started", adapters=list(self.adapters))
        self._install_signals()
        await self._stop.wait()
        sched.shutdown(wait=False)
        for a in self.adapters.values():
            await a.close()
        await self.storage.close()
        log.info("loop_stopped")

    def _guarded(
        self, coro_fn: Callable[[], Awaitable[None]]
    ) -> Callable[[], Awaitable[None]]:
        async def wrapper() -> None:
            try:
                await coro_fn()
            except Exception as exc:  # never let a tick kill the loop
                log.error("tick_error", error=str(exc))
        return wrapper

    def _install_signals(self) -> None:
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._stop.set)
            except NotImplementedError:  # Windows
                signal.signal(sig, lambda *_: self._stop.set())


async def run_loop(settings: Settings) -> None:
    await ResilientLoop(settings).run()
