"""The single polite, rate-limited, budgeted scheduler all adapters go through.

Responsibilities:
  * global token bucket (conservative requests/min)
  * per-adapter hard daily request cap (budget), persisted -> restart-safe
  * randomized human-scale delays
  * exponential backoff with jitter on transient errors
  * HALT (never route around) on CollectorHalted
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TypeVar

from reels_trend_intel.collectors.base import CollectorHalted
from reels_trend_intel.collectors.token_bucket import TokenBucket
from reels_trend_intel.config.settings import CollectionConfig
from reels_trend_intel.observability.logging import get_logger
from reels_trend_intel.observability.metrics import record_budget
from reels_trend_intel.storage.base import StorageBackend

T = TypeVar("T")
log = get_logger("collectors.scheduler")


class BudgetExhausted(Exception):
    pass


def utc_day(now: datetime | None = None) -> str:
    return (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y-%m-%d")


class PoliteScheduler:
    def __init__(
        self,
        storage: StorageBackend,
        cfg: CollectionConfig,
        seed: int = 1729,
        apply_delays: bool = True,
    ) -> None:
        self.storage = storage
        self.cfg = cfg
        self.bucket = TokenBucket(cfg.requests_per_minute, cfg.bucket_capacity)
        self._rng = random.Random(seed)
        self.apply_delays = apply_delays

    async def _check_budget(self, adapter: str) -> None:
        day = utc_day()
        state = await self.storage.get_budget(adapter, day)
        used = state.requests_used
        if used >= self.cfg.daily_request_cap:
            raise BudgetExhausted(f"{adapter} daily cap {self.cfg.daily_request_cap} reached")

    async def _consume_budget(self, adapter: str) -> None:
        day = utc_day()
        state = await self.storage.incr_budget(adapter, day, 1, self.cfg.daily_request_cap)
        record_budget(adapter, state.requests_used)

    async def remaining_fraction(self, adapter: str) -> float:
        state = await self.storage.get_budget(adapter, utc_day())
        cap = self.cfg.daily_request_cap
        return max(0, cap - state.requests_used) / cap if cap > 0 else 0.0

    async def guard(self, adapter: str, fn: Callable[[], Awaitable[T]]) -> T:
        """Execute `fn` under rate limit + budget + backoff. Re-raise CollectorHalted."""
        await self._check_budget(adapter)
        await self.bucket.acquire(1.0)
        if self.apply_delays:
            await asyncio.sleep(self._rng.uniform(self.cfg.min_delay_s, self.cfg.max_delay_s))
        await self._consume_budget(adapter)

        attempt = 0
        while True:
            try:
                return await fn()
            except CollectorHalted:
                raise  # never route around a block
            except Exception as exc:  # transient
                attempt += 1
                if attempt > self.cfg.max_retries:
                    log.error("request_failed", adapter=adapter, attempts=attempt, error=str(exc))
                    raise
                backoff = min(
                    self.cfg.backoff_max_s,
                    self.cfg.backoff_base_s * (2 ** (attempt - 1)),
                )
                backoff *= 0.5 + self._rng.random()  # full jitter
                log.warning("request_retry", adapter=adapter, attempt=attempt,
                            backoff_s=round(backoff, 2), error=str(exc))
                await asyncio.sleep(backoff if self.apply_delays else 0.0)
