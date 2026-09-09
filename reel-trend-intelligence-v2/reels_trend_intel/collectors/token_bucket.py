"""Async token-bucket rate limiter shared by the polite scheduler."""

from __future__ import annotations

import asyncio
import time


class TokenBucket:
    def __init__(self, rate_per_minute: float, capacity: int) -> None:
        self.rate_per_s = rate_per_minute / 60.0
        self.capacity = float(capacity)
        self._tokens = float(capacity)
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        self._last = now
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate_per_s)

    async def acquire(self, tokens: float = 1.0) -> None:
        """Block until `tokens` are available, then consume them."""
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                deficit = tokens - self._tokens
                wait = deficit / self.rate_per_s if self.rate_per_s > 0 else 0.05
            await asyncio.sleep(min(wait, 5.0))

    @property
    def available(self) -> float:
        self._refill()
        return self._tokens
