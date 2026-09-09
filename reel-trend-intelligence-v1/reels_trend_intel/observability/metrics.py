"""Metrics: per-stage throughput, queue depth, API-budget usage, VRAM/RAM.

Uses prometheus_client when available (exposed at /metrics), otherwise an
in-memory registry so the system runs identically without the dependency.
"""

from __future__ import annotations

import time
from collections import defaultdict
from types import TracebackType
from typing import Any

try:
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

    _HAS_PROM = True
except Exception:  # pragma: no cover
    _HAS_PROM = False


class _InMemoryMetric:
    def __init__(self) -> None:
        self.values: dict[tuple[str, ...], float] = defaultdict(float)

    def labels(self, *vals: str) -> _InMemoryBound:
        return _InMemoryBound(self, tuple(vals))


class _InMemoryBound:
    def __init__(self, parent: _InMemoryMetric, key: tuple[str, ...]) -> None:
        self.parent = parent
        self.key = key

    def inc(self, amount: float = 1.0) -> None:
        self.parent.values[self.key] += amount

    def set(self, value: float) -> None:
        self.parent.values[self.key] = value

    def observe(self, value: float) -> None:
        self.parent.values[self.key] = value


class Metrics:
    """Facade over prometheus or an in-memory fallback."""

    registry: Any
    processed: Any
    errors: Any
    queue_depth: Any
    stage_seconds: Any
    budget_used: Any
    resource: Any

    def __init__(self) -> None:
        if _HAS_PROM:
            self.registry = CollectorRegistry()
            self.processed = Counter(
                "rti_stage_processed_total", "Items processed per stage",
                ["stage"], registry=self.registry,
            )
            self.errors = Counter(
                "rti_stage_errors_total", "Errors per stage",
                ["stage"], registry=self.registry,
            )
            self.queue_depth = Gauge(
                "rti_queue_depth", "Current queue depth per stage",
                ["stage"], registry=self.registry,
            )
            self.stage_seconds = Histogram(
                "rti_stage_seconds", "Stage processing seconds",
                ["stage"], registry=self.registry,
            )
            self.budget_used = Gauge(
                "rti_api_budget_used", "Daily request budget used per adapter",
                ["adapter"], registry=self.registry,
            )
            self.resource = Gauge(
                "rti_resource", "Resource usage gauge (vram_mb, ram_mb, ...)",
                ["kind"], registry=self.registry,
            )
        else:  # pragma: no cover
            self.registry = None
            self.processed = _InMemoryMetric()
            self.errors = _InMemoryMetric()
            self.queue_depth = _InMemoryMetric()
            self.stage_seconds = _InMemoryMetric()
            self.budget_used = _InMemoryMetric()
            self.resource = _InMemoryMetric()

    def render(self) -> bytes:
        if _HAS_PROM:
            from prometheus_client import generate_latest

            return generate_latest(self.registry)
        # Minimal text rendering of the in-memory store.
        lines: list[str] = []
        for name, metric in (
            ("processed", self.processed),
            ("errors", self.errors),
            ("queue_depth", self.queue_depth),
            ("budget_used", self.budget_used),
            ("resource", self.resource),
        ):
            assert isinstance(metric, _InMemoryMetric)
            for key, val in metric.values.items():
                lbl = ",".join(key)
                lines.append(f"rti_{name}{{{lbl}}} {val}")
        return ("\n".join(lines) + "\n").encode()


METRICS = Metrics()


class StageTimer:
    """Context manager that records throughput + latency for a pipeline stage."""

    def __init__(self, stage: str, count: int = 1) -> None:
        self.stage = stage
        self.count = count
        self._start = 0.0

    def __enter__(self) -> StageTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        elapsed = time.perf_counter() - self._start
        METRICS.stage_seconds.labels(self.stage).observe(elapsed)
        if exc_type is not None:
            METRICS.errors.labels(self.stage).inc()
        else:
            METRICS.processed.labels(self.stage).inc(self.count)

    def set_count(self, count: int) -> None:
        self.count = count


def record_resource(kind: str, value: float) -> None:
    METRICS.resource.labels(kind).set(value)


def record_budget(adapter: str, used: float) -> None:
    METRICS.budget_used.labels(adapter).set(used)


def snapshot() -> dict[str, Any]:
    """Lightweight dict snapshot for the dashboard / health endpoint."""
    out: dict[str, Any] = {}
    if not _HAS_PROM:
        for name, metric in (
            ("processed", METRICS.processed),
            ("errors", METRICS.errors),
            ("queue_depth", METRICS.queue_depth),
            ("budget_used", METRICS.budget_used),
            ("resource", METRICS.resource),
        ):
            assert isinstance(metric, _InMemoryMetric)
            out[name] = {",".join(k): v for k, v in metric.values.items()}
    return out
