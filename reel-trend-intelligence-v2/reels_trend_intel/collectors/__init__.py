from __future__ import annotations

from typing import Any

from reels_trend_intel.collectors.base import CollectorAdapter, CollectorHalted
from reels_trend_intel.collectors.sample_adapter import SampleAdapter
from reels_trend_intel.collectors.scheduler import BudgetExhausted, PoliteScheduler

# value is either an adapter class or a "module:Class" lazy-import path
_REGISTRY: dict[str, type[CollectorAdapter] | str] = {
    "sample": SampleAdapter,
    "graph_api": "reels_trend_intel.collectors.graph_api:GraphApiAdapter",
    "licensed_provider": "reels_trend_intel.collectors.licensed_provider:LicensedProviderAdapter",
    "owned_session": "reels_trend_intel.collectors.owned_session:OwnedSessionAdapter",
}


def make_adapter(name: str, **kwargs: Any) -> CollectorAdapter:
    """Construct an adapter by name (config-driven active source set)."""
    target = _REGISTRY.get(name)
    if target is None:
        raise ValueError(f"unknown adapter: {name}")
    if isinstance(target, str):
        mod, _, cls = target.partition(":")
        import importlib

        obj = getattr(importlib.import_module(mod), cls)
        return obj(**kwargs)
    return target(**kwargs)


__all__ = [
    "CollectorAdapter",
    "CollectorHalted",
    "PoliteScheduler",
    "BudgetExhausted",
    "SampleAdapter",
    "make_adapter",
]
