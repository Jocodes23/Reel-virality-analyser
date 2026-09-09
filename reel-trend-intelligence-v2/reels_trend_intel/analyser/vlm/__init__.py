"""VLM provider registry — this is the toggle.

Engine code calls `make_vlm_adapter(cfg)` and never imports a provider. Selection
comes from `settings.analyser.vlm_provider` (env: `RTI_ANALYSER__VLM_PROVIDER`),
with a per-call `override` so a single analysis run can use a different provider
than the configured default.
"""

from __future__ import annotations

from reels_trend_intel.analyser.vlm.base import (
    PROMPT_VERSION,
    Availability,
    VLMAdapter,
    VLMResult,
    VLMUsage,
    analyse_with_retry,
)
from reels_trend_intel.config.settings import AnalyserConfig

# Import paths, resolved lazily so an uninstalled provider SDK never breaks import.
_REGISTRY: dict[str, str] = {
    "local": "reels_trend_intel.analyser.vlm.local_vlm:LocalVLMAdapter",
    "anthropic": "reels_trend_intel.analyser.vlm.anthropic_vlm:AnthropicVLMAdapter",
    "openai": "reels_trend_intel.analyser.vlm.openai_vlm:OpenAIVLMAdapter",
}

PROVIDERS = tuple(_REGISTRY)
# "auto" is a selection mode, not an adapter: prefer a hosted model when a key is
# present (better labels), else fall back to local (free, offline).
_AUTO_ORDER = ("anthropic", "openai")


def resolve_provider(cfg: AnalyserConfig, override: str | None = None) -> str:
    """Turn the configured/overridden choice into a concrete provider name.

    Only "auto" inspects the environment — an explicit choice is always honoured
    even if its credential is missing, so a misconfiguration surfaces loudly
    instead of silently running on a different model than you asked for.
    """
    name = (override or cfg.vlm_provider).lower()
    if name != "auto":
        return name
    import os

    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"):
        return "anthropic"
    if os.getenv(cfg.openai_api_key_env):
        return "openai"
    return "local"


def make_vlm_adapter(cfg: AnalyserConfig, override: str | None = None) -> VLMAdapter:
    """Build the configured VLM adapter, or `override` for a one-off run."""
    name = resolve_provider(cfg, override)
    target = _REGISTRY.get(name)
    if target is None:
        raise ValueError(
            f"unknown VLM provider {name!r}; choose one of {', '.join(PROVIDERS)} or 'auto'"
        )
    import importlib

    module_path, _, cls_name = target.partition(":")
    cls = getattr(importlib.import_module(module_path), cls_name)
    return cls(cfg)  # type: ignore[no-any-return]


def availability_report(cfg: AnalyserConfig) -> dict[str, Availability]:
    """Availability of every provider — powers `rti vlm-info`. Never raises."""
    out: dict[str, Availability] = {}
    for name in PROVIDERS:
        try:
            out[name] = make_vlm_adapter(cfg, override=name).availability()
        except Exception as exc:
            out[name] = Availability(False, f"adapter error: {exc}")
    return out


__all__ = [
    "VLMAdapter", "VLMResult", "VLMUsage", "Availability", "analyse_with_retry",
    "make_vlm_adapter", "resolve_provider", "availability_report", "PROVIDERS",
    "PROMPT_VERSION",
]
