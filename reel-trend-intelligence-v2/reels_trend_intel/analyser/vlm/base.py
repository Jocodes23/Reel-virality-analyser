"""Provider-agnostic VLM adapter interface.

Engine code depends only on `VLMAdapter` and `analyse_with_retry`. No provider
SDK is imported here — each adapter lazily imports its own client so the package
installs and the tests run with none of them present.

Contract: one call per reel, over a single keyframe montage, returning
`VLMAnalysis` as strict JSON. On a parse/validation failure we retry
`cfg.vlm_retries` times, then return `AnalysisStatus.FAILED` — never raise into
the batch.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from reels_trend_intel.analyser.types import (
    ANALYSER_SCHEMA_VERSION,
    AnalysisStatus,
    VLMAnalysis,
    completeness,
)
from reels_trend_intel.config.settings import AnalyserConfig
from reels_trend_intel.observability.logging import get_logger

log = get_logger("analyser.vlm")

# Bump when the wording below changes — stamped onto every analysed row.
PROMPT_VERSION = "1.0.0"

SYSTEM_PROMPT = (
    "You are a video-production analyst. You are shown a MONTAGE of keyframes "
    "sampled from a single short-form vertical video (an Instagram Reel). Frames "
    "are in chronological order, left to right, top to bottom, each labelled with "
    "its timestamp in seconds.\n\n"
    "Judge only what is visible. Describe HOW the video was made, not what it is "
    "about. Be decisive but honest: when the montage genuinely does not support a "
    "call, pick the closest enum member and lower the corresponding confidence.\n\n"
    "shot_description must be 2-4 sentences about craft only — camera movement, "
    "framing, lens feel, lighting, stabilisation and transitions. Do not summarise "
    "the subject matter, and do not mention the montage or the frames themselves."
)


def build_user_prompt(duration_s: float | None, cut_count: int | None) -> str:
    """Numeric context the model should not have to guess from stills."""
    bits = ["Analyse this reel's production style from the montage."]
    if duration_s is not None:
        bits.append(f"Measured duration: {duration_s:.1f}s.")
    if cut_count is not None:
        bits.append(f"Measured cut count: {cut_count} (detected numerically).")
    bits.append("Return the analysis as JSON matching the required schema.")
    return " ".join(bits)


@dataclass
class VLMResult:
    """Outcome of one analysis attempt sequence."""

    status: AnalysisStatus
    analysis: VLMAnalysis | None = None
    error: str | None = None
    attempts: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    provider: str = ""
    model_id: str = ""
    prompt_version: str = PROMPT_VERSION
    schema_version: str = ANALYSER_SCHEMA_VERSION
    # Graceful degradation: 1.0 when the model answered every field. Missing
    # dimensions carry 0.0 confidence, so downstream can weight the row honestly.
    completeness: float = 1.0
    missing_fields: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status is AnalysisStatus.DONE and self.analysis is not None

    @property
    def partial(self) -> bool:
        return self.ok and bool(self.missing_fields)


@dataclass
class VLMUsage:
    """Token usage reported by a provider, for cost accounting."""

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Availability:
    ready: bool
    detail: str
    extras: list[str] = field(default_factory=list)


class VLMAdapter(ABC):
    """One analysis call per reel. Implemented once per provider."""

    provider: str = "base"

    def __init__(self, cfg: AnalyserConfig) -> None:
        self.cfg = cfg
        self.last_missing_fields: list[str] = []

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Concrete model identifier, recorded on every analysed row."""

    @abstractmethod
    def availability(self) -> Availability:
        """Can this adapter run right now? Never raises — used by `rti vlm-info`."""

    #: Fields the model omitted on the most recent successful call. Adapters that
    #: accept partial responses set this; strict adapters leave it empty.
    last_missing_fields: list[str]

    @abstractmethod
    async def analyse(
        self, montage_png: bytes, *, duration_s: float | None = None,
        cut_count: int | None = None,
    ) -> tuple[VLMAnalysis, VLMUsage]:
        """Return validated analysis + token usage, or raise on failure."""

    def cost_usd(self, usage: VLMUsage) -> float:
        """Default: free (local). Hosted adapters override with real pricing."""
        return 0.0

    async def aclose(self) -> None:  # pragma: no cover - default no-op
        return None


async def analyse_with_retry(
    adapter: VLMAdapter, montage_png: bytes, *, duration_s: float | None = None,
    cut_count: int | None = None, retries: int | None = None,
) -> VLMResult:
    """Run one reel through the adapter, retrying parse failures, never raising.

    A failed reel is marked FAILED and the caller moves on — a single bad reel
    must never take down an analysis batch.
    """
    max_retries = adapter.cfg.vlm_retries if retries is None else retries
    started = time.perf_counter()
    last_error = "unknown error"

    for attempt in range(1, max_retries + 2):
        adapter.last_missing_fields = []
        try:
            analysis, usage = await adapter.analyse(
                montage_png, duration_s=duration_s, cut_count=cut_count
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            log.warning("vlm_attempt_failed", provider=adapter.provider,
                        attempt=attempt, error=last_error[:200])
            continue
        missing = list(adapter.last_missing_fields)
        if missing:
            log.info("vlm_partial_analysis", provider=adapter.provider,
                     missing=len(missing), completeness=completeness(missing))
        return VLMResult(
            status=AnalysisStatus.DONE, analysis=analysis, attempts=attempt,
            latency_s=round(time.perf_counter() - started, 3),
            cost_usd=adapter.cost_usd(usage), input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens, provider=adapter.provider,
            model_id=adapter.model_id, completeness=completeness(missing),
            missing_fields=missing,
        )

    log.error("vlm_failed", provider=adapter.provider, attempts=max_retries + 1,
              error=last_error[:200])
    return VLMResult(
        status=AnalysisStatus.FAILED, error=last_error, attempts=max_retries + 1,
        latency_s=round(time.perf_counter() - started, 3), provider=adapter.provider,
        model_id=adapter.model_id,
    )


def price(usage: VLMUsage, in_per_mtok: float, out_per_mtok: float) -> float:
    return round(
        usage.input_tokens / 1_000_000 * in_per_mtok
        + usage.output_tokens / 1_000_000 * out_per_mtok,
        6,
    )
