"""The swappable StorageBackend interface.

Two implementations ship: SQLite + sqlite-vec (default, zero-infra, offline) and
PostgreSQL 16 + TimescaleDB + pgvector (production). Raw payloads are landed
verbatim before any transform so the whole pipeline is replayable. All writes
are idempotent upserts; dedupe is by reel_id and perceptual hash.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from reels_trend_intel.features.types import ReelFeatures

from reels_trend_intel.storage.rows import (
    Audio,
    BudgetState,
    EmbeddingRow,
    EngagementSample,
    RawPayload,
    Reel,
    ReelSchedule,
    TrendModelRow,
    TrendRow,
)


class StorageBackend(ABC):
    # --- lifecycle ---------------------------------------------------------
    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    async def init_schema(self) -> None: ...

    # --- raw, replayable sink ---------------------------------------------
    @abstractmethod
    async def put_raw(self, payload: RawPayload) -> int: ...

    # --- reel + audio dimensions (idempotent bulk upserts) ----------------
    @abstractmethod
    async def upsert_reels(self, rows: Sequence[Reel]) -> None: ...

    @abstractmethod
    async def get_reel(self, reel_id: str) -> Reel | None: ...

    @abstractmethod
    async def list_reels(self, retired: bool | None = None) -> list[Reel]: ...

    @abstractmethod
    async def upsert_audio(self, rows: Sequence[Audio]) -> None: ...

    @abstractmethod
    async def get_audio(self, audio_id: str) -> Audio | None: ...

    # --- engagement time series -------------------------------------------
    @abstractmethod
    async def append_engagement(self, rows: Sequence[EngagementSample]) -> None: ...

    @abstractmethod
    async def engagement_series(
        self, reel_id: str, since: datetime | None = None
    ) -> list[EngagementSample]: ...

    @abstractmethod
    async def latest_engagement(self, reel_id: str) -> EngagementSample | None: ...

    @abstractmethod
    async def collection_stats(
        self, since_hours: float = 6.0, top_n: int = 5
    ) -> dict[str, Any]:
        """Live collection telemetry: volume, freshness and biggest recent movers.

        Trend-level dynamics are driven by posting times and so move slowly; this
        exposes what is genuinely changing minute-to-minute (engagement accrual)
        so "is the collector actually live?" is answerable at a glance.
        """

    # --- embedding cache (keyed by content hash; never recompute) ----------
    @abstractmethod
    async def has_embedding(self, content_hash: str, kind: str) -> bool: ...

    @abstractmethod
    async def get_embedding(self, content_hash: str, kind: str) -> list[float] | None: ...

    @abstractmethod
    async def put_embeddings(self, rows: Sequence[EmbeddingRow]) -> None: ...

    @abstractmethod
    async def knn(
        self, kind: str, query: list[float], k: int
    ) -> list[tuple[str, float]]: ...

    # --- features ----------------------------------------------------------
    @abstractmethod
    async def upsert_features(self, rows: Sequence[ReelFeatures]) -> None: ...

    @abstractmethod
    async def get_features(self, reel_id: str) -> ReelFeatures | None: ...

    @abstractmethod
    async def all_features(self) -> list[ReelFeatures]: ...

    @abstractmethod
    async def seen_phash(self, phash: int, hamming: int = 4) -> str | None: ...

    # --- gating / processing checkpoints ----------------------------------
    @abstractmethod
    async def mark_stage(self, reel_id: str, stage: str) -> None: ...

    @abstractmethod
    async def stages_done(self, reel_id: str) -> set[str]: ...

    # --- trends + fitted models -------------------------------------------
    @abstractmethod
    async def upsert_trend(self, trend: TrendRow, members: Sequence[str]) -> None: ...

    @abstractmethod
    async def list_trends(self) -> list[TrendRow]: ...

    @abstractmethod
    async def trend_members(self, trend_id: str) -> list[str]: ...

    @abstractmethod
    async def save_trend_model(self, model: TrendModelRow) -> None: ...

    @abstractmethod
    async def get_trend_model(self, trend_id: str) -> TrendModelRow | None: ...

    # --- report ------------------------------------------------------------
    @abstractmethod
    async def write_report(
        self,
        report_version: str,
        generated_at: datetime,
        ranking_key: str,
        payload: dict[str, Any],
    ) -> int: ...

    @abstractmethod
    async def read_latest_report(self) -> dict[str, Any] | None: ...

    # --- adaptive resampler state (restart-safe) --------------------------
    @abstractmethod
    async def upsert_schedule(self, sched: ReelSchedule) -> None: ...

    @abstractmethod
    async def due_for_resample(self, now: datetime, limit: int) -> list[ReelSchedule]: ...

    @abstractmethod
    async def set_next_sample(
        self, reel_id: str, when: datetime, tier: str, last_plays: int
    ) -> None: ...

    @abstractmethod
    async def retire_reel(self, reel_id: str) -> None: ...

    # --- adapter budget ----------------------------------------------------
    @abstractmethod
    async def get_budget(self, adapter: str, day: str) -> BudgetState: ...

    @abstractmethod
    async def incr_budget(self, adapter: str, day: str, n: int, cap: int) -> BudgetState: ...
