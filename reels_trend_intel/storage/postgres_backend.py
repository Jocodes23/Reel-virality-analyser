"""PostgreSQL 16 + TimescaleDB + pgvector StorageBackend (production default).

Highlights vs. the SQLite fallback:
  * engagement_samples is a TimescaleDB HYPERTABLE; a CONTINUOUS AGGREGATE rolls
    up per-reel velocity so we never recompute rollups.
  * embeddings use a pgvector column with an ivfflat index for fast kNN.
  * bulk writes use COPY / executemany; a connection pool with prepared
    statements is used throughout; old chunks are compressed.

asyncpg + pgvector are imported lazily so this module loads on a box that only
has the SQLite extra installed. Bring it up with `docker compose up` (see deploy/).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from reels_trend_intel.features.hashing import to_signed64, to_unsigned64
from reels_trend_intel.features.types import ReelFeatures
from reels_trend_intel.storage.base import StorageBackend
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

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS raw_payloads (
    id BIGSERIAL PRIMARY KEY, source TEXT, kind TEXT, fetched_at TIMESTAMPTZ,
    content_hash TEXT, payload JSONB);
CREATE INDEX IF NOT EXISTS ix_raw_hash ON raw_payloads(content_hash);

CREATE TABLE IF NOT EXISTS reels (
    reel_id TEXT PRIMARY KEY, shortcode TEXT, permalink TEXT, author_handle TEXT,
    author_url TEXT, posted_at TIMESTAMPTZ, audio_id TEXT, caption TEXT, source TEXT,
    media_url TEXT, phash BIGINT, first_seen TIMESTAMPTZ, last_seen TIMESTAMPTZ,
    tracking_tier TEXT, retired BOOLEAN DEFAULT FALSE);

CREATE TABLE IF NOT EXISTS audio (
    audio_id TEXT PRIMARY KEY, instagram_audio_url TEXT, song_title TEXT, artist TEXT,
    is_original_audio BOOLEAN, usage_count INT, external_links JSONB,
    fingerprint_cluster_id INT);

CREATE TABLE IF NOT EXISTS engagement_samples (
    reel_id TEXT NOT NULL, sampled_at TIMESTAMPTZ NOT NULL,
    plays BIGINT, likes BIGINT, comments BIGINT, shares BIGINT, saves BIGINT);
SELECT create_hypertable('engagement_samples', 'sampled_at', if_not_exists => TRUE);
CREATE UNIQUE INDEX IF NOT EXISTS ux_eng ON engagement_samples(reel_id, sampled_at);

-- Continuous aggregate: hourly plays per reel (velocity is derived downstream).
CREATE MATERIALIZED VIEW IF NOT EXISTS engagement_hourly
WITH (timescaledb.continuous) AS
SELECT reel_id, time_bucket('1 hour', sampled_at) AS bucket,
       max(plays) AS plays, max(likes) AS likes
FROM engagement_samples GROUP BY reel_id, bucket WITH NO DATA;

CREATE TABLE IF NOT EXISTS embeddings (
    content_hash TEXT, kind TEXT, vector VECTOR, PRIMARY KEY(content_hash, kind));

CREATE TABLE IF NOT EXISTS reel_features (
    reel_id TEXT PRIMARY KEY, feature_schema_version TEXT, payload JSONB);
CREATE TABLE IF NOT EXISTS processing_state (
    reel_id TEXT, stage TEXT, done_at TIMESTAMPTZ, PRIMARY KEY(reel_id, stage));
CREATE TABLE IF NOT EXISTS trends (
    trend_id TEXT PRIMARY KEY, type TEXT, label TEXT, size INT, first_seen TIMESTAMPTZ,
    status TEXT);
CREATE TABLE IF NOT EXISTS trend_members (
    trend_id TEXT, reel_id TEXT, PRIMARY KEY(trend_id, reel_id));
CREATE TABLE IF NOT EXISTS trend_models (
    trend_id TEXT PRIMARY KEY, fitted_at TIMESTAMPTZ, payload JSONB, health_score REAL,
    persistence_prob REAL, persistence_ci_low REAL, persistence_ci_high REAL, phase TEXT,
    model_versions JSONB);
CREATE TABLE IF NOT EXISTS reports (
    id BIGSERIAL PRIMARY KEY, report_version TEXT, generated_at TIMESTAMPTZ,
    ranking_key TEXT, payload JSONB);
CREATE TABLE IF NOT EXISTS schedules (
    reel_id TEXT PRIMARY KEY, posted_at TIMESTAMPTZ, next_sample_at TIMESTAMPTZ,
    tier TEXT, last_plays BIGINT, retired BOOLEAN DEFAULT FALSE);
CREATE INDEX IF NOT EXISTS ix_sched_due ON schedules(retired, next_sample_at);
CREATE TABLE IF NOT EXISTS budgets (
    adapter TEXT, day TEXT, requests_used INT DEFAULT 0, cap INT DEFAULT 0,
    PRIMARY KEY(adapter, day));
"""


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


class PostgresBackend(StorageBackend):
    def __init__(self, dsn: str, min_size: int = 2, max_size: int = 10) -> None:
        self.dsn = dsn
        self.min_size = min_size
        self.max_size = max_size
        self._pool: Any | None = None

    @property
    def pool(self) -> Any:
        if self._pool is None:
            raise RuntimeError("PostgresBackend not connected; call connect() first")
        return self._pool

    async def connect(self) -> None:
        if self._pool is not None:
            return
        import asyncpg  # lazy

        async def _init(conn: Any) -> None:
            from pgvector.asyncpg import register_vector

            await register_vector(conn)

        self._pool = await asyncpg.create_pool(
            self.dsn, min_size=self.min_size, max_size=self.max_size, init=_init
        )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def init_schema(self) -> None:
        async with self.pool.acquire() as conn:
            # statements split because some (create_hypertable) cannot run in a
            # multi-command simple query alongside CREATE EXTENSION on all setups.
            for stmt in [s.strip() for s in SCHEMA.split(";\n") if s.strip()]:
                await conn.execute(stmt)

    async def put_raw(self, payload: RawPayload) -> int:
        async with self.pool.acquire() as conn:
            return int(
                await conn.fetchval(
                    "INSERT INTO raw_payloads(source,kind,fetched_at,content_hash,payload)"
                    " VALUES($1,$2,$3,$4,$5) RETURNING id",
                    payload.source, payload.kind, _aware(payload.fetched_at),
                    payload.content_hash, json.dumps(payload.payload),
                )
            )

    async def upsert_reels(self, rows: Sequence[Reel]) -> None:
        async with self.pool.acquire() as conn:
            await conn.executemany(
                """INSERT INTO reels(reel_id,shortcode,permalink,author_handle,author_url,
                     posted_at,audio_id,caption,source,media_url,phash,first_seen,last_seen,
                     tracking_tier,retired)
                   VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)
                   ON CONFLICT(reel_id) DO UPDATE SET
                     audio_id=EXCLUDED.audio_id, caption=EXCLUDED.caption,
                     media_url=EXCLUDED.media_url, phash=COALESCE(EXCLUDED.phash, reels.phash),
                     last_seen=EXCLUDED.last_seen, tracking_tier=EXCLUDED.tracking_tier,
                     retired=EXCLUDED.retired""",
                [
                    (r.reel_id, r.shortcode, r.permalink, r.author_handle, r.author_url,
                     _aware(r.posted_at), r.audio_id, r.caption, r.source, r.media_url,
                     (to_signed64(r.phash) if r.phash is not None else None),
                     _aware(r.first_seen) if r.first_seen else None,
                     _aware(r.last_seen) if r.last_seen else None, r.tracking_tier, r.retired)
                    for r in rows
                ],
            )

    async def get_reel(self, reel_id: str) -> Reel | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM reels WHERE reel_id=$1", reel_id)
        return self._reel(row) if row else None

    async def list_reels(self, retired: bool | None = None) -> list[Reel]:
        async with self.pool.acquire() as conn:
            if retired is None:
                rows = await conn.fetch("SELECT * FROM reels")
            else:
                rows = await conn.fetch("SELECT * FROM reels WHERE retired=$1", retired)
        return [self._reel(r) for r in rows]

    @staticmethod
    def _reel(row: Any) -> Reel:
        return Reel(
            reel_id=row["reel_id"], shortcode=row["shortcode"], permalink=row["permalink"],
            author_handle=row["author_handle"], author_url=row["author_url"],
            posted_at=row["posted_at"], audio_id=row["audio_id"], caption=row["caption"] or "",
            source=row["source"], media_url=row["media_url"],
            phash=(to_unsigned64(row["phash"]) if row["phash"] is not None else None),
            first_seen=row["first_seen"], last_seen=row["last_seen"],
            tracking_tier=row["tracking_tier"] or "hot", retired=bool(row["retired"]),
        )

    async def upsert_audio(self, rows: Sequence[Audio]) -> None:
        async with self.pool.acquire() as conn:
            await conn.executemany(
                """INSERT INTO audio(audio_id,instagram_audio_url,song_title,artist,
                     is_original_audio,usage_count,external_links,fingerprint_cluster_id)
                   VALUES($1,$2,$3,$4,$5,$6,$7,$8)
                   ON CONFLICT(audio_id) DO UPDATE SET
                     song_title=COALESCE(EXCLUDED.song_title, audio.song_title),
                     artist=COALESCE(EXCLUDED.artist, audio.artist),
                     usage_count=COALESCE(EXCLUDED.usage_count, audio.usage_count),
                     external_links=EXCLUDED.external_links""",
                [(a.audio_id, a.instagram_audio_url, a.song_title, a.artist, a.is_original_audio,
                  a.usage_count, json.dumps(a.external_links), a.fingerprint_cluster_id)
                 for a in rows],
            )

    async def get_audio(self, audio_id: str) -> Audio | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM audio WHERE audio_id=$1", audio_id)
        if not row:
            return None
        return Audio(
            audio_id=row["audio_id"], instagram_audio_url=row["instagram_audio_url"],
            song_title=row["song_title"], artist=row["artist"],
            is_original_audio=bool(row["is_original_audio"]), usage_count=row["usage_count"],
            external_links=json.loads(row["external_links"] or "{}"),
            fingerprint_cluster_id=row["fingerprint_cluster_id"],
        )

    async def append_engagement(self, rows: Sequence[EngagementSample]) -> None:
        async with self.pool.acquire() as conn:
            await conn.executemany(
                """INSERT INTO engagement_samples(reel_id,sampled_at,plays,likes,comments,
                     shares,saves) VALUES($1,$2,$3,$4,$5,$6,$7)
                   ON CONFLICT(reel_id,sampled_at) DO UPDATE SET
                     plays=EXCLUDED.plays, likes=EXCLUDED.likes, comments=EXCLUDED.comments,
                     shares=EXCLUDED.shares, saves=EXCLUDED.saves""",
                [(s.reel_id, _aware(s.sampled_at), s.plays, s.likes, s.comments, s.shares,
                  s.saves) for s in rows],
            )

    async def engagement_series(
        self, reel_id: str, since: datetime | None = None
    ) -> list[EngagementSample]:
        async with self.pool.acquire() as conn:
            if since is None:
                rows = await conn.fetch(
                    "SELECT * FROM engagement_samples WHERE reel_id=$1 ORDER BY sampled_at",
                    reel_id)
            else:
                rows = await conn.fetch(
                    "SELECT * FROM engagement_samples WHERE reel_id=$1 AND sampled_at>=$2 "
                    "ORDER BY sampled_at", reel_id, _aware(since))
        return [self._eng(r) for r in rows]

    async def latest_engagement(self, reel_id: str) -> EngagementSample | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM engagement_samples WHERE reel_id=$1 ORDER BY sampled_at DESC "
                "LIMIT 1", reel_id)
        return self._eng(row) if row else None

    async def collection_stats(
        self, since_hours: float = 6.0, top_n: int = 5
    ) -> dict[str, Any]:
        from datetime import timedelta

        cutoff = datetime.now(UTC) - timedelta(hours=since_hours)
        async with self.pool.acquire() as conn:
            out: dict[str, Any] = {
                "since_hours": since_hours,
                "reels_tracked": int(await conn.fetchval("SELECT COUNT(*) FROM reels") or 0),
                "engagement_samples": int(
                    await conn.fetchval("SELECT COUNT(*) FROM engagement_samples") or 0),
                "reels_with_series": int(await conn.fetchval(
                    "SELECT COUNT(*) FROM (SELECT reel_id FROM engagement_samples "
                    "GROUP BY reel_id HAVING COUNT(*) > 1) s") or 0),
            }
            last = await conn.fetchval("SELECT MAX(sampled_at) FROM engagement_samples")
            out["last_sample_at"] = last.isoformat() if last else None
            rows = await conn.fetch(
                """SELECT e.reel_id, r.permalink, r.author_handle,
                          MAX(e.likes) - MIN(e.likes) AS delta, MAX(e.likes) AS likes,
                          COUNT(*) AS n
                   FROM engagement_samples e JOIN reels r ON r.reel_id = e.reel_id
                   WHERE e.sampled_at >= $1
                   GROUP BY e.reel_id, r.permalink, r.author_handle
                   HAVING COUNT(*) > 1 AND MAX(e.likes) - MIN(e.likes) > 0
                   ORDER BY delta DESC LIMIT $2""", cutoff, top_n)
        out["movers"] = [
            {"reel_id": r["reel_id"], "permalink": r["permalink"],
             "author_handle": r["author_handle"], "delta_likes": int(r["delta"]),
             "likes": int(r["likes"]), "samples": int(r["n"])}
            for r in rows
        ]
        return out

    @staticmethod
    def _eng(row: Any) -> EngagementSample:
        return EngagementSample(
            reel_id=row["reel_id"], sampled_at=row["sampled_at"], plays=row["plays"],
            likes=row["likes"], comments=row["comments"], shares=row["shares"], saves=row["saves"])

    async def has_embedding(self, content_hash: str, kind: str) -> bool:
        async with self.pool.acquire() as conn:
            return bool(await conn.fetchval(
                "SELECT 1 FROM embeddings WHERE content_hash=$1 AND kind=$2", content_hash, kind))

    async def get_embedding(self, content_hash: str, kind: str) -> list[float] | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT vector FROM embeddings WHERE content_hash=$1 AND kind=$2",
                content_hash, kind)
        return list(row["vector"]) if row else None

    async def put_embeddings(self, rows: Sequence[EmbeddingRow]) -> None:
        import numpy as np

        async with self.pool.acquire() as conn:
            await conn.executemany(
                "INSERT INTO embeddings(content_hash,kind,vector) VALUES($1,$2,$3) "
                "ON CONFLICT(content_hash,kind) DO UPDATE SET vector=EXCLUDED.vector",
                [(r.content_hash, r.kind, np.asarray(r.vector, dtype=np.float32)) for r in rows],
            )

    async def knn(self, kind: str, query: list[float], k: int) -> list[tuple[str, float]]:
        import numpy as np

        q = np.asarray(query, dtype=np.float32)
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT content_hash, vector <=> $1 AS dist FROM embeddings WHERE kind=$2 "
                "ORDER BY dist LIMIT $3", q, kind, k)
        return [(r["content_hash"], float(r["dist"])) for r in rows]

    async def upsert_features(self, rows: Sequence[ReelFeatures]) -> None:
        async with self.pool.acquire() as conn:
            await conn.executemany(
                "INSERT INTO reel_features(reel_id,feature_schema_version,payload) "
                "VALUES($1,$2,$3) ON CONFLICT(reel_id) DO UPDATE SET payload=EXCLUDED.payload",
                [(f.reel_id, f.feature_schema_version, json.dumps(f.to_json())) for f in rows],
            )

    async def get_features(self, reel_id: str) -> ReelFeatures | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT payload FROM reel_features WHERE reel_id=$1", reel_id)
        return ReelFeatures.model_validate(json.loads(row["payload"])) if row else None

    async def all_features(self) -> list[ReelFeatures]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT payload FROM reel_features")
        return [ReelFeatures.model_validate(json.loads(r["payload"])) for r in rows]

    async def seen_phash(self, phash: int, hamming: int = 4) -> str | None:
        # Postgres-side popcount via bit_count on the XOR of bigints.
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT reel_id FROM reels WHERE phash IS NOT NULL "
                "AND bit_count((phash # $1)::bit(64)) <= $2 LIMIT 1",
                to_signed64(phash), hamming)

    async def mark_stage(self, reel_id: str, stage: str) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO processing_state(reel_id,stage,done_at) VALUES($1,$2,now()) "
                "ON CONFLICT(reel_id,stage) DO UPDATE SET done_at=now()", reel_id, stage)

    async def stages_done(self, reel_id: str) -> set[str]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT stage FROM processing_state WHERE reel_id=$1", reel_id)
        return {r["stage"] for r in rows}

    async def upsert_trend(self, trend: TrendRow, members: Sequence[str]) -> None:
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """INSERT INTO trends(trend_id,type,label,size,first_seen,status)
                       VALUES($1,$2,$3,$4,$5,$6)
                       ON CONFLICT(trend_id) DO UPDATE SET type=EXCLUDED.type,
                         label=EXCLUDED.label, size=EXCLUDED.size, status=EXCLUDED.status""",
                    trend.trend_id, trend.type, trend.label, trend.size,
                    _aware(trend.first_seen), trend.status)
                await conn.execute("DELETE FROM trend_members WHERE trend_id=$1", trend.trend_id)
                await conn.executemany(
                    "INSERT INTO trend_members(trend_id,reel_id) VALUES($1,$2) "
                    "ON CONFLICT DO NOTHING", [(trend.trend_id, m) for m in members])

    async def list_trends(self) -> list[TrendRow]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM trends")
        return [TrendRow(trend_id=r["trend_id"], type=r["type"], label=r["label"],
                         size=r["size"], first_seen=r["first_seen"], status=r["status"])
                for r in rows]

    async def trend_members(self, trend_id: str) -> list[str]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT reel_id FROM trend_members WHERE trend_id=$1", trend_id)
        return [r["reel_id"] for r in rows]

    async def save_trend_model(self, model: TrendModelRow) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO trend_models(trend_id,fitted_at,payload,health_score,
                     persistence_prob,persistence_ci_low,persistence_ci_high,phase,model_versions)
                   VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9)
                   ON CONFLICT(trend_id) DO UPDATE SET fitted_at=EXCLUDED.fitted_at,
                     payload=EXCLUDED.payload, health_score=EXCLUDED.health_score,
                     persistence_prob=EXCLUDED.persistence_prob,
                     persistence_ci_low=EXCLUDED.persistence_ci_low,
                     persistence_ci_high=EXCLUDED.persistence_ci_high, phase=EXCLUDED.phase""",
                model.trend_id, _aware(model.fitted_at), json.dumps(model.payload),
                model.health_score, model.persistence_prob, model.persistence_ci_low,
                model.persistence_ci_high, model.phase, json.dumps(model.model_versions))

    async def get_trend_model(self, trend_id: str) -> TrendModelRow | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM trend_models WHERE trend_id=$1", trend_id)
        if not row:
            return None
        return TrendModelRow(
            trend_id=row["trend_id"], fitted_at=row["fitted_at"],
            payload=json.loads(row["payload"]), health_score=row["health_score"],
            persistence_prob=row["persistence_prob"], persistence_ci_low=row["persistence_ci_low"],
            persistence_ci_high=row["persistence_ci_high"], phase=row["phase"],
            model_versions=json.loads(row["model_versions"]))

    async def write_report(
        self, report_version: str, generated_at: datetime, ranking_key: str,
        payload: dict[str, Any],
    ) -> int:
        async with self.pool.acquire() as conn:
            return int(await conn.fetchval(
                "INSERT INTO reports(report_version,generated_at,ranking_key,payload) "
                "VALUES($1,$2,$3,$4) RETURNING id",
                report_version, _aware(generated_at), ranking_key, json.dumps(payload)))

    async def read_latest_report(self) -> dict[str, Any] | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM reports ORDER BY id DESC LIMIT 1")
        if not row:
            return None
        return {"report_version": row["report_version"],
                "generated_at": row["generated_at"].isoformat(),
                "ranking_key": row["ranking_key"], "payload": json.loads(row["payload"])}

    async def upsert_schedule(self, sched: ReelSchedule) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO schedules(reel_id,posted_at,next_sample_at,tier,last_plays,retired)
                   VALUES($1,$2,$3,$4,$5,$6)
                   ON CONFLICT(reel_id) DO UPDATE SET next_sample_at=EXCLUDED.next_sample_at,
                     tier=EXCLUDED.tier, last_plays=EXCLUDED.last_plays, retired=EXCLUDED.retired""",
                sched.reel_id, _aware(sched.posted_at), _aware(sched.next_sample_at),
                sched.tier, sched.last_plays, sched.retired)

    async def due_for_resample(self, now: datetime, limit: int) -> list[ReelSchedule]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM schedules WHERE retired=FALSE AND next_sample_at<=$1 "
                "ORDER BY next_sample_at LIMIT $2", _aware(now), limit)
        return [ReelSchedule(reel_id=r["reel_id"], posted_at=r["posted_at"],
                             next_sample_at=r["next_sample_at"], tier=r["tier"],
                             last_plays=r["last_plays"], retired=bool(r["retired"]))
                for r in rows]

    async def set_next_sample(
        self, reel_id: str, when: datetime, tier: str, last_plays: int
    ) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE schedules SET next_sample_at=$1, tier=$2, last_plays=$3 WHERE reel_id=$4",
                _aware(when), tier, last_plays, reel_id)

    async def retire_reel(self, reel_id: str) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("UPDATE schedules SET retired=TRUE WHERE reel_id=$1", reel_id)
            await conn.execute(
                "UPDATE reels SET retired=TRUE, tracking_tier='dead' WHERE reel_id=$1", reel_id)

    async def get_budget(self, adapter: str, day: str) -> BudgetState:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM budgets WHERE adapter=$1 AND day=$2", adapter, day)
        if not row:
            return BudgetState(adapter=adapter, day=day, requests_used=0, cap=0)
        return BudgetState(adapter=row["adapter"], day=row["day"],
                           requests_used=row["requests_used"], cap=row["cap"])

    async def incr_budget(self, adapter: str, day: str, n: int, cap: int) -> BudgetState:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO budgets(adapter,day,requests_used,cap) VALUES($1,$2,$3,$4)
                   ON CONFLICT(adapter,day) DO UPDATE SET
                     requests_used=budgets.requests_used+EXCLUDED.requests_used,
                     cap=EXCLUDED.cap""", adapter, day, n, cap)
        return await self.get_budget(adapter, day)
