"""SQLite + (optional) sqlite-vec StorageBackend — the zero-infra default.

Runs the entire pipeline offline with no server. Vector kNN uses sqlite-vec when
available, otherwise a NumPy brute-force cosine fallback (adequate at the scale
this box handles; Postgres+pgvector is the production accelerator). Raw payloads
are landed verbatim. All writes are idempotent upserts.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import aiosqlite
import numpy as np

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


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat()


def _dt(s: str | None) -> datetime | None:
    if s is None:
        return None
    return datetime.fromisoformat(s)


def _popcount(x: int) -> int:
    return bin(x & 0xFFFFFFFFFFFFFFFF).count("1")


SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_payloads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    kind TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_raw_hash ON raw_payloads(content_hash);

CREATE TABLE IF NOT EXISTS reels (
    reel_id TEXT PRIMARY KEY,
    shortcode TEXT NOT NULL,
    permalink TEXT NOT NULL,
    author_handle TEXT NOT NULL,
    author_url TEXT NOT NULL,
    posted_at TEXT NOT NULL,
    audio_id TEXT,
    caption TEXT,
    source TEXT,
    media_url TEXT,
    phash INTEGER,
    first_seen TEXT,
    last_seen TEXT,
    tracking_tier TEXT,
    retired INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS audio (
    audio_id TEXT PRIMARY KEY,
    instagram_audio_url TEXT NOT NULL,
    song_title TEXT,
    artist TEXT,
    is_original_audio INTEGER DEFAULT 0,
    usage_count INTEGER,
    external_links TEXT,
    fingerprint_cluster_id INTEGER
);

CREATE TABLE IF NOT EXISTS engagement_samples (
    reel_id TEXT NOT NULL,
    sampled_at TEXT NOT NULL,
    plays INTEGER, likes INTEGER, comments INTEGER, shares INTEGER, saves INTEGER,
    PRIMARY KEY (reel_id, sampled_at)
);
CREATE INDEX IF NOT EXISTS ix_eng_reel ON engagement_samples(reel_id, sampled_at);

CREATE TABLE IF NOT EXISTS embeddings (
    content_hash TEXT NOT NULL,
    kind TEXT NOT NULL,
    vector TEXT NOT NULL,
    PRIMARY KEY (content_hash, kind)
);

CREATE TABLE IF NOT EXISTS reel_features (
    reel_id TEXT PRIMARY KEY,
    feature_schema_version TEXT,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS processing_state (
    reel_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    done_at TEXT NOT NULL,
    PRIMARY KEY (reel_id, stage)
);

CREATE TABLE IF NOT EXISTS trends (
    trend_id TEXT PRIMARY KEY,
    type TEXT, label TEXT, size INTEGER, first_seen TEXT, status TEXT
);

CREATE TABLE IF NOT EXISTS trend_members (
    trend_id TEXT NOT NULL,
    reel_id TEXT NOT NULL,
    PRIMARY KEY (trend_id, reel_id)
);

CREATE TABLE IF NOT EXISTS trend_models (
    trend_id TEXT PRIMARY KEY,
    fitted_at TEXT,
    payload TEXT,
    health_score REAL,
    persistence_prob REAL,
    persistence_ci_low REAL,
    persistence_ci_high REAL,
    phase TEXT,
    model_versions TEXT
);

CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_version TEXT,
    generated_at TEXT,
    ranking_key TEXT,
    payload TEXT
);

CREATE TABLE IF NOT EXISTS schedules (
    reel_id TEXT PRIMARY KEY,
    posted_at TEXT,
    next_sample_at TEXT,
    tier TEXT,
    last_plays INTEGER,
    retired INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_sched_due ON schedules(retired, next_sample_at);

CREATE TABLE IF NOT EXISTS budgets (
    adapter TEXT NOT NULL,
    day TEXT NOT NULL,
    requests_used INTEGER DEFAULT 0,
    cap INTEGER DEFAULT 0,
    PRIMARY KEY (adapter, day)
);
"""


class SQLiteBackend(StorageBackend):
    def __init__(self, path: str) -> None:
        self.path = path
        self._db: aiosqlite.Connection | None = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("SQLiteBackend not connected; call connect() first")
        return self._db

    async def connect(self) -> None:
        if self._db is not None:
            return
        if self.path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._db = await aiosqlite.connect(self.path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL;")
        await self._db.execute("PRAGMA synchronous=NORMAL;")
        await self._db.execute("PRAGMA foreign_keys=ON;")

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def init_schema(self) -> None:
        await self.db.executescript(SCHEMA)
        await self.db.commit()

    # --- raw ---------------------------------------------------------------
    async def put_raw(self, payload: RawPayload) -> int:
        cur = await self.db.execute(
            "INSERT INTO raw_payloads(source,kind,fetched_at,content_hash,payload)"
            " VALUES(?,?,?,?,?)",
            (
                payload.source,
                payload.kind,
                _iso(payload.fetched_at),
                payload.content_hash,
                json.dumps(payload.payload),
            ),
        )
        await self.db.commit()
        return int(cur.lastrowid or 0)

    # --- reels / audio -----------------------------------------------------
    async def upsert_reels(self, rows: Sequence[Reel]) -> None:
        await self.db.executemany(
            """INSERT INTO reels(reel_id,shortcode,permalink,author_handle,author_url,
                 posted_at,audio_id,caption,source,media_url,phash,first_seen,last_seen,
                 tracking_tier,retired)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(reel_id) DO UPDATE SET
                 audio_id=excluded.audio_id, caption=excluded.caption,
                 media_url=excluded.media_url, phash=COALESCE(excluded.phash, reels.phash),
                 last_seen=excluded.last_seen, tracking_tier=excluded.tracking_tier,
                 retired=excluded.retired""",
            [
                (
                    r.reel_id, r.shortcode, r.permalink, r.author_handle, r.author_url,
                    _iso(r.posted_at), r.audio_id, r.caption, r.source, r.media_url,
                    (to_signed64(r.phash) if r.phash is not None else None),
                    _iso(r.first_seen) if r.first_seen else None,
                    _iso(r.last_seen) if r.last_seen else None, r.tracking_tier,
                    int(r.retired),
                )
                for r in rows
            ],
        )
        await self.db.commit()

    async def get_reel(self, reel_id: str) -> Reel | None:
        async with self.db.execute("SELECT * FROM reels WHERE reel_id=?", (reel_id,)) as cur:
            row = await cur.fetchone()
        return self._reel(row) if row else None

    async def list_reels(self, retired: bool | None = None) -> list[Reel]:
        q: str
        args: tuple[Any, ...]
        if retired is None:
            q, args = "SELECT * FROM reels", ()
        else:
            q, args = "SELECT * FROM reels WHERE retired=?", (int(retired),)
        async with self.db.execute(q, args) as cur:
            rows = await cur.fetchall()
        return [self._reel(r) for r in rows]

    @staticmethod
    def _reel(row: aiosqlite.Row) -> Reel:
        return Reel(
            reel_id=row["reel_id"], shortcode=row["shortcode"], permalink=row["permalink"],
            author_handle=row["author_handle"], author_url=row["author_url"],
            posted_at=_dt(row["posted_at"]),
            audio_id=row["audio_id"], caption=row["caption"] or "", source=row["source"],
            media_url=row["media_url"],
            phash=(to_unsigned64(row["phash"]) if row["phash"] is not None else None),
            first_seen=_dt(row["first_seen"]), last_seen=_dt(row["last_seen"]),
            tracking_tier=row["tracking_tier"] or "hot", retired=bool(row["retired"]),
        )

    async def upsert_audio(self, rows: Sequence[Audio]) -> None:
        await self.db.executemany(
            """INSERT INTO audio(audio_id,instagram_audio_url,song_title,artist,
                 is_original_audio,usage_count,external_links,fingerprint_cluster_id)
               VALUES(?,?,?,?,?,?,?,?)
               ON CONFLICT(audio_id) DO UPDATE SET
                 song_title=COALESCE(excluded.song_title, audio.song_title),
                 artist=COALESCE(excluded.artist, audio.artist),
                 usage_count=COALESCE(excluded.usage_count, audio.usage_count),
                 external_links=excluded.external_links,
                 fingerprint_cluster_id=COALESCE(excluded.fingerprint_cluster_id,
                                                 audio.fingerprint_cluster_id)""",
            [
                (
                    a.audio_id, a.instagram_audio_url, a.song_title, a.artist,
                    int(a.is_original_audio), a.usage_count,
                    json.dumps(a.external_links), a.fingerprint_cluster_id,
                )
                for a in rows
            ],
        )
        await self.db.commit()

    async def get_audio(self, audio_id: str) -> Audio | None:
        async with self.db.execute("SELECT * FROM audio WHERE audio_id=?", (audio_id,)) as cur:
            row = await cur.fetchone()
        if not row:
            return None
        return Audio(
            audio_id=row["audio_id"], instagram_audio_url=row["instagram_audio_url"],
            song_title=row["song_title"], artist=row["artist"],
            is_original_audio=bool(row["is_original_audio"]), usage_count=row["usage_count"],
            external_links=json.loads(row["external_links"] or "{}"),
            fingerprint_cluster_id=row["fingerprint_cluster_id"],
        )

    # --- engagement --------------------------------------------------------
    async def append_engagement(self, rows: Sequence[EngagementSample]) -> None:
        await self.db.executemany(
            """INSERT INTO engagement_samples(reel_id,sampled_at,plays,likes,comments,shares,saves)
               VALUES(?,?,?,?,?,?,?)
               ON CONFLICT(reel_id,sampled_at) DO UPDATE SET
                 plays=excluded.plays, likes=excluded.likes, comments=excluded.comments,
                 shares=excluded.shares, saves=excluded.saves""",
            [
                (s.reel_id, _iso(s.sampled_at), s.plays, s.likes, s.comments, s.shares, s.saves)
                for s in rows
            ],
        )
        await self.db.commit()

    async def engagement_series(
        self, reel_id: str, since: datetime | None = None
    ) -> list[EngagementSample]:
        q: str
        args: tuple[Any, ...]
        if since is None:
            q, args = ("SELECT * FROM engagement_samples WHERE reel_id=? ORDER BY sampled_at",
                       (reel_id,))
        else:
            q, args = ("SELECT * FROM engagement_samples WHERE reel_id=? AND sampled_at>=? "
                       "ORDER BY sampled_at", (reel_id, _iso(since)))
        async with self.db.execute(q, args) as cur:
            rows = await cur.fetchall()
        return [self._eng(r) for r in rows]

    async def latest_engagement(self, reel_id: str) -> EngagementSample | None:
        async with self.db.execute(
            "SELECT * FROM engagement_samples WHERE reel_id=? ORDER BY sampled_at DESC LIMIT 1",
            (reel_id,),
        ) as cur:
            row = await cur.fetchone()
        return self._eng(row) if row else None

    async def collection_stats(
        self, since_hours: float = 6.0, top_n: int = 5
    ) -> dict[str, Any]:
        cutoff = _iso(datetime.now(UTC) - timedelta(hours=since_hours))
        out: dict[str, Any] = {"since_hours": since_hours}
        for key, sql in (
            ("reels_tracked", "SELECT COUNT(*) FROM reels"),
            ("engagement_samples", "SELECT COUNT(*) FROM engagement_samples"),
            ("reels_with_series",
             "SELECT COUNT(*) FROM (SELECT reel_id FROM engagement_samples "
             "GROUP BY reel_id HAVING COUNT(*) > 1)"),
        ):
            async with self.db.execute(sql) as cur:
                row = await cur.fetchone()
            out[key] = int(row[0]) if row else 0
        async with self.db.execute(
            "SELECT MAX(sampled_at) FROM engagement_samples"
        ) as cur:
            row = await cur.fetchone()
        out["last_sample_at"] = row[0] if row else None
        async with self.db.execute(
            """SELECT e.reel_id, r.permalink, r.author_handle,
                      MAX(e.likes) - MIN(e.likes) AS delta, MAX(e.likes) AS likes,
                      COUNT(*) AS n
               FROM engagement_samples e JOIN reels r ON r.reel_id = e.reel_id
               WHERE e.sampled_at >= ?
               GROUP BY e.reel_id
               HAVING COUNT(*) > 1 AND delta > 0
               ORDER BY delta DESC LIMIT ?""",
            (cutoff, top_n),
        ) as cur:
            rows = await cur.fetchall()
        out["movers"] = [
            {"reel_id": r["reel_id"], "permalink": r["permalink"],
             "author_handle": r["author_handle"], "delta_likes": int(r["delta"]),
             "likes": int(r["likes"]), "samples": int(r["n"])}
            for r in rows
        ]
        return out

    @staticmethod
    def _eng(row: aiosqlite.Row) -> EngagementSample:
        return EngagementSample(
            reel_id=row["reel_id"], sampled_at=_dt(row["sampled_at"]),
            plays=row["plays"], likes=row["likes"], comments=row["comments"],
            shares=row["shares"], saves=row["saves"],
        )

    # --- embeddings --------------------------------------------------------
    async def has_embedding(self, content_hash: str, kind: str) -> bool:
        async with self.db.execute(
            "SELECT 1 FROM embeddings WHERE content_hash=? AND kind=?", (content_hash, kind)
        ) as cur:
            return await cur.fetchone() is not None

    async def get_embedding(self, content_hash: str, kind: str) -> list[float] | None:
        async with self.db.execute(
            "SELECT vector FROM embeddings WHERE content_hash=? AND kind=?", (content_hash, kind)
        ) as cur:
            row = await cur.fetchone()
        return json.loads(row["vector"]) if row else None

    async def put_embeddings(self, rows: Sequence[EmbeddingRow]) -> None:
        await self.db.executemany(
            "INSERT OR REPLACE INTO embeddings(content_hash,kind,vector) VALUES(?,?,?)",
            [(r.content_hash, r.kind, json.dumps([round(float(x), 6) for x in r.vector]))
             for r in rows],
        )
        await self.db.commit()

    async def knn(self, kind: str, query: list[float], k: int) -> list[tuple[str, float]]:
        async with self.db.execute(
            "SELECT content_hash, vector FROM embeddings WHERE kind=?", (kind,)
        ) as cur:
            rows = await cur.fetchall()
        if not rows:
            return []
        q = np.asarray(query, dtype=np.float32)
        qn = q / (np.linalg.norm(q) + 1e-9)
        scored: list[tuple[str, float]] = []
        for r in rows:
            v = np.asarray(json.loads(r["vector"]), dtype=np.float32)
            sim = float(np.dot(qn, v / (np.linalg.norm(v) + 1e-9)))
            scored.append((r["content_hash"], 1.0 - sim))  # cosine distance
        scored.sort(key=lambda t: t[1])
        return scored[:k]

    # --- features ----------------------------------------------------------
    async def upsert_features(self, rows: Sequence[ReelFeatures]) -> None:
        await self.db.executemany(
            "INSERT OR REPLACE INTO reel_features(reel_id,feature_schema_version,payload) "
            "VALUES(?,?,?)",
            [(f.reel_id, f.feature_schema_version, json.dumps(f.to_json())) for f in rows],
        )
        await self.db.commit()

    async def get_features(self, reel_id: str) -> ReelFeatures | None:
        async with self.db.execute(
            "SELECT payload FROM reel_features WHERE reel_id=?", (reel_id,)
        ) as cur:
            row = await cur.fetchone()
        return ReelFeatures.model_validate(json.loads(row["payload"])) if row else None

    async def all_features(self) -> list[ReelFeatures]:
        async with self.db.execute("SELECT payload FROM reel_features") as cur:
            rows = await cur.fetchall()
        return [ReelFeatures.model_validate(json.loads(r["payload"])) for r in rows]

    async def seen_phash(self, phash: int, hamming: int = 4) -> str | None:
        async with self.db.execute(
            "SELECT reel_id, phash FROM reels WHERE phash IS NOT NULL"
        ) as cur:
            rows = await cur.fetchall()
        for r in rows:
            if _popcount(to_unsigned64(int(r["phash"])) ^ phash) <= hamming:
                return str(r["reel_id"])
        return None

    # --- gating ------------------------------------------------------------
    async def mark_stage(self, reel_id: str, stage: str) -> None:
        await self.db.execute(
            "INSERT OR REPLACE INTO processing_state(reel_id,stage,done_at) VALUES(?,?,?)",
            (reel_id, stage, _iso(datetime.now(UTC))),
        )
        await self.db.commit()

    async def stages_done(self, reel_id: str) -> set[str]:
        async with self.db.execute(
            "SELECT stage FROM processing_state WHERE reel_id=?", (reel_id,)
        ) as cur:
            rows = await cur.fetchall()
        return {r["stage"] for r in rows}

    # --- trends ------------------------------------------------------------
    async def upsert_trend(self, trend: TrendRow, members: Sequence[str]) -> None:
        await self.db.execute(
            """INSERT INTO trends(trend_id,type,label,size,first_seen,status)
               VALUES(?,?,?,?,?,?)
               ON CONFLICT(trend_id) DO UPDATE SET
                 type=excluded.type, label=excluded.label, size=excluded.size,
                 status=excluded.status""",
            (trend.trend_id, trend.type, trend.label, trend.size,
             _iso(trend.first_seen), trend.status),
        )
        await self.db.execute("DELETE FROM trend_members WHERE trend_id=?", (trend.trend_id,))
        await self.db.executemany(
            "INSERT OR IGNORE INTO trend_members(trend_id,reel_id) VALUES(?,?)",
            [(trend.trend_id, m) for m in members],
        )
        await self.db.commit()

    async def list_trends(self) -> list[TrendRow]:
        async with self.db.execute("SELECT * FROM trends") as cur:
            rows = await cur.fetchall()
        return [
            TrendRow(
                trend_id=r["trend_id"], type=r["type"], label=r["label"], size=r["size"],
                first_seen=_dt(r["first_seen"]),
                status=r["status"],
            )
            for r in rows
        ]

    async def trend_members(self, trend_id: str) -> list[str]:
        async with self.db.execute(
            "SELECT reel_id FROM trend_members WHERE trend_id=?", (trend_id,)
        ) as cur:
            rows = await cur.fetchall()
        return [r["reel_id"] for r in rows]

    async def save_trend_model(self, model: TrendModelRow) -> None:
        await self.db.execute(
            """INSERT OR REPLACE INTO trend_models(trend_id,fitted_at,payload,health_score,
                 persistence_prob,persistence_ci_low,persistence_ci_high,phase,model_versions)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (
                model.trend_id, _iso(model.fitted_at), json.dumps(model.payload),
                model.health_score, model.persistence_prob, model.persistence_ci_low,
                model.persistence_ci_high, model.phase, json.dumps(model.model_versions),
            ),
        )
        await self.db.commit()

    async def get_trend_model(self, trend_id: str) -> TrendModelRow | None:
        async with self.db.execute(
            "SELECT * FROM trend_models WHERE trend_id=?", (trend_id,)
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return None
        return TrendModelRow(
            trend_id=row["trend_id"], fitted_at=_dt(row["fitted_at"]),
            payload=json.loads(row["payload"]), health_score=row["health_score"],
            persistence_prob=row["persistence_prob"], persistence_ci_low=row["persistence_ci_low"],
            persistence_ci_high=row["persistence_ci_high"], phase=row["phase"],
            model_versions=json.loads(row["model_versions"]),
        )

    # --- report ------------------------------------------------------------
    async def write_report(
        self, report_version: str, generated_at: datetime, ranking_key: str,
        payload: dict[str, Any],
    ) -> int:
        cur = await self.db.execute(
            "INSERT INTO reports(report_version,generated_at,ranking_key,payload) VALUES(?,?,?,?)",
            (report_version, _iso(generated_at), ranking_key, json.dumps(payload)),
        )
        await self.db.commit()
        return int(cur.lastrowid or 0)

    async def read_latest_report(self) -> dict[str, Any] | None:
        async with self.db.execute(
            "SELECT * FROM reports ORDER BY id DESC LIMIT 1"
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return None
        return {
            "report_version": row["report_version"],
            "generated_at": row["generated_at"],
            "ranking_key": row["ranking_key"],
            "payload": json.loads(row["payload"]),
        }

    # --- schedules ---------------------------------------------------------
    async def upsert_schedule(self, sched: ReelSchedule) -> None:
        await self.db.execute(
            """INSERT INTO schedules(reel_id,posted_at,next_sample_at,tier,last_plays,retired)
               VALUES(?,?,?,?,?,?)
               ON CONFLICT(reel_id) DO UPDATE SET
                 next_sample_at=excluded.next_sample_at, tier=excluded.tier,
                 last_plays=excluded.last_plays, retired=excluded.retired""",
            (sched.reel_id, _iso(sched.posted_at), _iso(sched.next_sample_at),
             sched.tier, sched.last_plays, int(sched.retired)),
        )
        await self.db.commit()

    async def due_for_resample(self, now: datetime, limit: int) -> list[ReelSchedule]:
        async with self.db.execute(
            "SELECT * FROM schedules WHERE retired=0 AND next_sample_at<=? "
            "ORDER BY next_sample_at LIMIT ?",
            (_iso(now), limit),
        ) as cur:
            rows = await cur.fetchall()
        return [
            ReelSchedule(
                reel_id=r["reel_id"], posted_at=_dt(r["posted_at"]),
                next_sample_at=_dt(r["next_sample_at"]),
                tier=r["tier"], last_plays=r["last_plays"], retired=bool(r["retired"]),
            )
            for r in rows
        ]

    async def set_next_sample(
        self, reel_id: str, when: datetime, tier: str, last_plays: int
    ) -> None:
        await self.db.execute(
            "UPDATE schedules SET next_sample_at=?, tier=?, last_plays=? WHERE reel_id=?",
            (_iso(when), tier, last_plays, reel_id),
        )
        await self.db.commit()

    async def retire_reel(self, reel_id: str) -> None:
        await self.db.execute("UPDATE schedules SET retired=1 WHERE reel_id=?", (reel_id,))
        await self.db.execute("UPDATE reels SET retired=1, tracking_tier='dead' WHERE reel_id=?",
                              (reel_id,))
        await self.db.commit()

    # --- budget ------------------------------------------------------------
    async def get_budget(self, adapter: str, day: str) -> BudgetState:
        async with self.db.execute(
            "SELECT * FROM budgets WHERE adapter=? AND day=?", (adapter, day)
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return BudgetState(adapter=adapter, day=day, requests_used=0, cap=0)
        return BudgetState(adapter=row["adapter"], day=row["day"],
                           requests_used=row["requests_used"], cap=row["cap"])

    async def incr_budget(self, adapter: str, day: str, n: int, cap: int) -> BudgetState:
        await self.db.execute(
            """INSERT INTO budgets(adapter,day,requests_used,cap) VALUES(?,?,?,?)
               ON CONFLICT(adapter,day) DO UPDATE SET
                 requests_used=budgets.requests_used+excluded.requests_used,
                 cap=excluded.cap""",
            (adapter, day, n, cap),
        )
        await self.db.commit()
        return await self.get_budget(adapter, day)
