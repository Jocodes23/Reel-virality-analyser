"""Assemble trends from joint-embedding clusters.

For each cluster: label via c-TF-IDF, classify the trend type, build the adoption
curve N(t) (new reels per bucket) + aggregate plays, derive the Hawkes event
stream, and persist trend + members. Returns BuiltTrend objects for the models.
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import UTC, datetime

from reels_trend_intel.config.settings import Settings
from reels_trend_intel.observability.logging import get_logger
from reels_trend_intel.storage.base import StorageBackend
from reels_trend_intel.storage.rows import Reel, TrendRow
from reels_trend_intel.trends.cluster import assemble_joint, reduce_and_cluster
from reels_trend_intel.trends.topics import label_clusters
from reels_trend_intel.trends.types import AdoptionCurve, BuiltTrend

log = get_logger("trends.build")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:32] or "trend"


def _classify_type(face_share: float, cut_rate: float, textsolid_share: float,
                   original_share: float) -> str:
    if cut_rate >= 2.0 or face_share >= 0.7:
        return "format"
    if textsolid_share >= 0.5:
        return "caption"
    if original_share >= 0.6:
        return "audio"
    return "visual"


async def build_trends(storage: StorageBackend, settings: Settings) -> list[BuiltTrend]:
    feats = await storage.all_features()
    if not feats:
        return []
    feat_by_id = {f.reel_id: f for f in feats}
    ids, matrix = await assemble_joint(storage, feats, settings.trends)
    labels = reduce_and_cluster(matrix, settings.trends, settings.app.seed)
    terms = label_clusters(feats, labels)

    members_by_cluster: dict[int, list[str]] = defaultdict(list)
    for rid, lbl in zip(ids, labels, strict=False):
        if int(lbl) >= 0:
            members_by_cluster[int(lbl)].append(rid)

    bucket_s = settings.trends.bucket_s
    built: list[BuiltTrend] = []
    used_ids: set[str] = set()
    for cluster, member_ids in sorted(members_by_cluster.items()):
        fetched = [await storage.get_reel(m) for m in member_ids]
        reels: list[Reel] = [r for r in fetched if r is not None]
        if not reels:
            continue
        posted = sorted(r.posted_at for r in reels)
        first_seen = posted[0]

        # type signals
        cfeats = [feat_by_id[r.reel_id] for r in reels if r.reel_id in feat_by_id]
        size = len(reels)
        face_share = sum(f.fmt.has_face for f in cfeats) / max(1, len(cfeats))
        cut_rate = sum((f.fmt.cut_rate or 0.0) for f in cfeats) / max(1, len(cfeats))
        textsolid = sum(f.scene.label == "text-on-solid" for f in cfeats) / max(1, len(cfeats))
        original = sum(f.audio.is_original_audio for f in cfeats) / max(1, len(cfeats))
        ttype = _classify_type(face_share, cut_rate, textsolid, original)

        top_terms = terms.get(cluster, [])
        label = " · ".join(top_terms[:3]) if top_terms else f"cluster {cluster}"

        # adoption curve N(t)
        last_h = (posted[-1] - first_seen).total_seconds() / 3600.0
        n_buckets = max(1, int(last_h * 3600 / bucket_s) + 1)
        counts = [0] * n_buckets
        plays = [0.0] * n_buckets
        event_times_h: list[float] = []
        for r in reels:
            dt_h = (r.posted_at - first_seen).total_seconds() / 3600.0
            event_times_h.append(dt_h)
            b = min(n_buckets - 1, int(dt_h * 3600 / bucket_s))
            counts[b] += 1
            latest = await storage.latest_engagement(r.reel_id)
            plays[b] += float(latest.plays) if latest else 0.0
        event_times_h.sort()

        trend_id = f"tr_{ttype}_{_slug(label)}"
        if trend_id in used_ids:
            trend_id = f"{trend_id}-{cluster}"
        used_ids.add(trend_id)

        curve = AdoptionCurve(bucket_s=bucket_s, start=first_seen, counts=counts, plays=plays)
        bt = BuiltTrend(
            trend_id=trend_id, type=ttype, label=label, member_ids=[r.reel_id for r in reels],
            first_seen=first_seen, size=size, curve=curve, event_times_h=event_times_h,
            top_terms=top_terms,
        )
        built.append(bt)
        await storage.upsert_trend(
            TrendRow(trend_id=trend_id, type=ttype, label=label, size=size,
                     first_seen=first_seen, status="active"),
            bt.member_ids,
        )
    log.info("trends_built", n=len(built), noise=int((labels == -1).sum()))
    return built


def now_utc() -> datetime:
    return datetime.now(UTC)
