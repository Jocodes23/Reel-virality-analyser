"""Assemble TrendRecords (sections A-F) and the ranked TrendReport.

The most-famous exemplar (top_reel) is chosen by a configurable metric (default
peak plays, tie-break engagement rate). The winning creative recipe is taken from
that exemplar plus aggregated member features. Member drill-down carries the full
per-reel metadata + engagement time-series — nothing collected is dropped.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta

from reels_trend_intel import FEATURE_SCHEMA_VERSION, MODEL_VERSIONS, REPORT_VERSION
from reels_trend_intel.config.settings import Settings
from reels_trend_intel.models.emerging import EmergingSignals
from reels_trend_intel.models.fusion import FusionResult
from reels_trend_intel.models.hawkes import HawkesResult
from reels_trend_intel.models.spikem import SpikeMResult
from reels_trend_intel.models.whitespace import OpportunityResult
from reels_trend_intel.report.links import resolve_external_links
from reels_trend_intel.report.ranking import sort_records
from reels_trend_intel.report.schema import (
    AudioInfo,
    ConfidenceInterval,
    CreativeSignal,
    Dynamics,
    EngagementPoint,
    HeadlineLinks,
    Identity,
    MemberReel,
    PaletteColor,
    Provenance,
    TopReel,
    TrendMetadata,
    TrendRecord,
    TrendReport,
)
from reels_trend_intel.storage.base import StorageBackend
from reels_trend_intel.storage.rows import canonical_audio_url
from reels_trend_intel.trends.types import BuiltTrend


@dataclass
class TrendModeling:
    hawkes: HawkesResult
    spikem: SpikeMResult
    emerging: EmergingSignals
    health: float
    persistence: FusionResult
    opportunity: OpportunityResult
    survival_cond: float


async def _member_reels(
    storage: StorageBackend, member_ids: list[str]
) -> list[MemberReel]:
    out: list[MemberReel] = []
    for rid in member_ids:
        reel = await storage.get_reel(rid)
        if reel is None:
            continue
        series = await storage.engagement_series(rid)
        feats = await storage.get_features(rid)
        latest = series[-1] if series else None
        out.append(MemberReel(
            reel_id=rid, permalink=reel.permalink, author_handle=reel.author_handle,
            author_url=reel.author_url, posted_at=reel.posted_at, audio_id=reel.audio_id,
            plays=latest.plays if latest else 0, likes=latest.likes if latest else 0,
            comments=latest.comments if latest else 0, shares=latest.shares if latest else 0,
            saves=latest.saves if latest else 0,
            engagement_series=[
                EngagementPoint(sampled_at=s.sampled_at, plays=s.plays, likes=s.likes,
                                comments=s.comments, shares=s.shares, saves=s.saves)
                for s in series
            ],
            color=feats.color.model_dump() if feats else {},
            audio=feats.audio.model_dump() if feats else {},
            caption=feats.caption.model_dump() if feats else {},
            fmt=feats.fmt.model_dump() if feats else {},
        ))
    return out


def _pick_top(members: list[MemberReel], metric: str) -> MemberReel:
    def er(m: MemberReel) -> float:
        return (m.likes + m.comments + m.shares + m.saves) / m.plays if m.plays else 0.0
    if metric == "engagement_rate":
        return max(members, key=lambda m: (er(m), m.plays))
    return max(members, key=lambda m: (m.plays, er(m)))  # peak_plays default


async def build_trend_record(
    storage: StorageBackend, settings: Settings, bt: BuiltTrend, mod: TrendModeling,
    now: datetime,
) -> TrendRecord:
    members = await _member_reels(storage, bt.member_ids)
    top = _pick_top(members, settings.report.top_reel_metric)
    er = (top.likes + top.comments + top.shares + top.saves) / top.plays if top.plays else 0.0
    why = (f"peak plays {top.plays:,}" if settings.report.top_reel_metric == "peak_plays"
           else f"engagement rate {er:.1%}")

    # --- A. headline links ---
    audio_id = top.audio_id
    if not audio_id and bt.member_ids:
        first = await storage.get_reel(bt.member_ids[0])
        audio_id = first.audio_id if first else None
    audio_row = await storage.get_audio(audio_id) if audio_id else None
    if audio_row:
        audio_info = AudioInfo(
            audio_id=audio_row.audio_id, instagram_audio_url=audio_row.instagram_audio_url,
            song_title=audio_row.song_title, artist=audio_row.artist,
            is_original_audio=audio_row.is_original_audio, usage_count=audio_row.usage_count,
            external_links=resolve_external_links(
                audio_row.song_title, audio_row.artist, settings.report.resolve_external_links),
        )
    else:
        audio_info = AudioInfo(audio_id=audio_id or "unknown",
                               instagram_audio_url=canonical_audio_url(audio_id or "unknown"))

    headline = HeadlineLinks(
        top_reel=TopReel(
            url=top.permalink, reel_id=top.reel_id, author_handle=top.author_handle,
            author_url=top.author_url, plays=top.plays, likes=top.likes, comments=top.comments,
            shares=top.shares, saves=top.saves, posted_at=top.posted_at, why_top=why,
        ),
        audio=audio_info,
    )

    # --- B. identity ---
    identity = Identity(
        trend_id=bt.trend_id, type=bt.type, label=bt.label, member_reel_ids=bt.member_ids,
        size=bt.size, first_seen=bt.first_seen,
    )

    # --- C. dynamics ---
    peak_time = bt.first_seen + timedelta(hours=max(mod.spikem.peak_time_h, 0.0))
    post_before = peak_time if mod.spikem.phase == "pre-peak" else now + timedelta(days=1)
    dynamics = Dynamics(
        phase=mod.spikem.phase, velocity=mod.emerging.velocity,
        acceleration=mod.emerging.acceleration, r_t=mod.hawkes.r_t,
        trend_health_score=mod.health,
        persistence_probability=mod.persistence.persistence_probability,
        persistence_ci=ConfidenceInterval(low=mod.persistence.ci_low,
                                          high=mod.persistence.ci_high),
        persistence_horizon_days=settings.models.persistence_horizon_days,
        forecast_peak_time=peak_time, post_before=post_before,
        supercritical=mod.hawkes.supercritical,
    )

    # --- D. creative signal (winning recipe = top exemplar + member aggregates) ---
    top_feats = await storage.get_features(top.reel_id)
    palette = ([PaletteColor(hex=h, proportion=p) for h, p in top_feats.color.palette]
               if top_feats else [])
    hours = Counter(m.fmt.get("hour_of_day", 0) for m in members)
    days = Counter(m.fmt.get("day_of_week", 0) for m in members)
    hashtag_counter: Counter[str] = Counter()
    for m in members:
        hashtag_counter.update(m.caption.get("hashtags", []))
    top_hashtags = [h for h, _ in hashtag_counter.most_common(8)]
    creative = CreativeSignal(
        palette=palette,
        mood=top_feats.color.key if top_feats else "mid-key",
        lighting=("low-key/warm" if top_feats and top_feats.color.warm_cool_ratio > 1
                  else "high-key/cool"),
        composition=top_feats.scene.label if top_feats else "unknown",
        warm_cool_ratio=top_feats.color.warm_cool_ratio if top_feats else 1.0,
        best_posting_hour=hours.most_common(1)[0][0] if hours else 0,
        best_posting_day=days.most_common(1)[0][0] if days else 0,
        caption_template=f"{bt.label} — " + " ".join(top_hashtags[:3]),
        hashtags=top_hashtags,
        example_reel_ids=[m.reel_id for m in sorted(members, key=lambda x: x.plays,
                                                    reverse=True)[:3]],
    )

    # --- E. metadata (aggregated + member drill-down) ---
    mean_plays = sum(m.plays for m in members) / max(1, len(members))
    mean_er = (sum((m.likes + m.comments + m.shares + m.saves) for m in members)
               / max(1, sum(m.plays for m in members) or 1))
    scene_dist = Counter(m.fmt and m.caption.get("language", "und") for m in members)
    aggregated = {
        "n_members": len(members),
        "mean_plays": round(mean_plays, 1),
        "mean_engagement_rate": round(mean_er, 4),
        "audio_id": audio_id,
        "top_hashtags": top_hashtags,
        "top_terms": bt.top_terms,
        "type": bt.type,
        "languages": dict(scene_dist),
        "adoption_curve_counts": bt.curve.counts,
        "bucket_seconds": bt.curve.bucket_s,
        "bass": {
            "p": mod.emerging.bass_p, "q": mod.emerging.bass_q, "m": mod.emerging.bass_m,
            "peak_time_h": mod.emerging.bass_peak_time_h,
        },
        "final_size_estimate": round(mod.hawkes.final_size_estimate, 1),
        "burst_onset_bucket": mod.emerging.burst_onset_bucket,
        "changepoint_bucket": mod.emerging.changepoint_bucket,
    }
    metadata = TrendMetadata(aggregated_features=aggregated, members=members)

    # --- F. provenance ---
    provenance = Provenance(
        model_versions=MODEL_VERSIONS, feature_schema_version=FEATURE_SCHEMA_VERSION,
        report_version=REPORT_VERSION, sample_size=bt.size, data_last_updated=now,
        calibration={"method": settings.models.calibration},
    )

    return TrendRecord(
        headline_links=headline, identity=identity, dynamics=dynamics,
        creative_signal=creative, metadata=metadata, provenance=provenance,
        health_score=mod.health, opportunity_score=mod.opportunity.opportunity_score,
        is_whitespace=mod.opportunity.is_whitespace,
    )


async def build_report(
    storage: StorageBackend, settings: Settings,
    items: list[tuple[BuiltTrend, TrendModeling]], now: datetime,
    calibration: dict | None = None,
) -> TrendReport:
    records = [await build_trend_record(storage, settings, bt, mod, now)
               for bt, mod in items]
    ranking_key = settings.report.default_ranking
    records = sort_records(records, ranking_key)
    provenance = Provenance(
        model_versions=MODEL_VERSIONS, feature_schema_version=FEATURE_SCHEMA_VERSION,
        report_version=REPORT_VERSION, sample_size=sum(bt.size for bt, _ in items),
        data_last_updated=now, calibration=calibration or {},
    )
    return TrendReport(
        report_version=REPORT_VERSION, generated_at=now, ranking_key=ranking_key,
        niche=settings.app.niche, provenance=provenance, trends=records,
    )
