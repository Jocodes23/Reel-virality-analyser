"""Report exports.

  * JSONL  : one full TrendRecord per line (incl. member drill-down).
  * Parquet: key columns for querying + a lossless `record_json` column carrying
             the FULL record (nothing collected is dropped).
  * CSV    : one row per trend with key columns + the two clickable links
             (top_reel.url and audio.instagram_audio_url).
  * Excel  : same one-row-per-trend table when xlsxwriter/polars Excel is available.
"""

from __future__ import annotations

import json
import os

import polars as pl

from reels_trend_intel.report.schema import TrendRecord, TrendReport


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def export_jsonl(report: TrendReport, path: str) -> str:
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for rec in report.trends:
            f.write(json.dumps(rec.model_dump(mode="json"), ensure_ascii=False) + "\n")
    return path


def _row(rank: int, rec: TrendRecord) -> dict:
    d = rec.dynamics
    h = rec.headline_links
    return {
        "rank": rank,
        "trend_id": rec.identity.trend_id,
        "type": rec.identity.type,
        "label": rec.identity.label,
        "size": rec.identity.size,
        "phase": d.phase,
        "health_score": round(rec.health_score, 4),
        "persistence_probability": round(d.persistence_probability, 4),
        "persistence_ci_low": round(d.persistence_ci.low, 4),
        "persistence_ci_high": round(d.persistence_ci.high, 4),
        "r_t": round(d.r_t, 4),
        "velocity": round(d.velocity, 4),
        "acceleration": round(d.acceleration, 4),
        "opportunity_score": round(rec.opportunity_score, 4),
        "is_whitespace": rec.is_whitespace,
        "supercritical": d.supercritical,
        "forecast_peak_time": d.forecast_peak_time.isoformat() if d.forecast_peak_time else None,
        "post_before": d.post_before.isoformat() if d.post_before else None,
        "top_reel_url": h.top_reel.url,
        "top_reel_plays": h.top_reel.plays,
        "audio_instagram_url": h.audio.instagram_audio_url,
        "song_title": h.audio.song_title,
        "artist": h.audio.artist,
        "spotify": h.audio.external_links.spotify,
        "youtube": h.audio.external_links.youtube,
        "best_posting_hour": rec.creative_signal.best_posting_hour,
        "best_posting_day": rec.creative_signal.best_posting_day,
        "top_hashtags": " ".join(rec.creative_signal.hashtags),
    }


def export_parquet(report: TrendReport, path: str) -> str:
    _ensure_dir(path)
    rows = []
    for i, rec in enumerate(report.trends, 1):
        row = _row(i, rec)
        row["record_json"] = json.dumps(rec.model_dump(mode="json"), ensure_ascii=False)
        rows.append(row)
    df = pl.DataFrame(rows) if rows else pl.DataFrame()
    df.write_parquet(path)
    return path


def export_csv(report: TrendReport, path: str) -> str:
    _ensure_dir(path)
    rows = [_row(i, rec) for i, rec in enumerate(report.trends, 1)]
    df = pl.DataFrame(rows) if rows else pl.DataFrame()
    df.write_csv(path)
    return path


def export_excel(report: TrendReport, path: str) -> str | None:
    _ensure_dir(path)
    rows = [_row(i, rec) for i, rec in enumerate(report.trends, 1)]
    try:
        df = pl.DataFrame(rows) if rows else pl.DataFrame()
        df.write_excel(path)  # requires xlsxwriter
        return path
    except Exception:
        return None


def export_all(report: TrendReport, out_dir: str) -> dict[str, str | None]:
    base = os.path.join(out_dir, "trend_report")
    written: dict[str, str | None] = {
        "jsonl": export_jsonl(report, base + ".jsonl"),
        "parquet": export_parquet(report, base + ".parquet"),
        "csv": export_csv(report, base + ".csv"),
    }
    written["excel"] = export_excel(report, base + ".xlsx")
    return written
