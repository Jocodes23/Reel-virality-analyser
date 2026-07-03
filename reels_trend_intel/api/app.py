"""Read-only FastAPI surface + a tiny dashboard.

  GET /trends?niche=&phase=&min_persistence=&limit=  -> ranked list w/ headline_links
  GET /trends/{trend_id}                             -> one full record
  GET /report                                        -> the full latest TrendReport
  GET /health                                        -> liveness + data freshness
  GET /metrics                                       -> Prometheus exposition
  GET /                                              -> HTML dashboard
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates

from reels_trend_intel.config.settings import get_settings
from reels_trend_intel.observability.metrics import METRICS, snapshot
from reels_trend_intel.storage import make_storage

_TEMPLATES = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    storage = make_storage(settings)
    await storage.connect()
    app.state.storage = storage
    app.state.settings = settings
    yield
    await storage.close()


app = FastAPI(title="reels-trend-intel", version="1.0.0", lifespan=lifespan)


async def _latest_trends(app: FastAPI) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = await app.state.storage.read_latest_report()
    if not report:
        raise HTTPException(404, "no report available yet — run the pipeline first")
    payload = report["payload"]
    return payload, payload.get("trends", [])


def _matches_niche(rec: dict[str, Any], niche: str) -> bool:
    toks = [t for t in niche.replace("/", " ").split() if t]
    hay = " ".join([
        rec["identity"]["label"], rec["identity"]["type"],
        " ".join(rec["creative_signal"].get("hashtags", [])),
        " ".join(rec["metadata"]["aggregated_features"].get("top_terms", [])),
    ]).lower()
    return any(tok.lower() in hay for tok in toks)


@app.get("/trends")
async def get_trends(
    request: Request,
    niche: str | None = Query(None),
    phase: str | None = Query(None),
    min_persistence: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(50, ge=1, le=500),
) -> JSONResponse:
    payload, trends = await _latest_trends(request.app)
    out = []
    for rec in trends:
        if phase and rec["dynamics"]["phase"] != phase:
            continue
        if rec["dynamics"]["persistence_probability"] < min_persistence:
            continue
        if niche and not _matches_niche(rec, niche):
            continue
        out.append(rec)
    return JSONResponse({
        "report_version": payload["report_version"],
        "generated_at": payload["generated_at"],
        "ranking_key": payload["ranking_key"],
        "niche": payload.get("niche"),
        "count": len(out[:limit]),
        "trends": out[:limit],
    })


@app.get("/trends/{trend_id}")
async def get_trend(request: Request, trend_id: str) -> JSONResponse:
    _, trends = await _latest_trends(request.app)
    for rec in trends:
        if rec["identity"]["trend_id"] == trend_id:
            return JSONResponse(rec)
    raise HTTPException(404, f"trend not found: {trend_id}")


@app.get("/report")
async def get_report(request: Request) -> JSONResponse:
    payload, _ = await _latest_trends(request.app)
    return JSONResponse(payload)


@app.get("/health")
async def health(request: Request) -> JSONResponse:
    report = await request.app.state.storage.read_latest_report()
    return JSONResponse({
        "status": "ok",
        "has_report": report is not None,
        "data_last_updated": report["generated_at"] if report else None,
    })


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(METRICS.render().decode(), media_type="text/plain")


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    try:
        payload, trends = await _latest_trends(request.app)
    except HTTPException:
        payload, trends = {"generated_at": None, "ranking_key": "health", "niche": None}, []
    return _TEMPLATES.TemplateResponse(request, "dashboard.html", {
        "payload": payload, "trends": trends, "metrics": snapshot(),
    })
