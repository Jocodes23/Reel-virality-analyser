"""Instagram Graph API adapter (Business/Creator accounts) — supports BOTH token flavours.

Meta issues two kinds of tokens:
  * "Instagram Login" tokens (start with ``IGAA``/``IGQV``) talk to
    ``graph.instagram.com``. They can read the account's OWN media + insights and
    other public Business/Creator accounts via *Business Discovery* (by username),
    but CANNOT use hashtag search.
  * "Facebook Login" tokens talk to ``graph.facebook.com`` and additionally support
    hashtag search / recent_media.

The flavour is auto-detected from the token prefix. Discovery therefore comes from,
in order: (1) the account's own reels, (2) Business Discovery over a configurable
seed list of public niche accounts, (3) hashtag search (Facebook-login only).

Credentials come from the environment only (RTI_GRAPH_API_TOKEN,
RTI_GRAPH_API_IG_USER_ID); the token is never logged. Inert without credentials.
All requests flow through the polite scheduler.
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from typing import Any

import httpx

from reels_trend_intel.collectors.base import CollectorAdapter
from reels_trend_intel.observability.logging import get_logger
from reels_trend_intel.storage.rows import Audio, EngagementSample, Reel

log = get_logger("collectors.graph_api")
IG_BASE = "https://graph.instagram.com/v21.0"
FB_BASE = "https://graph.facebook.com/v19.0"
_MEDIA_FIELDS = ("id,caption,media_type,media_product_type,permalink,timestamp,"
                 "like_count,comments_count,thumbnail_url,media_url")
# Reels insights metrics (``views`` superseded ``plays`` in 2024; we try both).
_INSIGHT_SETS = ("views,reach,saved,shares,likes,comments",
                 "plays,reach,saved,shares,likes,comments")


def _shortcode_from_permalink(permalink: str) -> str:
    return permalink.rstrip("/").split("/")[-1]


class GraphApiAdapter(CollectorAdapter):
    name = "graph_api"

    def __init__(
        self, hashtags: list[str] | None = None, seed_usernames: list[str] | None = None
    ) -> None:
        self.token = os.getenv("RTI_GRAPH_API_TOKEN")
        self.ig_user_id = os.getenv("RTI_GRAPH_API_IG_USER_ID")
        self.hashtags = hashtags or []
        self.seed_usernames = seed_usernames or []
        self._client: httpx.AsyncClient | None = None
        self._own_username: str | None = None
        self._media_id: dict[str, str] = {}          # shortcode -> media id
        self._bd_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        self._seen: set[str] = set()
        self.bd_cache_ttl_s: float = 600.0

    # --- plumbing ----------------------------------------------------------
    @property
    def configured(self) -> bool:
        return bool(self.token and self.ig_user_id)

    @property
    def flavour(self) -> str:
        return "ig_login" if (self.token or "").startswith(("IGAA", "IGQV")) else "facebook"

    @property
    def base(self) -> str:
        return IG_BASE if self.flavour == "ig_login" else FB_BASE

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def _get(self, path: str, **params: Any) -> dict[str, Any]:
        params["access_token"] = self.token
        r = await self._http().get(f"{self.base}/{path}", params=params)
        body: dict[str, Any] = {}
        try:
            body = r.json()
        except Exception:
            pass
        if r.status_code != 200:
            err = (body.get("error") or {}).get("message", r.text[:200])
            raise RuntimeError(f"graph_api {path} -> {r.status_code}: {err}")
        return body

    async def _username(self) -> str:
        """Own IG handle. NOTE: on Facebook-Login tokens ``/me`` is the *Facebook*
        user and its ``username`` field is deprecated (error #12) — the handle must
        come from the IG Business account id instead. Failure is cached as "" so it
        can never block engagement collection for discovered reels.
        """
        if self._own_username is None:
            path = "me" if self.flavour == "ig_login" else str(self.ig_user_id)
            try:
                body = await self._get(path, fields="username")
                self._own_username = str(body.get("username") or "")
            except Exception as exc:
                log.warning("graph_api_username_unavailable", error=str(exc))
                self._own_username = ""
        return self._own_username

    # --- discovery ---------------------------------------------------------
    async def discover(self, limit: int) -> list[tuple[Reel, Audio | None]]:
        if not self.configured:
            log.info("graph_api_inert", reason="no credentials configured")
            return []
        out: list[tuple[Reel, Audio | None]] = []

        # (1) the account's own reels
        try:
            me = await self._username()
            body = await self._get(f"{self.ig_user_id}/media", fields=_MEDIA_FIELDS, limit=50)
            for m in body.get("data", []):
                self._add(out, m, me, limit)
        except Exception as exc:
            log.warning("graph_api_own_media_failed", error=str(exc))

        # (2) Business Discovery over seed niche accounts (public Business/Creator only)
        for uname in self.seed_usernames:
            if len(out) >= limit:
                break
            try:
                for m in await self._business_discovery(uname):
                    self._add(out, m, uname, limit)
            except Exception as exc:
                log.warning("graph_api_bizdisc_failed", username=uname, error=str(exc))

        # (3) hashtag search — Facebook-login tokens only
        if self.flavour == "facebook":
            for tag in self.hashtags:
                if len(out) >= limit:
                    break
                try:
                    hs = await self._get("ig_hashtag_search", user_id=self.ig_user_id, q=tag)
                    data = hs.get("data", [])
                    if not data:
                        continue
                    rm = await self._get(f"{data[0]['id']}/recent_media",
                                         user_id=self.ig_user_id,
                                         fields=_MEDIA_FIELDS + ",username")
                    for m in rm.get("data", []):
                        self._add(out, m, m.get("username", "unknown"), limit)
                except Exception as exc:
                    log.warning("graph_api_hashtag_failed", tag=tag, error=str(exc))
        return out[:limit]

    async def _business_discovery(self, uname: str) -> list[dict[str, Any]]:
        # Cached: Business Discovery returns each account's media (incl. like/comment
        # counts) in ONE call, so per-reel engagement reads are cache hits rather than
        # extra API calls. Meta rate-limits ~200 calls/hour/user, so this matters.
        now = time.monotonic()
        cached = self._bd_cache.get(uname)
        if cached and now - cached[0] < self.bd_cache_ttl_s:
            return cached[1]
        fields = (f"business_discovery.username({uname}){{username,media.limit(50)"
                  f"{{{_MEDIA_FIELDS}}}}}")
        body = await self._get(self.ig_user_id, fields=fields)
        medias = ((body.get("business_discovery") or {}).get("media") or {}).get("data", [])
        self._bd_cache[uname] = (now, medias)
        return medias

    def _add(self, out: list, m: dict[str, Any], handle: str, limit: int) -> None:
        if len(out) >= limit:
            return
        if m.get("media_product_type") not in ("REELS", None) and m.get("media_type") != "VIDEO":
            return
        permalink = m.get("permalink") or ""
        if not permalink:
            return
        code = _shortcode_from_permalink(permalink)
        if code in self._seen:
            return
        self._seen.add(code)
        self._media_id[code] = str(m.get("id", ""))
        ts = m.get("timestamp")
        posted = (datetime.fromisoformat(ts.replace("+0000", "+00:00"))
                  if ts else datetime.now(UTC))
        reel = Reel.from_shortcode(
            code, author_handle=handle, author_url=f"https://www.instagram.com/{handle}/",
            posted_at=posted, caption=m.get("caption") or "", source=self.name,
            media_url=m.get("thumbnail_url") or m.get("media_url"),
        )
        out.append((reel, None))

    # --- engagement --------------------------------------------------------
    async def fetch_engagement(
        self, reel: Reel, at: datetime | None = None
    ) -> EngagementSample | None:
        if not self.configured:
            return None
        now = datetime.now(UTC)
        media_id = self._media_id.get(reel.shortcode)
        try:
            if reel.author_handle == await self._username():
                return await self._own_engagement(reel, media_id, now)
            return await self._discovered_engagement(reel, now)
        except Exception as exc:
            log.warning("graph_api_engagement_failed", reel=reel.reel_id, error=str(exc))
            return None

    async def _own_engagement(self, reel: Reel, media_id: str | None, now: datetime
                              ) -> EngagementSample | None:
        if not media_id:
            body = await self._get(f"{self.ig_user_id}/media", fields="id,permalink", limit=50)
            for m in body.get("data", []):
                self._media_id[_shortcode_from_permalink(m.get("permalink", ""))] = str(m["id"])
            media_id = self._media_id.get(reel.shortcode)
            if not media_id:
                return None
        fields = await self._get(media_id, fields="like_count,comments_count")
        vals: dict[str, int] = {}
        for metrics in _INSIGHT_SETS:
            try:
                ins = await self._get(f"{media_id}/insights", metric=metrics)
                for d in ins.get("data", []):
                    v = (d.get("values") or [{}])[0].get("value", d.get("total_value"))
                    if isinstance(v, (int, float)):
                        vals[d["name"]] = int(v)
                break
            except RuntimeError:
                continue
        return EngagementSample(
            reel_id=reel.reel_id, sampled_at=now,
            plays=vals.get("views", vals.get("plays", 0)),
            likes=vals.get("likes", int(fields.get("like_count") or 0)),
            comments=vals.get("comments", int(fields.get("comments_count") or 0)),
            shares=vals.get("shares", 0), saves=vals.get("saved", 0),
        )

    async def _discovered_engagement(self, reel: Reel, now: datetime) -> EngagementSample | None:
        # Other accounts' media: insights unavailable; refresh public like/comment counts.
        for m in await self._business_discovery(reel.author_handle):
            if _shortcode_from_permalink(m.get("permalink", "")) == reel.shortcode:
                return EngagementSample(
                    reel_id=reel.reel_id, sampled_at=now, plays=0,
                    likes=int(m.get("like_count") or 0),
                    comments=int(m.get("comments_count") or 0), shares=0, saves=0,
                )
        return None

    async def fetch_media(self, reel: Reel) -> bytes | None:
        if not reel.media_url:
            return None
        r = await self._http().get(reel.media_url)
        r.raise_for_status()
        return r.content

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
