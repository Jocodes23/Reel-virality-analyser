"""Instagram Graph API adapter (Business/Creator accounts).

Uses the official Graph API. Requires a long-lived access token + an IG Business
account id, supplied via environment (never committed):
    RTI_GRAPH_API_TOKEN, RTI_GRAPH_API_IG_USER_ID
Without credentials this adapter is inert (discover() -> []), so it is safe to
leave enabled in config. All requests flow through the polite scheduler.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import httpx

from reels_trend_intel.collectors.base import CollectorAdapter
from reels_trend_intel.observability.logging import get_logger
from reels_trend_intel.storage.rows import Audio, EngagementSample, Reel

log = get_logger("collectors.graph_api")
GRAPH = "https://graph.facebook.com/v19.0"


class GraphApiAdapter(CollectorAdapter):
    name = "graph_api"

    def __init__(self, hashtags: list[str] | None = None) -> None:
        self.token = os.getenv("RTI_GRAPH_API_TOKEN")
        self.ig_user_id = os.getenv("RTI_GRAPH_API_IG_USER_ID")
        self.hashtags = hashtags or []
        self._client: httpx.AsyncClient | None = None

    @property
    def configured(self) -> bool:
        return bool(self.token and self.ig_user_id)

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=20.0)
        return self._client

    async def discover(self, limit: int) -> list[tuple[Reel, Audio | None]]:
        if not self.configured:
            log.info("graph_api_inert", reason="no credentials configured")
            return []
        out: list[tuple[Reel, Audio | None]] = []
        client = self._http()
        for tag in self.hashtags:
            # hashtag_search -> recent_media (Business Discovery)
            r = await client.get(
                f"{GRAPH}/ig_hashtag_search",
                params={"user_id": self.ig_user_id, "q": tag, "access_token": self.token},
            )
            r.raise_for_status()
            data = r.json().get("data", [])
            if not data:
                continue
            hid = data[0]["id"]
            m = await client.get(
                f"{GRAPH}/{hid}/recent_media",
                params={
                    "user_id": self.ig_user_id,
                    "fields": "id,permalink,caption,timestamp,media_type,media_url,username",
                    "access_token": self.token,
                },
            )
            m.raise_for_status()
            for node in m.json().get("data", [])[:limit]:
                if node.get("media_type") not in ("VIDEO", "REELS"):
                    continue
                short = str(node["permalink"]).rstrip("/").split("/")[-1]
                reel = Reel.from_shortcode(
                    short, author_handle=node.get("username", "unknown"),
                    author_url=f"https://www.instagram.com/{node.get('username','')}/",
                    posted_at=datetime.fromisoformat(
                        node["timestamp"].replace("+0000", "+00:00")
                    ),
                    caption=node.get("caption", ""), source=self.name,
                    media_url=node.get("media_url"),
                )
                out.append((reel, None))
        return out[:limit]

    async def fetch_engagement(
        self, reel: Reel, at: datetime | None = None
    ) -> EngagementSample | None:
        if not self.configured:
            return None
        client = self._http()
        r = await client.get(
            f"{GRAPH}/{reel.reel_id}/insights",
            params={"metric": "plays,likes,comments,shares,saved", "access_token": self.token},
        )
        r.raise_for_status()
        vals = {d["name"]: d["values"][0]["value"] for d in r.json().get("data", [])}
        return EngagementSample(
            reel_id=reel.reel_id, sampled_at=datetime.now(UTC),
            plays=int(vals.get("plays", 0)), likes=int(vals.get("likes", 0)),
            comments=int(vals.get("comments", 0)), shares=int(vals.get("shares", 0)),
            saves=int(vals.get("saved", 0)),
        )

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
