"""Licensed third-party dataset/provider adapter.

Generic client for a *licensed* data provider that already handles compliant
collection and exposes reels + engagement via a REST API. Configure via env:
    RTI_PROVIDER_BASE_URL, RTI_PROVIDER_API_KEY
Inert without configuration. This is the lowest-risk source: you pay a provider
that holds the licence, so there is no scraping on our side.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import httpx

from reels_trend_intel.collectors.base import CollectorAdapter
from reels_trend_intel.observability.logging import get_logger
from reels_trend_intel.storage.rows import Audio, EngagementSample, Reel

log = get_logger("collectors.licensed_provider")


class LicensedProviderAdapter(CollectorAdapter):
    name = "licensed_provider"

    def __init__(self, query: str | None = None) -> None:
        self.base = os.getenv("RTI_PROVIDER_BASE_URL")
        self.key = os.getenv("RTI_PROVIDER_API_KEY")
        self.query = query or "reels"
        self._client: httpx.AsyncClient | None = None

    @property
    def configured(self) -> bool:
        return bool(self.base and self.key)

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=20.0, headers={"Authorization": f"Bearer {self.key}"}
            )
        return self._client

    async def discover(self, limit: int) -> list[tuple[Reel, Audio | None]]:
        if not self.configured:
            log.info("provider_inert", reason="no credentials configured")
            return []
        r = await self._http().get(
            f"{self.base}/reels", params={"q": self.query, "limit": limit}
        )
        r.raise_for_status()
        out: list[tuple[Reel, Audio | None]] = []
        for node in r.json().get("items", []):
            reel = Reel.from_shortcode(
                node["shortcode"], author_handle=node.get("author_handle", "unknown"),
                author_url=node.get("author_url", ""),
                posted_at=datetime.fromisoformat(node["posted_at"]),
                audio_id=node.get("audio_id"), caption=node.get("caption", ""),
                source=self.name, media_url=node.get("media_url"),
            )
            audio = None
            if node.get("audio_id"):
                audio = Audio.from_id(
                    node["audio_id"], song_title=node.get("song_title"),
                    artist=node.get("artist"), is_original_audio=node.get("is_original", False),
                    usage_count=node.get("usage_count"),
                )
            out.append((reel, audio))
        return out

    async def fetch_engagement(
        self, reel: Reel, at: datetime | None = None
    ) -> EngagementSample | None:
        if not self.configured:
            return None
        r = await self._http().get(f"{self.base}/reels/{reel.reel_id}/metrics")
        r.raise_for_status()
        m = r.json()
        return EngagementSample(
            reel_id=reel.reel_id, sampled_at=datetime.now(UTC),
            plays=int(m.get("plays", 0)), likes=int(m.get("likes", 0)),
            comments=int(m.get("comments", 0)), shares=int(m.get("shares", 0)),
            saves=int(m.get("saves", 0)),
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
