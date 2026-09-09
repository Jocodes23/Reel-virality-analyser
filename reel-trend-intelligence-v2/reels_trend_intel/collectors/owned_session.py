"""Owned-account authenticated-session adapter (gray-area; moderate ban risk).

COMPLIANCE POSTURE (enforced):
  * Uses ONE (or very few) account(s) you personally own; session supplied via
    env RTI_OWNED_SESSION_COOKIE (never committed / never logged).
  * Strict global token bucket, randomized human-scale delays, exponential
    backoff, and a hard daily request cap are applied by the PoliteScheduler.
  * Collects ONLY public Reels (public hashtag surfaces + public media info).
  * On ANY login challenge / checkpoint it HALTS immediately and surfaces it —
    it never works around a block.

EXPLICITLY NOT IMPLEMENTED (and will not be): fingerprint spoofing, User-Agent /
device rotation, CAPTCHA solving, ban-evasion proxy rotation. A single static
web identifier is used only because the web JSON API requires it to respond; it
is never rotated to dodge detection. If blocked, we stop.
"""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime

import httpx

from reels_trend_intel.collectors.base import CollectorAdapter, CollectorHalted
from reels_trend_intel.observability.logging import get_logger
from reels_trend_intel.storage.rows import Audio, EngagementSample, Reel

log = get_logger("collectors.owned_session")

# Signals in a response that mean a login challenge / checkpoint -> HALT.
_CHALLENGE_MARKERS = ("checkpoint_required", "challenge_required", "/challenge/", "login_required")
# Public web client identifiers. Static, never rotated (rotation == evasion).
_WEB_APP_ID = "936619743392459"
_STATIC_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_SHORTCODE_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"


def shortcode_to_media_id(shortcode: str) -> int:
    """IG shortcodes are base64url of the media pk — decode locally (no request)."""
    n = 0
    for ch in shortcode:
        n = n * 64 + _SHORTCODE_ALPHABET.index(ch)
    return n


class OwnedSessionAdapter(CollectorAdapter):
    name = "owned_session"

    def __init__(
        self, halt_on_challenge: bool = True, hashtags: list[str] | None = None
    ) -> None:
        self.cookie = os.getenv("RTI_OWNED_SESSION_COOKIE")
        self.halt_on_challenge = halt_on_challenge
        self.hashtags = hashtags or ["food", "recipe"]
        self._client: httpx.AsyncClient | None = None
        self._seen: set[str] = set()

    @property
    def configured(self) -> bool:
        return bool(self.cookie)

    def _csrf(self) -> str:
        m = re.search(r"csrftoken=([^;]+)", self.cookie or "")
        return m.group(1) if m else ""

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=20.0,
                headers={
                    "Cookie": self.cookie or "",
                    "User-Agent": _STATIC_UA,      # single static UA; never rotated
                    "X-IG-App-ID": _WEB_APP_ID,    # required by the web JSON API
                    "X-CSRFToken": self._csrf(),
                    "Referer": "https://www.instagram.com/",
                    "Accept": "application/json",
                },
            )
        return self._client

    def _guard_response(self, resp: httpx.Response) -> None:
        """Detect challenge/checkpoint and HALT. Never circumvent."""
        body = resp.text[:2000].lower()
        location = resp.headers.get("location", "").lower()
        if resp.status_code in (401, 403) or any(
            m in body or m in location for m in _CHALLENGE_MARKERS
        ):
            self.halt("login challenge/checkpoint detected")
            log.error("owned_session_halt", status=resp.status_code,
                      reason="challenge/checkpoint — stopping, not routing around")
            if self.halt_on_challenge:
                raise CollectorHalted("owned_session halted on challenge/checkpoint")

    async def discover(self, limit: int) -> list[tuple[Reel, Audio | None]]:
        """Discover PUBLIC reels from the account's own hashtag surfaces."""
        if not self.configured or self.halted:
            return []
        out: list[tuple[Reel, Audio | None]] = []
        client = self._http()
        for tag in self.hashtags:
            if len(out) >= limit or self.halted:
                break
            resp = await client.get(
                "https://www.instagram.com/api/v1/tags/web_info/",
                params={"tag_name": tag},
            )
            self._guard_response(resp)
            if resp.status_code != 200:
                log.warning("owned_session_discover_status", tag=tag, status=resp.status_code)
                continue
            for media in self._iter_tag_medias(resp.json()):
                reel, audio = self._parse_media(media)
                if reel is None or reel.reel_id in self._seen:
                    continue
                self._seen.add(reel.reel_id)
                out.append((reel, audio))
                if len(out) >= limit:
                    break
        return out

    @staticmethod
    def _iter_tag_medias(payload: dict) -> list[dict]:
        """Defensively walk the (frequently-changing) tag web_info structure."""
        medias: list[dict] = []
        data = payload.get("data", payload)
        for bucket in ("top", "recent"):
            sections = (data.get(bucket, {}) or {}).get("sections", [])
            for sec in sections:
                for m in (sec.get("layout_content", {}) or {}).get("medias", []):
                    if "media" in m:
                        medias.append(m["media"])
        return medias

    def _parse_media(self, m: dict) -> tuple[Reel | None, Audio | None]:
        code = m.get("code")
        if not code or m.get("media_type") not in (2, None):  # 2 = video/reel
            return None, None
        user = m.get("user", {}) or {}
        handle = user.get("username", "unknown")
        taken = m.get("taken_at")
        posted = datetime.fromtimestamp(taken, UTC) if taken else datetime.now(UTC)
        caption = ((m.get("caption") or {}) or {}).get("text", "") if m.get("caption") else ""
        audio_id = None
        audio = None
        clips = m.get("clips_metadata") or {}
        music = (clips.get("music_info") or {}).get("music_asset_info") if clips else None
        if music:
            audio_id = str(music.get("audio_cluster_id") or music.get("id") or "")
            if audio_id:
                audio = Audio.from_id(
                    audio_id, song_title=music.get("title"),
                    artist=music.get("display_artist"),
                    is_original_audio=bool(music.get("is_original_sound")),
                )
        img = None
        cands = (m.get("image_versions2") or {}).get("candidates", [])
        if cands:
            img = cands[0].get("url")
        reel = Reel.from_shortcode(
            code, author_handle=handle,
            author_url=f"https://www.instagram.com/{handle}/",
            posted_at=posted, audio_id=audio_id, caption=caption, source=self.name,
            media_url=img,
        )
        return reel, audio

    async def fetch_engagement(
        self, reel: Reel, at: datetime | None = None
    ) -> EngagementSample | None:
        """Re-sample PUBLIC engagement for a known reel via media info."""
        if not self.configured or self.halted:
            return None
        media_id = shortcode_to_media_id(reel.shortcode)
        resp = await self._http().get(
            f"https://www.instagram.com/api/v1/media/{media_id}/info/"
        )
        self._guard_response(resp)
        if resp.status_code != 200:
            return None
        items = resp.json().get("items", [])
        if not items:
            return None
        it = items[0]
        return EngagementSample(
            reel_id=reel.reel_id, sampled_at=datetime.now(UTC),
            plays=int(it.get("play_count") or it.get("ig_play_count") or 0),
            likes=int(it.get("like_count") or 0),
            comments=int(it.get("comment_count") or 0),
            shares=int(((it.get("share_count") or {}) or {}).get("count", 0) or 0),
            saves=0,  # save counts are not exposed on public media info
        )

    async def fetch_media(self, reel: Reel) -> bytes | None:
        if not reel.media_url or self.halted:
            return None
        resp = await self._http().get(reel.media_url)
        self._guard_response(resp)
        return resp.content

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
