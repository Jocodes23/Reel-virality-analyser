"""Best-effort external link resolution for songs (Spotify / Apple / YouTube).

Offline / without provider keys this returns canonical *search* links (always
valid, clickable destinations to the track). If a real resolver API is configured
it can be swapped in. Resolution is best-effort and skipped silently on failure.
"""

from __future__ import annotations

from urllib.parse import quote_plus

from reels_trend_intel.report.schema import ExternalLinks


def resolve_external_links(
    song_title: str | None, artist: str | None, enabled: bool = True
) -> ExternalLinks:
    if not enabled or not song_title:
        return ExternalLinks()
    q = quote_plus(f"{song_title} {artist}".strip())
    return ExternalLinks(
        spotify=f"https://open.spotify.com/search/{q}",
        apple_music=f"https://music.apple.com/us/search?term={q}",
        youtube=f"https://www.youtube.com/results?search_query={q}",
    )
