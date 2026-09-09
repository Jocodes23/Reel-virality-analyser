from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from reels_trend_intel.config.settings import FeatureConfig
from reels_trend_intel.features.audio import extract_audio_offline
from reels_trend_intel.features.caption import extract_caption
from reels_trend_intel.features.color import extract_color
from reels_trend_intel.features.embeddings import DeterministicBackend
from reels_trend_intel.features.format import extract_format
from reels_trend_intel.features.hashing import (
    hamming,
    perceptual_hash,
    to_signed64,
    to_unsigned64,
)
from tests.conftest import make_image


def test_color_extracts_palette(image_bytes):
    cf = extract_color(image_bytes, k=5)
    assert len(cf.palette) >= 1
    assert cf.palette[0][0].startswith("#") and len(cf.palette[0][0]) == 7
    assert 0.0 <= cf.luminance <= 1.0 and 0.0 <= cf.saturation <= 1.0
    assert cf.key in {"high-key", "low-key", "mid-key"}
    # warm image -> warm/cool ratio > 1
    assert cf.warm_cool_ratio > 0


def test_color_high_vs_low_key():
    dark = extract_color(make_image((20, 20, 25)))
    light = extract_color(make_image((235, 235, 230)))
    assert dark.luminance < light.luminance
    assert dark.key == "low-key" and light.key == "high-key"


def _structured(kind: str) -> bytes:
    import io

    from PIL import Image

    arr = np.zeros((64, 64, 3), np.uint8)
    if kind == "left":
        arr[:, :32] = 220
    else:  # "top"
        arr[:32, :] = 220
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def test_phash_similar_vs_different():
    # pHash encodes STRUCTURE: identical media -> identical hash; different spatial
    # layout -> differing bits. (Uniform images collapse to a degenerate hash.)
    a = perceptual_hash(_structured("left"))
    a2 = perceptual_hash(_structured("left"))
    b = perceptual_hash(_structured("top"))
    assert a == a2                       # deterministic / dedupe-able
    assert hamming(a, b) > 0             # different structure -> different hash


def test_signed64_roundtrip():
    for u in [0, 1, 2**63 - 1, 2**63, 2**64 - 1]:
        assert to_unsigned64(to_signed64(u)) == u


def test_caption_features():
    cf = extract_caption("🔥🔥 best smash burger — save this & follow #food #burger")
    assert "#food" in cf.hashtags and "#burger" in cf.hashtags
    assert cf.has_cta is True
    assert cf.emoji_density > 0
    assert cf.length > 0


def test_audio_offline_same_song_same_fingerprint():
    a = extract_audio_offline(None, "aud_x")
    b = extract_audio_offline(None, "aud_x")
    c = extract_audio_offline(None, "aud_y")
    assert a.fingerprint == b.fingerprint
    assert a.fingerprint != c.fingerprint
    assert a.tempo is not None


def test_format_timing():
    dt = datetime(2026, 6, 22, 14, 30, tzinfo=UTC)  # Monday
    ff = extract_format(dt, {"duration_s": 12.0, "cut_rate": 2.0, "has_face": True})
    assert ff.hour_of_day == 14 and ff.day_of_week == 0
    assert ff.duration_s == 12.0 and ff.has_face is True


def test_deterministic_embeddings_similarity():
    cfg = FeatureConfig()
    be = DeterministicBackend(cfg)
    imgs = [make_image((200, 100, 50), seed=1), make_image((200, 100, 50), seed=2),
            make_image((20, 40, 200), seed=3)]
    E = be.embed_images(imgs)
    # same-palette images are more similar than the different one
    sim_same = float(E[0] @ E[1])
    sim_diff = float(E[0] @ E[2])
    assert sim_same > sim_diff
    caps = be.embed_captions(["#food burger", "#food burger yum", "#travel beach"])
    assert caps.shape[0] == 3
