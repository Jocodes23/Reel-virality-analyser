"""Caption/text features: hashtags, length, emoji density, CTA, language, OCR.

Embedding of the caption is produced separately by the embedding backend. OCR
runs only on a sampled keyframe (cost control) and only when the OCR extra is
installed; offline synthetic frames have no text, so it returns None.
"""

from __future__ import annotations

import re

from reels_trend_intel.features.types import CaptionFeatures

_HASHTAG = re.compile(r"#\w+", re.UNICODE)
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]", re.UNICODE
)
_CTA_PATTERNS = [
    "save this", "save it", "follow for more", "follow me", "comment", "tag a",
    "link in bio", "share this", "drop a", "double tap",
]


def extract_caption(text: str) -> CaptionFeatures:
    hashtags = [h.lower() for h in _HASHTAG.findall(text)]
    emojis = _EMOJI.findall(text)
    length = len(text)
    emoji_density = len(emojis) / max(1, len(text))
    low = text.lower()
    has_cta = any(p in low for p in _CTA_PATTERNS)
    language = _detect_language(text)
    return CaptionFeatures(
        text=text, hashtags=hashtags, length=length, emoji_density=emoji_density,
        has_cta=has_cta, language=language, ocr_text=None,
    )


def _detect_language(text: str) -> str:
    try:
        from langdetect import detect  # optional

        stripped = _HASHTAG.sub("", text).strip()
        return detect(stripped) if len(stripped) > 8 else "und"
    except Exception:
        # Heuristic fallback: assume English for ASCII-dominant captions.
        ascii_ratio = sum(c.isascii() for c in text) / max(1, len(text))
        return "en" if ascii_ratio > 0.8 else "und"


def run_ocr(image_bytes: bytes) -> str | None:  # pragma: no cover - requires rapidocr
    try:
        from rapidocr_onnxruntime import RapidOCR

        ocr = RapidOCR()
        import io

        import numpy as np
        from PIL import Image

        arr = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        result, _ = ocr(arr)
        if not result:
            return None
        return " ".join(line[1] for line in result)
    except Exception:
        return None
