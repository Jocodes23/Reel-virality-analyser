"""Background/scene classification.

RealBackend uses CLIP zero-shot (image vs. label prompts). Offline, a cheap
deterministic heuristic scores the candidate labels from caption keywords + the
colour signature. Labels: food-closeup / table / outdoor / studio / text-on-solid.
"""

from __future__ import annotations

from reels_trend_intel.features.types import ColorFeatures, SceneFeatures

LABELS = ["food-closeup", "table", "outdoor", "studio", "text-on-solid"]

_LABEL_KEYWORDS: dict[str, list[str]] = {
    "food-closeup": ["burger", "taco", "dish", "food", "bite", "crispy", "cheese"],
    "table": ["matcha", "coffee", "cafe", "latte", "table", "morning", "aesthetic"],
    "outdoor": ["street", "travel", "outdoor", "midnight", "jump", "cliff", "summer"],
    "studio": ["grwm", "getready", "routine", "makeup", "glow", "studio"],
    "text-on-solid": ["recipe", "save this", "ingredient", "hack", "steps", "how to"],
}


def classify_scene_heuristic(
    caption: str, color: ColorFeatures, niche_keywords: list[str]
) -> SceneFeatures:
    text = caption.lower()
    scores: dict[str, float] = {}
    for label, kws in _LABEL_KEYWORDS.items():
        s = sum(1.0 for kw in kws if kw in text)
        scores[label] = s
    # Colour priors nudge ambiguous cases.
    if color.key == "low-key" and color.warm_cool_ratio > 1.2:
        scores["food-closeup"] += 0.5
    if color.key == "high-key" and color.saturation < 0.25:
        scores["text-on-solid"] += 0.5
    if color.warm_cool_ratio < 0.7:
        scores["outdoor"] += 0.3
    total = sum(scores.values()) or 1.0
    probs = {k: v / total for k, v in scores.items()}
    label = max(probs, key=lambda k: probs[k]) if total > 0 else "table"
    is_niche = any(kw in text for kw in niche_keywords) or label in ("food-closeup", "table")
    return SceneFeatures(label=label, scores=probs, is_niche=is_niche)


def classify_scene_clip(
    image_bytes: bytes, caption: str, niche_keywords: list[str], model: object
) -> SceneFeatures:  # pragma: no cover - requires model weights
    import io

    import numpy as np
    from PIL import Image

    prompts = [f"a photo of {lbl.replace('-', ' ')}" for lbl in LABELS]
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    iv = model.encode([img], convert_to_numpy=True, normalize_embeddings=True)  # type: ignore
    tv = model.encode(prompts, convert_to_numpy=True, normalize_embeddings=True)  # type: ignore
    sims = (iv @ tv.T)[0]
    exp = np.exp(sims - sims.max())
    probs = exp / exp.sum()
    scores = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    label = max(scores, key=lambda k: scores[k])
    is_niche = any(kw in caption.lower() for kw in niche_keywords) or label in (
        "food-closeup", "table")
    return SceneFeatures(label=label, scores=scores, is_niche=is_niche)
