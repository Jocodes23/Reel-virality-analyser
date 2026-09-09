"""Caption/hashtag themes via c-TF-IDF (a lightweight BERTopic-style labeller).

For each cluster we compute class-based TF-IDF over the members' hashtags + tokens
and take the top terms as the human label. When the `bertopic` extra is installed
the full model can be swapped in; this fallback needs only scikit-learn.
"""

from __future__ import annotations

import re

import numpy as np

from reels_trend_intel.features.types import ReelFeatures

_TOKEN = re.compile(r"[#\w]{3,}", re.UNICODE)
_STOP = {"the", "and", "for", "that", "this", "with", "make", "but", "your", "you",
         "reels", "fyp", "save", "follow", "more", "internet", "broke", "pov"}


def _tokens(f: ReelFeatures) -> list[str]:
    toks = [t.lower() for t in _TOKEN.findall(f.caption.text)]
    toks += [h.lstrip("#") for h in f.caption.hashtags]
    toks.append(f.scene.label)
    return [t for t in toks if t not in _STOP and not t.isdigit()]


def label_clusters(
    feats: list[ReelFeatures], labels: np.ndarray, top_k: int = 4
) -> dict[int, list[str]]:
    """Return {cluster_label: [top terms]} using c-TF-IDF."""
    vocab: dict[str, int] = {}
    docs: dict[int, dict[str, int]] = {}
    df: dict[str, int] = {}
    for f, lbl in zip(feats, labels, strict=False):
        c = int(lbl)
        bag = docs.setdefault(c, {})
        seen: set[str] = set()
        for tok in _tokens(f):
            vocab.setdefault(tok, len(vocab))
            bag[tok] = bag.get(tok, 0) + 1
            seen.add(tok)
        for tok in seen:
            df[tok] = df.get(tok, 0) + 1

    n_clusters = len(docs)
    out: dict[int, list[str]] = {}
    for c, bag in docs.items():
        total = sum(bag.values()) or 1
        scores: dict[str, float] = {}
        for tok, cnt in bag.items():
            tf = cnt / total
            idf = np.log((1 + n_clusters) / (1 + df.get(tok, 1))) + 1.0
            scores[tok] = tf * idf
        top = sorted(scores, key=lambda t: scores[t], reverse=True)[:top_k]
        out[c] = top
    return out
