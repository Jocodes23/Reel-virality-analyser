"""Joint-embedding clustering -> emergent reel "types".

joint = [w_img·img ‖ w_aud·audio ‖ w_cap·caption]  (each modality L2-normalised)
        -> UMAP (if installed) or PCA -> HDBSCAN (sklearn built-in).

Returns reel_id -> cluster label (-1 = noise). Deterministic embeddings are
similarity-preserving, so this recovers real structure offline.
"""

from __future__ import annotations

import numpy as np

from reels_trend_intel.config.settings import TrendConfig
from reels_trend_intel.features.types import ReelFeatures
from reels_trend_intel.observability.logging import get_logger
from reels_trend_intel.storage.base import StorageBackend

log = get_logger("trends.cluster")


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


async def assemble_joint(
    storage: StorageBackend, feats: list[ReelFeatures], cfg: TrendConfig
) -> tuple[list[str], np.ndarray]:
    # Determine per-modality dims from the first available vectors.
    async def fetch(hsh: str | None, kind: str) -> np.ndarray | None:
        if not hsh:
            return None
        v = await storage.get_embedding(hsh, kind)
        return np.asarray(v, dtype=np.float32) if v is not None else None

    rows: list[tuple[str, np.ndarray | None, np.ndarray | None, np.ndarray | None]] = []
    img_d = cap_d = aud_d = 0
    for f in feats:
        iv = await fetch(f.image_hash, "image")
        cv = await fetch(f.caption_hash, "caption")
        av = await fetch(f.audio_hash, "audio")
        img_d = img_d or (iv.shape[0] if iv is not None else 0)
        cap_d = cap_d or (cv.shape[0] if cv is not None else 0)
        aud_d = aud_d or (av.shape[0] if av is not None else 0)
        rows.append((f.reel_id, iv, cv, av))

    ids: list[str] = []
    mats: list[np.ndarray] = []
    for rid, iv, cv, av in rows:
        parts = [
            cfg.weight_image * _l2(iv if iv is not None else np.zeros(img_d, np.float32)),
            cfg.weight_audio * _l2(av if av is not None else np.zeros(aud_d, np.float32)),
            cfg.weight_caption * _l2(cv if cv is not None else np.zeros(cap_d, np.float32)),
        ]
        mats.append(np.concatenate(parts).astype(np.float32))
        ids.append(rid)
    return ids, np.vstack(mats) if mats else np.zeros((0, 1), np.float32)


def reduce_and_cluster(matrix: np.ndarray, cfg: TrendConfig, seed: int = 1729) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.zeros(0, dtype=int)
    n = matrix.shape[0]
    reduced = _reduce(matrix, cfg, seed)
    from sklearn.cluster import HDBSCAN

    min_cs = min(cfg.hdbscan_min_cluster_size, max(2, n // 3))
    clusterer = HDBSCAN(
        min_cluster_size=min_cs,
        min_samples=min(cfg.hdbscan_min_samples, min_cs),
        metric="euclidean",
        cluster_selection_method=cfg.hdbscan_cluster_selection_method,
    )
    labels = clusterer.fit_predict(reduced)
    log.info("clustered", n=n, clusters=int(labels.max() + 1 if labels.size else 0),
             noise=int((labels == -1).sum()))
    return labels


def _reduce(matrix: np.ndarray, cfg: TrendConfig, seed: int) -> np.ndarray:
    n, d = matrix.shape
    target = min(cfg.umap_n_components, max(2, n - 1), d)
    if n <= cfg.umap_n_components + 1:
        return matrix
    try:
        import umap  # optional

        reducer = umap.UMAP(
            n_components=target, n_neighbors=min(cfg.umap_n_neighbors, n - 1),
            random_state=seed, metric="euclidean",
        )
        return np.asarray(reducer.fit_transform(matrix), dtype=np.float32)
    except Exception:
        from sklearn.decomposition import PCA

        return np.asarray(
            PCA(n_components=target, random_state=seed).fit_transform(matrix), dtype=np.float32
        )
