"""Embedding backends behind one interface.

Two implementations:
  * DeterministicBackend (default, offline, no downloads): similarity-PRESERVING
    descriptors (image thumbnail, hashed caption n-grams, per-audio_id signature)
    so clustering recovers real structure without any model weights.
  * RealBackend (config: features.embedding_backend="real"): CLIP-ViT-B/32 for
    images, all-MiniLM-L6-v2 for captions via sentence-transformers, with device
    resolution (cuda/cpu/auto), batching, a VRAM cap, and automatic CPU fallback.

Vectors are cached upstream by content hash (never recompute).
"""

from __future__ import annotations

import hashlib
import io
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import HashingVectorizer

from reels_trend_intel.config.settings import FeatureConfig
from reels_trend_intel.observability.logging import get_logger

log = get_logger("features.embeddings")


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + 1e-9)


def resolve_device(device: str) -> str:
    if device != "cuda" and device != "auto":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    if device == "cuda":
        log.warning("cuda_requested_unavailable", fallback="cpu")
    return "cpu"


class EmbeddingBackend(ABC):
    name: str = "base"
    image_dim: int = 0
    caption_dim: int = 0
    audio_dim: int = 0

    @abstractmethod
    def embed_images(self, images: list[bytes]) -> np.ndarray: ...

    @abstractmethod
    def embed_captions(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def embed_audio(self, sigs: list[tuple[str, float, float]]) -> np.ndarray: ...


class DeterministicBackend(EmbeddingBackend):
    name = "deterministic"
    image_dim = 12 * 12 * 3
    caption_dim = 256
    audio_dim = 64

    def __init__(self, cfg: FeatureConfig) -> None:
        self.cfg = cfg
        self._hv = HashingVectorizer(
            n_features=self.caption_dim, alternate_sign=False,
            ngram_range=(1, 2), norm="l2",
        )

    def embed_images(self, images: list[bytes]) -> np.ndarray:
        out = np.zeros((len(images), self.image_dim), dtype=np.float32)
        for i, data in enumerate(images):
            img = Image.open(io.BytesIO(data)).convert("RGB").resize((12, 12),
                                                                      Image.Resampling.BILINEAR)
            out[i] = (np.asarray(img, dtype=np.float32) / 255.0).flatten()
        return _l2(out)

    def embed_captions(self, texts: list[str]) -> np.ndarray:
        safe = [t if t.strip() else "∅" for t in texts]
        return np.asarray(self._hv.transform(safe).todense(), dtype=np.float32)

    def embed_audio(self, sigs: list[tuple[str, float, float]]) -> np.ndarray:
        out = np.zeros((len(sigs), self.audio_dim), dtype=np.float32)
        for i, (audio_id, tempo, energy) in enumerate(sigs):
            seed = int(hashlib.sha256((audio_id or "none").encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self.audio_dim - 2).astype(np.float32)
            out[i, :-2] = vec / (np.linalg.norm(vec) + 1e-9)
            out[i, -2] = np.tanh((tempo - 110.0) / 40.0)
            out[i, -1] = float(np.clip(energy, 0, 1))
        return out


class RealBackend(EmbeddingBackend):
    """CLIP image + MiniLM caption embeddings (sentence-transformers)."""

    name = "real"

    def __init__(self, cfg: FeatureConfig) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self._clip: Any = None   # SentenceTransformer | False(sentinel) | None
        self._txt: Any = None
        self._det = DeterministicBackend(cfg)  # audio fallback + safety net
        self.image_dim = 512
        self.caption_dim = 384
        self.audio_dim = self._det.audio_dim

    def _vram_ok(self) -> bool:
        if self.device != "cuda":
            return True
        try:
            import torch

            free, _ = torch.cuda.mem_get_info()
            return free / (1024 * 1024) > (self.cfg.vram_cap_mb * 0.2)
        except Exception:
            return True

    def _load_clip(self) -> object | None:
        if self._clip is None:
            try:
                from sentence_transformers import SentenceTransformer

                dev = self.device if self._vram_ok() else "cpu"
                self._clip = SentenceTransformer(self.cfg.clip_model, device=dev)
            except Exception as exc:  # network/VRAM/etc -> deterministic fallback
                log.warning("clip_load_failed", error=str(exc), fallback="deterministic")
                self._clip = False  # sentinel: unavailable
        return self._clip or None

    def _load_txt(self) -> object | None:
        if self._txt is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._txt = SentenceTransformer(self.cfg.caption_model, device=self.device)
            except Exception as exc:
                log.warning("txt_load_failed", error=str(exc), fallback="deterministic")
                self._txt = False
        return self._txt or None

    def embed_images(self, images: list[bytes]) -> np.ndarray:
        model = self._load_clip()
        if model is None:
            self.image_dim = self._det.image_dim
            return self._det.embed_images(images)
        pil = [Image.open(io.BytesIO(b)).convert("RGB") for b in images]
        vecs = model.encode(pil, batch_size=self.cfg.batch_size,  # type: ignore[attr-defined]
                            convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)

    def embed_captions(self, texts: list[str]) -> np.ndarray:
        model = self._load_txt()
        if model is None:
            self.caption_dim = self._det.caption_dim
            return self._det.embed_captions(texts)
        vecs = model.encode(texts, batch_size=self.cfg.batch_size,  # type: ignore[attr-defined]
                            convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)

    def embed_audio(self, sigs: list[tuple[str, float, float]]) -> np.ndarray:
        return self._det.embed_audio(sigs)


def get_embedding_backend(cfg: FeatureConfig) -> EmbeddingBackend:
    if cfg.embedding_backend == "real":
        return RealBackend(cfg)
    return DeterministicBackend(cfg)
