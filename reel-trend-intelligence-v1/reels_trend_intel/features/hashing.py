"""Content hashing + perceptual hashing for dedupe and the embedding cache."""

from __future__ import annotations

import hashlib
import io

import numpy as np
from PIL import Image


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def perceptual_hash(image_bytes: bytes, hash_size: int = 8) -> int:
    """64-bit DCT perceptual hash (pHash). Robust to small re-encodes/resizes."""
    img = Image.open(io.BytesIO(image_bytes)).convert("L").resize(
        (hash_size * 4, hash_size * 4), Image.Resampling.LANCZOS
    )
    pixels = np.asarray(img, dtype=np.float32)
    dct = _dct2(pixels)
    low = dct[:hash_size, :hash_size]
    med = np.median(low[1:].flatten())  # exclude DC term
    bits = (low > med).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def _dct2(a: np.ndarray) -> np.ndarray:
    return _dct1(_dct1(a.T).T)


def _dct1(a: np.ndarray) -> np.ndarray:
    n = a.shape[1]
    k = np.arange(n)
    basis = np.cos(np.pi * (2 * k[:, None] + 1) * k[None, :] / (2 * n))
    return a @ basis.T


def hamming(a: int, b: int) -> int:
    return bin((a ^ b) & ((1 << 64) - 1)).count("1")


_MASK64 = (1 << 64) - 1


def to_signed64(u: int) -> int:
    """Map an unsigned 64-bit value to signed (DB BIGINT/INTEGER are signed)."""
    u &= _MASK64
    return u - (1 << 64) if u >= (1 << 63) else u


def to_unsigned64(s: int) -> int:
    return s & _MASK64
