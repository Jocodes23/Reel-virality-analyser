"""Colour features via k-means in CIELAB (k=5).

Outputs dominant/accent/background hex + proportions, mean luminance, saturation,
warm/cool ratio, contrast, and high-key/low-key classification. Pure NumPy +
scikit-learn; runs on CPU, no model download.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from reels_trend_intel.features.types import ColorFeatures


def _rgb_to_hex(rgb: np.ndarray) -> str:
    r, g, b = (int(np.clip(c, 0, 255)) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """rgb in [0,1] -> CIELAB. Vectorised over (...,3)."""
    mask = rgb > 0.04045
    lin = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    m = np.array([
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505],
    ])
    xyz = lin @ m.T
    white = np.array([0.95047, 1.0, 1.08883])
    xyz = xyz / white
    eps = 0.008856
    kappa = 903.3
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    eps = 0.008856
    def inv(t: np.ndarray) -> np.ndarray:
        return np.where(t**3 > eps, t**3, (116 * t - 16) / 903.3)
    white = np.array([0.95047, 1.0, 1.08883])
    xyz = np.stack([inv(fx), inv(fy), inv(fz)], axis=-1) * white
    m_inv = np.array([
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570],
    ])
    lin = xyz @ m_inv.T
    srgb = np.where(lin > 0.0031308, 1.055 * np.clip(lin, 0, None) ** (1 / 2.4) - 0.055,
                    12.92 * lin)
    return np.clip(srgb, 0, 1) * 255


def extract_color(image_bytes: bytes, k: int = 5, max_side: int = 256,
                  seed: int = 1729) -> ColorFeatures:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((max_side, max_side))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    pixels = arr.reshape(-1, 3)
    lab = _srgb_to_lab(pixels)

    n_clusters = min(k, max(1, len(np.unique(pixels, axis=0))))
    km = KMeans(n_clusters=n_clusters, n_init=4, random_state=seed)
    labels = km.fit_predict(lab)
    centers_lab = km.cluster_centers_
    counts = np.bincount(labels, minlength=n_clusters).astype(np.float64)
    props = counts / counts.sum()
    order = np.argsort(props)[::-1]

    centers_rgb = _lab_to_rgb(centers_lab)
    palette = [(_rgb_to_hex(centers_rgb[i]), float(props[i])) for i in order]

    chroma = np.sqrt(centers_lab[:, 1] ** 2 + centers_lab[:, 2] ** 2)
    accent_idx = int(np.argmax(chroma))
    background_idx = int(order[0])
    dominant_idx = int(order[1]) if n_clusters > 1 else background_idx

    L = lab[:, 0] / 100.0
    luminance = float(np.clip(L.mean(), 0, 1))
    contrast = float(np.clip(L.std(), 0, 1))
    warm = float((lab[:, 1] > 0).mean())
    cool = float((lab[:, 1] <= 0).mean())
    warm_cool = warm / (cool + 1e-6)
    # saturation via HSV
    hsv = np.asarray(Image.fromarray((arr * 255).astype(np.uint8)).convert("HSV"),
                     dtype=np.float32) / 255.0
    saturation = float(hsv[..., 1].mean())
    if luminance >= 0.62:
        key = "high-key"
    elif luminance <= 0.32:
        key = "low-key"
    else:
        key = "mid-key"

    return ColorFeatures(
        dominant_hex=_rgb_to_hex(centers_rgb[dominant_idx]),
        accent_hex=_rgb_to_hex(centers_rgb[accent_idx]),
        background_hex=_rgb_to_hex(centers_rgb[background_idx]),
        palette=palette,
        luminance=luminance,
        saturation=saturation,
        warm_cool_ratio=float(warm_cool),
        contrast=contrast,
        key=key,
    )
