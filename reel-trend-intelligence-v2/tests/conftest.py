from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from reels_trend_intel.config.settings import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings()
    s.storage.sqlite_path = str(tmp_path / "test.db")
    s.report.export_dir = str(tmp_path / "exports")
    s.app.seed = 7
    return s


def make_image(color=(200, 120, 60), size=64, seed=0) -> bytes:
    rng = np.random.default_rng(seed)
    arr = np.clip(np.array(color, np.float32) + rng.normal(0, 8, (size, size, 3)), 0, 255)
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8), "RGB").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def image_bytes() -> bytes:
    return make_image()
