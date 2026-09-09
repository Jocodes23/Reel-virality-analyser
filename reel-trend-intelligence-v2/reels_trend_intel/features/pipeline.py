"""Gated + cached + batched feature extraction.

Per reel: cheap gate first (dedupe/quality) -> only survivors get colour/scene/
audio/caption/format extraction -> embeddings are computed in BATCHES and cached
by content hash so we never embed the same media twice. Embedding *vectors* land
in the storage embedding cache; ReelFeatures keeps the hash keys that point at
them.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from reels_trend_intel import FEATURE_SCHEMA_VERSION
from reels_trend_intel.config.settings import Settings
from reels_trend_intel.features import audio as audio_mod
from reels_trend_intel.features import caption as caption_mod
from reels_trend_intel.features.color import extract_color
from reels_trend_intel.features.embeddings import EmbeddingBackend, get_embedding_backend
from reels_trend_intel.features.format import extract_format
from reels_trend_intel.features.gate import CheapGate
from reels_trend_intel.features.hashing import content_hash, perceptual_hash, text_hash
from reels_trend_intel.features.scene import classify_scene_heuristic
from reels_trend_intel.features.types import ReelFeatures
from reels_trend_intel.observability.logging import get_logger
from reels_trend_intel.observability.metrics import StageTimer
from reels_trend_intel.storage.base import StorageBackend
from reels_trend_intel.storage.rows import Audio, EmbeddingRow, EngagementSample, Reel

log = get_logger("features.pipeline")


@dataclass
class FeatureInput:
    reel: Reel
    audio: Audio | None
    media: bytes | None
    metadata: dict[str, Any]
    latest: EngagementSample | None


class FeaturePipeline:
    def __init__(
        self, storage: StorageBackend, settings: Settings,
        backend: EmbeddingBackend | None = None,
    ) -> None:
        self.storage = storage
        self.settings = settings
        self.gate = CheapGate(settings.features)
        self.backend = backend or get_embedding_backend(settings.features)
        self._rng = random.Random(settings.app.seed)

    async def extract_batch(self, items: list[FeatureInput]) -> list[ReelFeatures]:
        survivors: list[tuple[FeatureInput, int, str]] = []
        with StageTimer("gate", count=len(items)):
            for it in items:
                if it.media is None:
                    await self.storage.mark_stage(it.reel.reel_id, "gated:no_media")
                    continue
                ph = perceptual_hash(it.media)
                # persist phash so dedupe works across restarts
                it.reel.phash = ph
                res = await self.gate.check(it.reel.reel_id, ph, it.latest, self.storage)
                if not res.passed:
                    await self.storage.mark_stage(it.reel.reel_id, f"gated:{res.reason}")
                    continue
                survivors.append((it, ph, content_hash(it.media)))

        if not survivors:
            return []

        # persist phash updates for dedupe
        await self.storage.upsert_reels([s[0].reel for s in survivors])

        results: list[ReelFeatures] = []
        with StageTimer("extract", count=len(survivors)):
            # scalar/structured features (per item; cheap, CPU)
            scratch: list[dict[str, Any]] = []
            for it, _ph, img_hash in survivors:
                color = extract_color(
                    it.media,  # type: ignore[arg-type]
                    k=self.settings.features.color_k,
                    max_side=self.settings.features.color_max_side,
                    seed=self.settings.app.seed,
                )
                scene = classify_scene_heuristic(
                    it.reel.caption, color, self.settings.app.niche_keywords
                )
                cap = caption_mod.extract_caption(it.reel.caption)
                if (self.settings.features.enable_ocr
                        and self._rng.random() < self.settings.features.ocr_sample_rate):
                    cap.ocr_text = caption_mod.run_ocr(it.media)  # type: ignore[arg-type]
                aud = audio_mod.extract_audio_offline(it.audio, it.reel.audio_id)
                fmt = extract_format(it.reel.posted_at, it.metadata)
                scratch.append({
                    "it": it, "img_hash": img_hash, "color": color, "scene": scene,
                    "cap": cap, "aud": aud, "fmt": fmt,
                    "cap_hash": text_hash(it.reel.caption),
                    "aud_hash": (text_hash("aud:" + aud.audio_id) if aud.audio_id else None),
                })

        await self._embed_and_cache(scratch)

        for s in scratch:
            it = s["it"]
            feats = ReelFeatures(
                reel_id=it.reel.reel_id, feature_schema_version=FEATURE_SCHEMA_VERSION,
                color=s["color"], scene=s["scene"], audio=s["aud"], caption=s["cap"],
                fmt=s["fmt"], image_hash=s["img_hash"], caption_hash=s["cap_hash"],
                audio_hash=s["aud_hash"],
            )
            results.append(feats)
        await self.storage.upsert_features(results)
        for f in results:
            await self.storage.mark_stage(f.reel_id, "featurized")
        log.info("features_extracted", survivors=len(results), gated=len(items) - len(results))
        return results

    async def _embed_and_cache(self, scratch: list[dict[str, Any]]) -> None:
        """Compute missing embeddings in batches; cache by content hash."""
        with StageTimer("embed", count=len(scratch)):
            # IMAGE
            img_missing = [(s["img_hash"], s["it"].media) for s in scratch
                           if not await self.storage.has_embedding(s["img_hash"], "image")]
            uniq_img: dict[str, bytes] = {}
            for h, b in img_missing:
                uniq_img.setdefault(h, b)
            if uniq_img:
                hashes = list(uniq_img)
                vecs = self.backend.embed_images([uniq_img[h] for h in hashes])
                await self.storage.put_embeddings(
                    [EmbeddingRow(content_hash=h, kind="image", vector=vecs[i].tolist())
                     for i, h in enumerate(hashes)])
            # CAPTION
            cap_missing = {s["cap_hash"]: s["it"].reel.caption for s in scratch
                           if not await self.storage.has_embedding(s["cap_hash"], "caption")}
            if cap_missing:
                hashes = list(cap_missing)
                vecs = self.backend.embed_captions([cap_missing[h] for h in hashes])
                await self.storage.put_embeddings(
                    [EmbeddingRow(content_hash=h, kind="caption", vector=vecs[i].tolist())
                     for i, h in enumerate(hashes)])
            # AUDIO
            aud_missing: dict[str, tuple[str, float, float]] = {}
            for s in scratch:
                ah = s["aud_hash"]
                if ah and not await self.storage.has_embedding(ah, "audio"):
                    a = s["aud"]
                    aud_missing[ah] = (a.audio_id or "none", a.tempo or 0.0, a.energy or 0.0)
            if aud_missing:
                hashes = list(aud_missing)
                vecs = self.backend.embed_audio([aud_missing[h] for h in hashes])
                await self.storage.put_embeddings(
                    [EmbeddingRow(content_hash=h, kind="audio", vector=vecs[i].tolist())
                     for i, h in enumerate(hashes)])
