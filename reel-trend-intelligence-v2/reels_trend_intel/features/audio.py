"""Audio features: acoustic fingerprint (for same-song clustering) + tempo/energy.

RealBackend: Chromaprint/AcoustID fingerprint + librosa tempo/energy from the
waveform. Offline (no waveform): a DETERMINISTIC fingerprint derived from the
audio_id, so reels sharing a song share a fingerprint and cluster together — the
same downstream behaviour, without audio bytes.
"""

from __future__ import annotations

import hashlib

import numpy as np

from reels_trend_intel.features.types import AudioFeatures
from reels_trend_intel.observability.logging import get_logger
from reels_trend_intel.storage.rows import Audio

log = get_logger("features.audio")


def _deterministic_fp(audio_id: str) -> str:
    return hashlib.sha1(audio_id.encode()).hexdigest()[:16]


def extract_audio_offline(audio: Audio | None, audio_id: str | None) -> AudioFeatures:
    aid = audio_id or (audio.audio_id if audio else None)
    if aid is None:
        return AudioFeatures()
    rng = np.random.default_rng(int(hashlib.sha256(aid.encode()).hexdigest()[:8], 16))
    tempo = float(80 + rng.random() * 90)       # 80..170 bpm
    energy = float(rng.random())
    return AudioFeatures(
        audio_id=aid,
        fingerprint=_deterministic_fp(aid),
        tempo=tempo,
        energy=energy,
        is_original_audio=bool(audio.is_original_audio) if audio else False,
    )


def extract_audio_real(
    audio: Audio | None, audio_id: str | None, waveform: bytes | None
) -> AudioFeatures:  # pragma: no cover - requires librosa + waveform
    if waveform is None:
        return extract_audio_offline(audio, audio_id)
    try:
        import io

        import librosa

        y, sr = librosa.load(io.BytesIO(waveform), mono=True)
        tempo = float(librosa.beat.tempo(y=y, sr=sr)[0])
        energy = float(np.sqrt(np.mean(y**2)))
        try:
            import acoustid

            fp = acoustid.fingerprint(sr, 1, (y * 32767).astype("int16").tobytes())[1]
            fingerprint = hashlib.sha1(fp).hexdigest()[:16]
        except Exception:
            fingerprint = _deterministic_fp(audio_id or "none")
        return AudioFeatures(
            audio_id=audio_id, fingerprint=fingerprint, tempo=tempo, energy=energy,
            is_original_audio=bool(audio.is_original_audio) if audio else False,
        )
    except Exception as exc:
        log.warning("audio_real_failed", error=str(exc), fallback="offline")
        return extract_audio_offline(audio, audio_id)
