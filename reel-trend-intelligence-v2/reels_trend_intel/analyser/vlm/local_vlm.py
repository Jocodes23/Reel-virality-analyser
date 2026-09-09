"""Local VLM adapter (Qwen2-VL / Qwen3-VL class models via transformers).

Runs entirely on your machine — no API key, no per-reel cost, nothing leaves the
box. The tradeoff is honest and worth stating: on a 4 GB card only a small
(~2B, 4-bit) model fits, and small VLMs are noticeably weaker at the nuanced
judgements D1 (cinematography) and D3 (grade) depend on. Prefer a hosted provider
when label quality matters; use local for bulk pre-passes, offline work, or when
data must not leave the machine.

Local models do not guarantee schema-valid JSON, so the response is extracted and
validated here; `analyse_with_retry` handles the retry.

Install with: pip install "transformers>=4.45" accelerate bitsandbytes qwen-vl-utils
"""

from __future__ import annotations

import io
import json
import re

from reels_trend_intel.analyser.types import VLMAnalysis
from reels_trend_intel.analyser.vlm.base import (
    SYSTEM_PROMPT,
    Availability,
    VLMAdapter,
    VLMUsage,
    build_user_prompt,
)
from reels_trend_intel.observability.logging import get_logger

log = get_logger("analyser.vlm.local")

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict:
    """Pull the JSON object out of a local model's reply.

    Small models like to wrap JSON in prose or ```json fences; take the outermost
    brace-balanced span rather than trusting the whole string.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = _JSON_RE.search(cleaned)
    if not match:
        raise ValueError(f"no JSON object found in local model output: {cleaned[:200]!r}")
    return json.loads(match.group(0))


class LocalVLMAdapter(VLMAdapter):
    provider = "local"

    def __init__(self, cfg) -> None:  # type: ignore[no-untyped-def]
        super().__init__(cfg)
        self._model: object | None = None
        self._processor: object | None = None

    @property
    def model_id(self) -> str:
        return self.cfg.local_model

    def _resolve_device(self) -> str:
        want = self.cfg.local_device
        if want == "cpu":
            return "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def availability(self) -> Availability:
        try:
            import transformers  # noqa: F401
        except Exception:
            return Availability(False, "transformers not installed",
                                ["pip install transformers accelerate"])
        device = self._resolve_device()
        detail = f"ready — {self.model_id} on {device}"
        extras: list[str] = []
        if device == "cuda":
            try:
                import torch

                total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
                detail += f" ({total_mb:.0f} MiB VRAM)"
                if total_mb < 6000 and not self.cfg.local_load_4bit:
                    extras.append("under 6 GB VRAM — enable local_load_4bit")
                if total_mb < 6000:
                    extras.append("small-VLM label quality is materially lower than hosted")
            except Exception:
                pass
        else:
            extras.append("running on CPU — expect very slow per-reel analysis")
        if self.cfg.local_load_4bit:
            try:
                import bitsandbytes  # noqa: F401
            except Exception:
                return Availability(False, "local_load_4bit set but bitsandbytes missing",
                                    ["pip install bitsandbytes"])
        return Availability(True, detail, extras)

    def _load(self) -> tuple[object, object]:
        if self._model is None or self._processor is None:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor

            device = self._resolve_device()
            kwargs: dict = {"dtype": torch.float16 if device == "cuda" else torch.float32}
            if self.cfg.local_load_4bit and device == "cuda":
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
                )
                kwargs["device_map"] = "auto"
            model = AutoModelForVision2Seq.from_pretrained(self.cfg.local_model, **kwargs)
            if "device_map" not in kwargs:
                model = model.to(device)
            self._model = model
            self._processor = AutoProcessor.from_pretrained(self.cfg.local_model)
            log.info("local_vlm_loaded", model=self.cfg.local_model, device=device,
                     four_bit=self.cfg.local_load_4bit)
        return self._model, self._processor

    async def analyse(
        self, montage_png: bytes, *, duration_s: float | None = None,
        cut_count: int | None = None,
    ) -> tuple[VLMAnalysis, VLMUsage]:
        import asyncio

        return await asyncio.to_thread(self._analyse_sync, montage_png, duration_s, cut_count)

    def _analyse_sync(
        self, montage_png: bytes, duration_s: float | None, cut_count: int | None
    ) -> tuple[VLMAnalysis, VLMUsage]:
        import torch
        from PIL import Image

        model, processor = self._load()
        image = Image.open(io.BytesIO(montage_png)).convert("RGB")
        instruction = (
            f"{SYSTEM_PROMPT}\n\n{build_user_prompt(duration_s, cut_count)}\n\n"
            "Reply with ONLY a JSON object matching this schema (no prose, no code "
            f"fences):\n{json.dumps(VLMAnalysis.model_json_schema())}"
        )
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": instruction}]}]
        text = processor.apply_chat_template(  # type: ignore[attr-defined]
            messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image],  # type: ignore[operator]
                           return_tensors="pt").to(model.device)  # type: ignore[attr-defined]
        with torch.inference_mode():
            generated = model.generate(  # type: ignore[attr-defined]
                **inputs, max_new_tokens=self.cfg.local_max_new_tokens, do_sample=False)
        trimmed = generated[:, inputs["input_ids"].shape[1]:]
        reply = processor.batch_decode(  # type: ignore[attr-defined]
            trimmed, skip_special_tokens=True)[0]
        analysis = VLMAnalysis.model_validate(extract_json(reply))
        usage = VLMUsage(
            input_tokens=int(inputs["input_ids"].shape[1]),
            output_tokens=int(trimmed.shape[1]),
        )
        return analysis, usage

    def cost_usd(self, usage: VLMUsage) -> float:
        return 0.0  # local inference has no per-call price

    async def aclose(self) -> None:
        self._model = None
        self._processor = None
