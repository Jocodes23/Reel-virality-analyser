"""Claude (Anthropic) VLM adapter.

Uses the official `anthropic` SDK's structured-output helper, which validates the
response against our pydantic schema server-side and hands back a parsed object —
so there is no hand-rolled JSON extraction to go wrong.

Credentials come from the environment (`ANTHROPIC_API_KEY`) or an `ant auth login`
profile; nothing is read from code. Install with: pip install "anthropic>=0.69".
"""

from __future__ import annotations

import base64

from reels_trend_intel.analyser.types import VLMAnalysis
from reels_trend_intel.analyser.vlm.base import (
    SYSTEM_PROMPT,
    Availability,
    VLMAdapter,
    VLMUsage,
    build_user_prompt,
    price,
)
from reels_trend_intel.observability.logging import get_logger

log = get_logger("analyser.vlm.anthropic")


class AnthropicVLMAdapter(VLMAdapter):
    provider = "anthropic"

    def __init__(self, cfg) -> None:  # type: ignore[no-untyped-def]
        super().__init__(cfg)
        self._client: object | None = None

    @property
    def model_id(self) -> str:
        return self.cfg.anthropic_model

    def availability(self) -> Availability:
        try:
            import anthropic  # noqa: F401
        except Exception:
            return Availability(False, "anthropic SDK not installed",
                                ["pip install anthropic"])
        import os

        has_cred = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
        if not has_cred:
            return Availability(
                False,
                "no credential (set ANTHROPIC_API_KEY, or run `ant auth login`)",
                ["export ANTHROPIC_API_KEY=sk-ant-..."],
            )
        return Availability(True, f"ready — {self.model_id}")

    def _get_client(self) -> object:
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic(timeout=self.cfg.request_timeout_s)
        return self._client

    async def analyse(
        self, montage_png: bytes, *, duration_s: float | None = None,
        cut_count: int | None = None,
    ) -> tuple[VLMAnalysis, VLMUsage]:
        client = self._get_client()
        b64 = base64.standard_b64encode(montage_png).decode("utf-8")
        response = await client.messages.parse(  # type: ignore[attr-defined]
            model=self.cfg.anthropic_model,
            max_tokens=self.cfg.anthropic_max_tokens,
            system=SYSTEM_PROMPT,
            output_config={"effort": self.cfg.anthropic_effort},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": build_user_prompt(duration_s, cut_count)},
                ],
            }],
            output_format=VLMAnalysis,
        )
        # Safety classifiers can decline (HTTP 200 + stop_reason "refusal").
        if getattr(response, "stop_reason", None) == "refusal":
            detail = getattr(response, "stop_details", None)
            raise RuntimeError(f"claude refused the request: {detail}")
        parsed = response.parsed_output
        if parsed is None:
            raise ValueError("claude returned no parsed output")
        usage = VLMUsage(
            input_tokens=int(getattr(response.usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(response.usage, "output_tokens", 0) or 0),
        )
        return parsed, usage

    def cost_usd(self, usage: VLMUsage) -> float:
        return price(usage, self.cfg.anthropic_price_in_per_mtok,
                     self.cfg.anthropic_price_out_per_mtok)

    async def aclose(self) -> None:
        client = self._client
        if client is not None and hasattr(client, "close"):
            await client.close()  # type: ignore[misc]
            self._client = None
