"""ChatGPT (OpenAI) VLM adapter.

Uses the official `openai` SDK with a strict `json_schema` response format built
from the same pydantic model, so both hosted providers are held to an identical
contract. Credentials come from `OPENAI_API_KEY` only.

Install with: pip install "openai>=1.40". The model must be vision-capable.
"""

from __future__ import annotations

import base64
import json

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

log = get_logger("analyser.vlm.openai")


def _strict_schema() -> dict:
    """pydantic -> OpenAI strict json_schema (needs additionalProperties:false)."""
    schema = VLMAnalysis.model_json_schema()

    def harden(node: object) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                node.setdefault("additionalProperties", False)
                props = node.get("properties")
                if isinstance(props, dict):
                    node["required"] = list(props.keys())
            for value in node.values():
                harden(value)
        elif isinstance(node, list):
            for item in node:
                harden(item)

    harden(schema)
    return schema


class OpenAIVLMAdapter(VLMAdapter):
    provider = "openai"

    def __init__(self, cfg) -> None:  # type: ignore[no-untyped-def]
        super().__init__(cfg)
        self._client: object | None = None

    @property
    def model_id(self) -> str:
        return self.cfg.openai_model

    def availability(self) -> Availability:
        try:
            import openai  # noqa: F401
        except Exception:
            return Availability(False, "openai SDK not installed", ["pip install openai"])
        import os

        if not os.getenv("OPENAI_API_KEY"):
            return Availability(False, "no credential (set OPENAI_API_KEY)",
                                ["export OPENAI_API_KEY=sk-..."])
        return Availability(True, f"ready — {self.model_id}")

    def _get_client(self) -> object:
        if self._client is None:
            import openai

            self._client = openai.AsyncOpenAI(timeout=self.cfg.request_timeout_s)
        return self._client

    async def analyse(
        self, montage_png: bytes, *, duration_s: float | None = None,
        cut_count: int | None = None,
    ) -> tuple[VLMAnalysis, VLMUsage]:
        client = self._get_client()
        b64 = base64.standard_b64encode(montage_png).decode("utf-8")
        response = await client.chat.completions.create(  # type: ignore[attr-defined]
            model=self.cfg.openai_model,
            max_tokens=self.cfg.openai_max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": build_user_prompt(duration_s, cut_count)},
                ]},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "reel_analysis", "strict": True,
                                "schema": _strict_schema()},
            },
        )
        choice = response.choices[0]
        if getattr(choice, "finish_reason", None) == "content_filter":
            raise RuntimeError("openai refused the request (content_filter)")
        content = choice.message.content
        if not content:
            raise ValueError("openai returned empty content")
        analysis = VLMAnalysis.model_validate(json.loads(content))
        u = getattr(response, "usage", None)
        usage = VLMUsage(
            input_tokens=int(getattr(u, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(u, "completion_tokens", 0) or 0),
        )
        return analysis, usage

    def cost_usd(self, usage: VLMUsage) -> float:
        return price(usage, self.cfg.openai_price_in_per_mtok,
                     self.cfg.openai_price_out_per_mtok)

    async def aclose(self) -> None:
        client = self._client
        if client is not None and hasattr(client, "close"):
            await client.close()  # type: ignore[misc]
            self._client = None
