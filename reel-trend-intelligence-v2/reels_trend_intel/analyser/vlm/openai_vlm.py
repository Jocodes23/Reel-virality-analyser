"""OpenAI-compatible hosted VLM adapter — works with far more than just ChatGPT.

Uses the official `openai` SDK with a strict `json_schema` response format built
from the same pydantic model, so every provider is held to an identical contract.

Because the OpenAI protocol is the de-facto standard, pointing `openai_base_url`
at another gateway is all it takes to use a different model — **just add an API
key and a model name**:

    OpenAI       (leave base_url unset)          model: gpt-4o
    OpenRouter   https://openrouter.ai/api/v1    model: anthropic/claude-3.5-sonnet
    Together     https://api.together.xyz/v1     model: meta-llama/Llama-Vision-Free
    Groq         https://api.groq.com/openai/v1  model: llama-3.2-11b-vision-preview
    vLLM (self)  http://localhost:8000/v1        model: <whatever you serve>

The key is read from whichever env var `openai_api_key_env` names, so several
gateways can coexist. Install with: pip install "openai>=1.40".
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

    @property
    def endpoint(self) -> str:
        return self.cfg.openai_base_url or "https://api.openai.com/v1 (default)"

    def availability(self) -> Availability:
        try:
            import openai  # noqa: F401
        except Exception:
            return Availability(False, "openai SDK not installed", ["pip install openai"])
        import os

        key_var = self.cfg.openai_api_key_env
        if not os.getenv(key_var):
            return Availability(
                False, f"no credential (set {key_var})",
                [f"set {key_var}=... in .env, then RTI_ANALYSER__VLM_PROVIDER=openai"],
            )
        return Availability(True, f"ready — {self.model_id} @ {self.endpoint}")

    def _get_client(self) -> object:
        if self._client is None:
            import os

            import openai

            key = os.getenv(self.cfg.openai_api_key_env)
            if not key:
                raise RuntimeError(f"{self.cfg.openai_api_key_env} is not set")
            kwargs: dict = {"api_key": key, "timeout": self.cfg.request_timeout_s}
            if self.cfg.openai_base_url:
                kwargs["base_url"] = self.cfg.openai_base_url
            self._client = openai.AsyncOpenAI(**kwargs)
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
