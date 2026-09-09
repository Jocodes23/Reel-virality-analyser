"""Tests for the swappable VLM layer (no provider SDK or network required)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from reels_trend_intel.analyser.types import (
    AnalysisStatus,
    Archetype,
    GradeClass,
    ShotClass,
    VLMAnalysis,
)
from reels_trend_intel.analyser.vlm import (
    PROVIDERS,
    availability_report,
    make_vlm_adapter,
)
from reels_trend_intel.analyser.vlm.base import (
    VLMAdapter,
    VLMUsage,
    analyse_with_retry,
    price,
)
from reels_trend_intel.analyser.vlm.local_vlm import extract_json
from reels_trend_intel.config.settings import AnalyserConfig

VALID = {
    "shot_class": "polished_handheld",
    "shot_description": "Handheld mid-shots with a gentle push-in. Warm practical "
                        "lighting, shallow depth of field, cuts on motion.",
    "shot_attributes": {
        "camera_movement": "handheld", "framing": "medium", "lighting": "natural",
        "stabilisation_quality": 0.7, "angle_variety": 0.5,
    },
    "shot_confidence": 0.82,
    "grade_class": "graded_subtle",
    "grade_description": "Mild warm grade with lifted blacks, consistent across frames.",
    "grade_confidence": 0.66,
    "archetype": "food",
    "archetype_confidence": 0.9,
    "text_present": True,
    "text_position": "bottom",
    "text_style": "bold_outline",
    "text_density": 0.4,
    "structure": "montage",
}


class _FakeAdapter(VLMAdapter):
    """Deterministic stand-in: fails `fail_times` then returns a valid analysis."""

    provider = "fake"

    def __init__(self, cfg: AnalyserConfig, fail_times: int = 0) -> None:
        super().__init__(cfg)
        self.fail_times = fail_times
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "fake-vlm-1"

    def availability(self):  # type: ignore[no-untyped-def]
        from reels_trend_intel.analyser.vlm.base import Availability

        return Availability(True, "fake ready")

    async def analyse(self, montage_png, *, duration_s=None, cut_count=None):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ValueError("simulated malformed JSON")
        return VLMAnalysis.model_validate(VALID), VLMUsage(1200, 300)


# --- schema -----------------------------------------------------------------
def test_schema_accepts_valid_payload():
    a = VLMAnalysis.model_validate(VALID)
    assert a.shot_class is ShotClass.POLISHED_HANDHELD
    assert a.archetype is Archetype.FOOD
    assert a.grade_class is GradeClass.GRADED_SUBTLE
    assert 0.0 <= a.shot_confidence <= 1.0


def test_schema_rejects_bad_enum_and_out_of_range():
    with pytest.raises(ValidationError):
        VLMAnalysis.model_validate({**VALID, "shot_class": "not_a_real_class"})
    with pytest.raises(ValidationError):
        VLMAnalysis.model_validate({**VALID, "shot_confidence": 1.7})


def test_schema_forbids_extra_fields_for_strict_json():
    with pytest.raises(ValidationError):
        VLMAnalysis.model_validate({**VALID, "smuggled": "value"})
    # extra="forbid" is what emits additionalProperties:false for strict modes
    assert VLMAnalysis.model_json_schema().get("additionalProperties") is False


# --- retry / failure contract ----------------------------------------------
@pytest.mark.asyncio
async def test_succeeds_first_try_and_costs_are_computed():
    cfg = AnalyserConfig()
    res = await analyse_with_retry(_FakeAdapter(cfg), b"png", duration_s=12.0, cut_count=7)
    assert res.ok and res.status is AnalysisStatus.DONE
    assert res.attempts == 1
    assert res.analysis is not None and res.analysis.archetype is Archetype.FOOD
    assert res.input_tokens == 1200 and res.output_tokens == 300
    assert res.model_id == "fake-vlm-1" and res.prompt_version


@pytest.mark.asyncio
async def test_retries_once_then_succeeds():
    cfg = AnalyserConfig(vlm_retries=1)
    adapter = _FakeAdapter(cfg, fail_times=1)
    res = await analyse_with_retry(adapter, b"png")
    assert res.ok and res.attempts == 2 and adapter.calls == 2


@pytest.mark.asyncio
async def test_marks_failed_instead_of_raising():
    """A bad reel must never take down the batch."""
    cfg = AnalyserConfig(vlm_retries=1)
    adapter = _FakeAdapter(cfg, fail_times=99)
    res = await analyse_with_retry(adapter, b"png")
    assert res.status is AnalysisStatus.FAILED
    assert res.analysis is None
    assert adapter.calls == 2          # initial attempt + one retry, then stop
    assert "simulated malformed JSON" in (res.error or "")


# --- the toggle -------------------------------------------------------------
def test_factory_builds_each_provider_without_sdks_installed():
    cfg = AnalyserConfig()
    for name in PROVIDERS:
        adapter = make_vlm_adapter(cfg, override=name)
        assert adapter.provider == name
        assert isinstance(adapter.model_id, str) and adapter.model_id


def test_factory_honours_config_and_override():
    assert make_vlm_adapter(AnalyserConfig(vlm_provider="openai")).provider == "openai"
    cfg = AnalyserConfig(vlm_provider="anthropic")
    assert make_vlm_adapter(cfg, override="local").provider == "local"


def test_factory_rejects_unknown_provider():
    with pytest.raises(ValueError, match="unknown VLM provider"):
        make_vlm_adapter(AnalyserConfig(), override="gemini")


def test_availability_report_never_raises():
    report = availability_report(AnalyserConfig())
    assert set(report) == set(PROVIDERS)
    for av in report.values():
        assert isinstance(av.ready, bool) and av.detail


# --- pricing + local JSON extraction ---------------------------------------
def test_price_matches_published_rates():
    # 1M input @ $5 + 1M output @ $25
    assert price(VLMUsage(1_000_000, 1_000_000), 5.0, 25.0) == 30.0
    assert price(VLMUsage(0, 0), 5.0, 25.0) == 0.0


def test_local_adapter_is_free():
    from reels_trend_intel.analyser.vlm.local_vlm import LocalVLMAdapter

    assert LocalVLMAdapter(AnalyserConfig()).cost_usd(VLMUsage(9999, 9999)) == 0.0


@pytest.mark.parametrize("raw", [
    json.dumps(VALID),
    "```json\n" + json.dumps(VALID) + "\n```",
    "Here is the analysis:\n" + json.dumps(VALID) + "\nHope that helps!",
])
def test_extract_json_survives_local_model_chattiness(raw):
    assert VLMAnalysis.model_validate(extract_json(raw)).archetype is Archetype.FOOD


def test_extract_json_raises_when_absent():
    with pytest.raises(ValueError, match="no JSON object"):
        extract_json("I cannot analyse this image.")
