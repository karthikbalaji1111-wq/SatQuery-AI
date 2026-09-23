"""SAR agent dispatch and claim-bound deterministic numeric authority."""
from datetime import UTC, datetime

import pytest
from app.services.agent.executor import _analysis_items
from app.services.agent.prompts import _system_instruction
from app.services.agent.schemas import AgentPlan
from app.services.analysis.schemas import AnalysisResult, SarBackscatterResult
from pydantic import ValidationError

from tests.test_agent_executor import FakeAnalysisService, make_execution_result, make_plan, run
from tests.test_grounding_hardening import accepted, measured, validate


def sar_item(name="vv_mean_db", value=-8.25, unit="dB", **kwargs):
    return measured(name, value, unit, source="sar_backscatter", **kwargs)


@pytest.mark.parametrize("text,name,value,unit", [
    ("The mean VV was -8.25 dB.", "vv_mean_db", -8.25, "dB"),
    ("VH averaged -12.5 dB.", "vh_mean_db", -12.5, "dB"),
    ("The VV minus VH difference was 4.25 dB.", "vv_minus_vh_mean_db", 4.25, "dB"),
    ("The minimum VV was -20 dB.", "vv_min_db", -20, "dB"),
    ("VV valid pixel count was 30 pixels.", "vv_valid_pixel_count", 30, "pixels"),
])
def test_sar_correct_identity_survives(text, name, value, unit):
    assert accepted(validate(text, [sar_item(name, value, unit)]))


@pytest.mark.parametrize("text", [
    "The mean VH was -8.25 dB.",
    "The maximum VV was -8.25 dB.",
    "The mean VV was -8.25 index.",
    # A length unit against decibel evidence. Named explicitly because a
    # backscatter figure misread as a distance is the kind of unit error a
    # reader would not catch from the sentence alone.
    "The mean VV was -8.25 meters.",
    "The mean VV was -8.25 m.",
    "The mean NDVI was -8.25 index.",
    "The mean VV was -.987 dB.",
    "The mean was -8.25 dB.",
    "The Sentinel-2 mean VV was -8.25 dB.",
    "The mean VV was -8.25 dB, proving flooding.",
])
def test_sar_wrong_identity_or_unsupported_claim_fails(text):
    assert not accepted(validate(text, [sar_item()]))


def test_sar_citation_and_authority_are_not_optional():
    text = "The mean VV was -8.25 dB."
    assert not accepted(validate(text, []))
    assert not accepted(validate(text, [sar_item()], refs=[]))
    assert not accepted(validate(text, [sar_item()], refs=["sar_backscatter.missing"]))
    assert not accepted(validate(text, [measured("vv_mean_db", -8.25, "dB", "model")]))
    assert not accepted(validate(text, [measured("ndwi_mean", -8.25, "index")]))
    assert not accepted(validate(text, [sar_item(item_id="ndwi.vv_mean_db")]))


def sar_result():
    measurement = sar_item().measurement
    return AnalysisResult(status="ok", task="visualize", answer="", windows_considered=[],
        measurements=[measurement], sar_backscatter=SarBackscatterResult(
            scene_id="S1A_IW_20250115", window_label="single",
            acquired_at=datetime(2025, 1, 15, tzinfo=UTC),
            collection="sentinel-1-rtc", measurements=[measurement],
        ))


def test_sar_evidence_is_separate_and_scene_date_bound():
    result = sar_result()
    items = _analysis_items(result)
    assert [i.id for i in items if i.measurement] == ["sar_backscatter.vv_mean_db"]
    assert accepted(validate("Mean VV on 2025-01-15 was -8.25 dB.", items, analysis=result))
    assert not accepted(validate("Mean VV on 2025-01-16 was -8.25 dB.", items, analysis=result))
    assert accepted(validate("Mean VV for S1A_IW_20250115 was -8.25 dB.", items, analysis=result))
    assert not accepted(
        validate("Mean VV for S1A_IW_20250116 was -8.25 dB.", items, analysis=result)
    )


def test_sar_tool_coalesces_analysis_and_passes_polarization():
    intent = make_execution_result().plan.intent.model_copy(
        update={"modalities": ["sentinel-1-sar"]}
    )
    plan = make_plan({"tool":"execute_query", "intent":intent.model_dump(mode="json"),
                      "sar_polarization":"vh"}, {"tool":"sar_backscatter_statistics"})
    outcome, query, analysis = run(plan, analysis=FakeAnalysisService(result=sar_result()))
    assert len(analysis.calls) == 1
    assert analysis.calls[0].include_sar_backscatter
    assert not analysis.calls[0].include_ndwi
    assert query.calls[0].sar_polarization == "vh"
    assert query.calls[0].include_imagery
    assert all(step.status == "ok" for step in outcome.steps)
    assert any(item.source == "sar_backscatter" for item in outcome.evidence.items)
    with pytest.raises(ValidationError):
        AgentPlan.model_validate({"steps": [plan.steps[0].model_dump(),
            {"tool":"sar_backscatter_statistics", "calibration": "invented"}]})


def test_undated_policy_is_bounded_and_sar_is_not_optical():
    prompt = _system_instruction()
    assert "If no dates are supplied" in prompt
    assert datetime.now(UTC).date().isoformat() in prompt
    assert "sar_backscatter_statistics" in prompt
    assert "Never apply this default over explicit dates" in prompt
