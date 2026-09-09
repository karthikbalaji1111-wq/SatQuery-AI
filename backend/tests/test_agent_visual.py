"""Phase 18.1 - single-image vision-language reasoning.

The first capability that looks at the PICTURE rather than the numbers, and
therefore the first whose output cannot be mechanically checked. Three
properties are load-bearing here and every test below exists to pin one of them:

1. The server chooses the image. The model asks a question; it never names a
   scene, a URL or a byte. Imagery comes only from a validated ``execute_query``
   result that already ran.
2. A model claim is ATTRIBUTED, never validated. There is no evidence a visual
   statement could be contained by, so it is labelled as an observation of a
   named model rather than dressed as a verified fact.
3. A model may never authorise a number. ``_allowed_values`` feeds numeric
   grounding; if model evidence reached it, a VLM could mint a measurement and
   then cite itself. That is the sharpest edge in this phase.

No test contacts Gemini.
"""

from __future__ import annotations

import asyncio
import base64
from typing import Any

import pytest
from app.core.errors import InvalidInputError, UpstreamServiceError
from app.services.agent.executor import AgentExecutor
from app.services.agent.grounding import DraftAnswer, validate_answer
from app.services.agent.registry import REGISTERED_TOOLS, TOOL_REGISTRY, resolve_tool
from app.services.agent.schemas import (
    AgentEvidence,
    AgentPlan,
    EvidenceItem,
    ExecuteQueryParams,
    NdwiParams,
    RsModelParams,
    VisualObservation,
)
from app.services.agent.visual import MockVisualAnalyst, VisualAnalyst, VisualAnswer
from app.services.geospatial.schemas import BoundingBox
from app.services.query.schemas import (
    ExecutedWindow,
    QueryExecutionResult,
    ResolvedQueryPlan,
    SatQueryIntent,
    TimeRange,
)
from app.services.satellite.schemas import ImageryResponse, Scene, WindowInfo
from pydantic import ValidationError

BBOX = BoundingBox(west=80.10, south=12.90, east=80.30, north=13.20)
S2 = "sentinel-2-optical"
S1 = "sentinel-1-sar"
PNG_BYTES = b"\x89PNG\r\n\x1a\n-fake-but-nonempty"
PNG_B64 = base64.b64encode(PNG_BYTES).decode("ascii")


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def imagery(
    *, asset: str = "visual", bands: list[str] | None = None, b64: str = PNG_B64
) -> ImageryResponse:
    return ImageryResponse(
        scene_id="S2B_44PMV_20250104_0_L2A",
        bbox=BBOX,
        asset=asset,
        asset_href="https://example.test/TCI.tif",
        width=112,
        height=300,
        format="png",
        media_type="image/png",
        bands=bands if bands is not None else ["red", "green", "blue"],
        crs="EPSG:32644",
        resolution=10.0,
        normalization="none (source is 8-bit RGB)",
        window=WindowInfo(col_off=0, row_off=0, width=112, height=300),
        source_shape=[10980, 10980],
        transform=[10.0, 0.0, 421900.0, 0.0, -10.0, 1444560.0],
        corners_wgs84=[[80.28, 13.07], [80.29, 13.07], [80.29, 13.04], [80.28, 13.04]],
        image_base64=b64,
    )


def window(
    *,
    modality: str = S2,
    label: str = "single",
    scene_id: str = "S2B_44PMV_20250104_0_L2A",
    img: ImageryResponse | None = None,
) -> ExecutedWindow:
    scene = Scene(
        id=scene_id,
        datetime="2025-01-04T05:15:13Z",
        bbox=BBOX,
        geometry=None,
        cloud_cover=1.0,
        collection="sentinel-2-l2a",
        platform=None,
        processing_level="L2A",
        thumbnail_url=None,
        assets=[],
    )
    return ExecutedWindow(
        modality=modality,  # type: ignore[arg-type]
        label=label,
        time_range=TimeRange.model_validate(
            {"start_date": "2025-01-01", "end_date": "2025-01-31"}
        ),
        scene_count=1,
        scenes=[scene],
        selected_scene_id=scene.id,
        imagery=img,
        imagery_error=None,
    )


def execution(windows: list[ExecutedWindow] | None = None) -> QueryExecutionResult:
    intent = SatQueryIntent.model_validate(
        {
            "location_query": "Marina Beach, Chennai",
            "temporal_mode": "single",
            "time_windows": [{"start_date": "2025-01-01", "end_date": "2025-01-31"}],
            "modalities": [S2],
            "task": "visualize",
        }
    )
    return QueryExecutionResult(
        plan=ResolvedQueryPlan(intent=intent, bbox=BBOX),
        executed_modalities=[S2],
        skipped_modalities=[],
        windows=windows if windows is not None else [window(img=imagery())],
        catalog="https://earth-search.aws.element84.com/v1",
    )


class RecordingAnalyst(VisualAnalyst):
    """Records exactly what the executor hands the provider boundary."""

    model_name = "fake-vlm/1"

    def __init__(self, answer: str = "Water is visible.", error: Exception | None = None):
        self.calls: list[dict[str, Any]] = []
        self._answer = answer
        self._error = error

    async def observe(
        self, *, question: str, image: bytes, media_type: str, model_hint: str | None = None
    ) -> VisualAnswer:
        self.calls.append(
            {
                "question": question,
                "image": image,
                "media_type": media_type,
                "model_hint": model_hint,
            }
        )
        if self._error is not None:
            raise self._error
        return VisualAnswer(answer=self._answer)


class FakeQuery:
    def __init__(self, result: QueryExecutionResult | None = None):
        self._result = result if result is not None else execution()
        self.calls = 0

    async def execute(self, request: Any) -> QueryExecutionResult:
        self.calls += 1
        return self._result


class FakeAnalysis:
    async def analyze(self, request: Any) -> Any:  # pragma: no cover
        raise AssertionError("visual questions must not invoke the analysis service")


def build_executor(
    *, analyst: VisualAnalyst | None = None, query: FakeQuery | None = None
) -> tuple[AgentExecutor, FakeQuery, RecordingAnalyst]:
    analyst = analyst or RecordingAnalyst()
    query = query or FakeQuery()
    ex = AgentExecutor(
        query_execution_service=query,  # type: ignore[arg-type]
        analysis_service=FakeAnalysis(),  # type: ignore[arg-type]
        visual_analyst=analyst,
    )
    return ex, query, analyst  # type: ignore[return-value]


def visual_plan(question: str = "Is there visible water?") -> AgentPlan:
    return AgentPlan(
        steps=[
            ExecuteQueryParams(
                tool="execute_query",
                intent=SatQueryIntent.model_validate(
                    {
                        "location_query": "Marina Beach, Chennai",
                        "temporal_mode": "single",
                        "time_windows": [
                            {"start_date": "2025-01-01", "end_date": "2025-01-31"}
                        ],
                        "modalities": [S2],
                        "task": "visualize",
                    }
                ),
                include_imagery=True,
            ),
            RsModelParams(tool="rs_model_analysis", question=question),
        ]
    )


def run(ex: AgentExecutor, plan: AgentPlan) -> Any:
    return asyncio.run(ex.execute(plan))


# =========================================================================== #
# A. Tool registration - the allowlist stays closed
# =========================================================================== #


def test_the_visual_tool_is_registered() -> None:
    assert "rs_model_analysis" in REGISTERED_TOOLS
    assert resolve_tool("rs_model_analysis").operation == "visual"


def test_the_registry_is_still_exactly_the_approved_tools() -> None:
    """The allowlist is CLOSED. Its size may grow; its membership is pinned."""

    assert set(REGISTERED_TOOLS) == {
        "execute_query",
        "spectral_indices",
        "ndwi_statistics",
        "temporal_ndwi_statistics",
        "rs_model_analysis",
    }


def test_an_unknown_tool_is_still_rejected() -> None:
    with pytest.raises(InvalidInputError):
        resolve_tool("segment_everything")


def test_the_visual_spec_holds_no_callable() -> None:
    spec = TOOL_REGISTRY["rs_model_analysis"]
    assert not any(callable(getattr(spec, f)) for f in spec.__dataclass_fields__)


def test_the_visual_tool_has_no_analysis_flag() -> None:
    """It is not an analysis tool; it must never set an AnalysisRequest flag."""

    assert TOOL_REGISTRY["rs_model_analysis"].analysis_flag is None


# --- the parameter contract is closed and carries no imagery --------------- #


def test_the_question_is_the_only_parameter() -> None:
    assert set(RsModelParams.model_fields) == {"tool", "question"}


@pytest.mark.parametrize(
    "field", ["image_base64", "scene_id", "asset_href", "url", "path", "image"]
)
def test_imagery_cannot_be_supplied_by_the_model(field: str) -> None:
    """The server picks the image; a plan may not name one."""

    with pytest.raises(ValidationError):
        RsModelParams.model_validate(
            {"tool": "rs_model_analysis", "question": "q", field: "x"}
        )


def test_an_empty_question_is_refused() -> None:
    with pytest.raises(ValidationError):
        RsModelParams(tool="rs_model_analysis", question="")


# =========================================================================== #
# B. Image selection - server-authoritative, S2 visual only
# =========================================================================== #


def test_sentinel_2_visual_imagery_is_accepted() -> None:
    ex, _, analyst = build_executor()
    outcome = run(ex, visual_plan())

    assert len(analyst.calls) == 1
    assert [s.status for s in outcome.steps] == ["ok", "ok"]


def test_sentinel_1_imagery_is_refused() -> None:
    ex, _, analyst = build_executor(
        query=FakeQuery(
            execution([window(modality=S1, img=imagery(asset="vv", bands=["vv", "vv", "vv"]))])
        )
    )
    outcome = run(ex, visual_plan())

    assert analyst.calls == []
    assert outcome.steps[1].status == "rejected"
    assert "sentinel-2" in (outcome.steps[1].rejection_reason or "").lower()


def test_a_non_visual_asset_is_refused() -> None:
    ex, _, analyst = build_executor(
        query=FakeQuery(execution([window(img=imagery(asset="green", bands=["green"]))]))
    )
    outcome = run(ex, visual_plan())

    assert analyst.calls == []
    assert outcome.steps[1].status == "rejected"


def test_missing_imagery_is_refused() -> None:
    ex, _, analyst = build_executor(query=FakeQuery(execution([window(img=None)])))
    outcome = run(ex, visual_plan())

    assert analyst.calls == []
    assert outcome.steps[1].status == "rejected"
    assert "image" in (outcome.steps[1].rejection_reason or "").lower()


def test_empty_image_bytes_are_refused() -> None:
    ex, _, analyst = build_executor(
        query=FakeQuery(execution([window(img=imagery(b64=""))]))
    )
    outcome = run(ex, visual_plan())

    assert analyst.calls == []
    assert outcome.steps[1].status == "rejected"


def test_a_visual_only_plan_is_unrepresentable() -> None:
    """Stronger than an executor refusal: the plan cannot even be built.

    The Phase 15 contract already requires exactly one ``execute_query``, so a
    plan that asks to look at an image without retrieving one fails validation
    before the executor is reached. The executor's own check (below) is
    defence-in-depth for the case where discovery ran and produced nothing.
    """

    with pytest.raises(ValidationError):
        AgentPlan(steps=[RsModelParams(tool="rs_model_analysis", question="q")])


def test_the_visual_tool_is_refused_when_discovery_produced_nothing() -> None:
    """Enforced by the executor, not by trusting the planner."""

    ex, _, analyst = build_executor(query=FakeQuery(execution([])))
    outcome = run(ex, visual_plan())

    assert analyst.calls == []
    assert outcome.steps[1].status == "rejected"


# =========================================================================== #
# C. Provider request - the bytes, and nothing else
# =========================================================================== #


def test_exactly_the_existing_png_bytes_are_sent() -> None:
    ex, query, analyst = build_executor()
    run(ex, visual_plan())

    assert analyst.calls[0]["image"] == PNG_BYTES
    assert analyst.calls[0]["media_type"] == "image/png"
    assert query.calls == 1  # no second retrieval


def test_the_question_reaches_the_provider_verbatim() -> None:
    ex, _, analyst = build_executor()
    run(ex, visual_plan("Is dense vegetation visible?"))

    assert analyst.calls[0]["question"] == "Is dense vegetation visible?"


def test_no_georeferencing_or_measurement_metadata_is_sent() -> None:
    """The model must have to look at the image, not read the answer off metadata."""

    ex, _, analyst = build_executor()
    run(ex, visual_plan())
    sent = " ".join(
        str(v) for k, v in analyst.calls[0].items() if k != "image"
    ).lower()

    for leaked in (
        "epsg",
        "421900",
        "corners",
        "transform",
        "ndwi",
        "80.1",
        "asset_href",
        "https://",
        "s2b_44pmv",
    ):
        assert leaked not in sent, f"metadata leaked to the VLM: {leaked}"


# =========================================================================== #
# D. Provider response
# =========================================================================== #


def test_a_valid_structured_answer_is_accepted() -> None:
    ex, _, _ = build_executor(analyst=RecordingAnalyst(answer="Water is visible."))
    outcome = run(ex, visual_plan())

    visual = [i for i in outcome.evidence.items if i.source == "model"]
    assert len(visual) == 1
    assert visual[0].visual is not None
    assert visual[0].visual.statement == "Water is visible."


def test_an_empty_answer_is_refused_by_the_contract() -> None:
    with pytest.raises(ValidationError):
        VisualAnswer(answer="")


def test_a_provider_failure_produces_no_evidence() -> None:
    ex, _, _ = build_executor(
        analyst=RecordingAnalyst(error=UpstreamServiceError("provider down"))
    )
    outcome = run(ex, visual_plan())

    assert [i for i in outcome.evidence.items if i.source == "model"] == []
    assert outcome.steps[1].status == "failed"


def test_a_provider_failure_preserves_the_deterministic_evidence() -> None:
    ex, _, _ = build_executor(
        analyst=RecordingAnalyst(error=UpstreamServiceError("provider down"))
    )
    outcome = run(ex, visual_plan())

    assert outcome.evidence.execution is not None
    assert any(i.source == "execution" for i in outcome.evidence.items)


# =========================================================================== #
# E. Evidence - attributed, distinct, never a measurement
# =========================================================================== #


def test_visual_evidence_uses_the_model_source() -> None:
    ex, _, _ = build_executor()
    item = [i for i in run(ex, visual_plan()).evidence.items if i.source == "model"][0]

    assert item.source == "model"


def test_visual_evidence_is_never_a_measurement() -> None:
    ex, _, _ = build_executor()
    item = [i for i in run(ex, visual_plan()).evidence.items if i.source == "model"][0]

    assert item.measurement is None
    assert item.visual is not None


def test_visual_evidence_records_the_model_identity() -> None:
    ex, _, _ = build_executor()
    item = [i for i in run(ex, visual_plan()).evidence.items if i.source == "model"][0]

    assert item.produced_by == "fake-vlm/1"
    assert item.visual is not None
    assert item.visual.model == "fake-vlm/1"
    assert item.visual.scene_id == "S2B_44PMV_20250104_0_L2A"


def test_visual_evidence_ids_are_namespaced_and_unique() -> None:
    ex, _, _ = build_executor()
    outcome = run(ex, visual_plan())
    ids = [i.id for i in outcome.evidence.items]

    assert len(ids) == len(set(ids))
    visual_ids = [i for i in ids if i.startswith("model.")]
    assert visual_ids == ["model.visual.S2B_44PMV_20250104_0_L2A"]


def test_a_visual_observation_cannot_be_empty() -> None:
    with pytest.raises(ValidationError):
        VisualObservation(statement="", model="m", provider="p", scene_id="s")


# =========================================================================== #
# F. Grounding security - a model may never authorise a number
# =========================================================================== #


def deterministic_item(value: float) -> EvidenceItem:
    return EvidenceItem(
        id="ndwi.ndwi_mean",
        source="ndwi",
        measurement={"name": "ndwi_mean", "value": value, "unit": "index"},  # type: ignore[arg-type]
        produced_by="analysis.engines.compute_ndwi_measurements",
    )


def model_item(statement: str) -> EvidenceItem:
    return EvidenceItem(
        id="model.visual.SCENE",
        source="model",
        visual=VisualObservation(
            statement=statement, model="fake-vlm/1", provider="fake", scene_id="SCENE"
        ),
        produced_by="fake-vlm/1",
    )


def test_a_model_statement_cannot_authorise_its_own_number() -> None:
    """The sharpest edge in this phase, pinned directly."""

    evidence = AgentEvidence(items=[model_item("Water covers approximately 42 percent.")])
    result = validate_answer(
        DraftAnswer(summary="Water covers 42 percent.", evidence_refs=["model.visual.SCENE"]),
        evidence,
    )

    assert result.numeric_grounding == "fail"


def test_deterministic_evidence_still_authorises_a_number() -> None:
    evidence = AgentEvidence(items=[deterministic_item(42.0)])
    result = validate_answer(
        DraftAnswer(summary="The mean was 42.", evidence_refs=["ndwi.ndwi_mean"]), evidence
    )

    assert result.numeric_grounding == "pass"


def test_model_evidence_beside_deterministic_evidence_adds_no_values() -> None:
    evidence = AgentEvidence(
        items=[deterministic_item(42.0), model_item("It looks like about 99 percent.")]
    )
    result = validate_answer(
        DraftAnswer(summary="Coverage was 99 percent.", evidence_refs=["ndwi.ndwi_mean"]),
        evidence,
    )

    assert result.numeric_grounding == "fail"


def test_a_purely_visual_claim_is_not_reported_as_numerically_grounded() -> None:
    evidence = AgentEvidence(items=[model_item("Water is visible along the shoreline.")])
    result = validate_answer(
        DraftAnswer(
            summary="Water is visible along the shoreline.",
            evidence_refs=["model.visual.SCENE"],
        ),
        evidence,
    )

    assert result.visual_claims == "attributed"
    assert result.numeric_grounding == "pass"  # vacuous: no numbers were stated


def test_visual_claims_is_not_run_without_model_evidence() -> None:
    evidence = AgentEvidence(items=[deterministic_item(42.0)])
    result = validate_answer(
        DraftAnswer(summary="The mean was 42.", evidence_refs=["ndwi.ndwi_mean"]), evidence
    )

    assert result.visual_claims == "not_run"


# =========================================================================== #
# G. Planner / mock
# =========================================================================== #


def test_the_mock_analyst_is_flagged_as_a_mock() -> None:
    assert getattr(MockVisualAnalyst, "is_mock", False) is True


def test_the_mock_analyst_never_inspects_the_image() -> None:
    answer = asyncio.run(
        MockVisualAnalyst().observe(
            question="q", image=PNG_BYTES, media_type="image/png"
        )
    )
    assert isinstance(answer, VisualAnswer)
    assert answer.answer


# =========================================================================== #
# H. Imagery is a precondition of the visual tool, not a planner preference
# =========================================================================== #
#
# Observed live: for a plainly visual question the planner sometimes emitted
# ``include_imagery=False`` and then asked to look at the image, so the executor
# correctly refused and the run produced nothing useful. The prompt is not the
# place to fix that - a prompt is a request, and this is an invariant.
#
# A validated plan therefore cannot express the combination at all. The
# executor's own check stays exactly as it was: this makes the bad plan
# unrepresentable, it does not make the image guaranteed to exist.


def plan_with(include_imagery: bool, *, visual: bool = True) -> AgentPlan:
    intent = SatQueryIntent.model_validate(
        {
            "location_query": "Marina Beach, Chennai",
            "temporal_mode": "single",
            "time_windows": [{"start_date": "2025-01-01", "end_date": "2025-01-31"}],
            "modalities": [S2],
            "task": "visualize",
        }
    )
    steps: list[Any] = [
        ExecuteQueryParams(
            tool="execute_query", intent=intent, include_imagery=include_imagery
        )
    ]
    if visual:
        steps.append(RsModelParams(tool="rs_model_analysis", question="Water?"))
    return AgentPlan(steps=steps)


def test_a_visual_plan_always_ends_up_requesting_imagery() -> None:
    """The invariant, stated directly."""

    plan = plan_with(include_imagery=False)
    execute = plan.steps[0]

    assert isinstance(execute, ExecuteQueryParams)
    assert execute.include_imagery is True


def test_a_visual_plan_that_already_requested_imagery_is_unchanged() -> None:
    execute = plan_with(include_imagery=True).steps[0]

    assert isinstance(execute, ExecuteQueryParams)
    assert execute.include_imagery is True


def test_the_invariant_holds_however_the_plan_was_built() -> None:
    """Also when parsed from raw model output, which is the real path."""

    plan = AgentPlan.model_validate(
        {
            "steps": [
                {
                    "tool": "execute_query",
                    "intent": {
                        "location_query": "Chennai",
                        "temporal_mode": "single",
                        "time_windows": [
                            {"start_date": "2025-01-01", "end_date": "2025-01-31"}
                        ],
                        "modalities": [S2],
                        "task": "visualize",
                    },
                    "include_imagery": False,
                },
                {"tool": "rs_model_analysis", "question": "Is water visible?"},
            ]
        }
    )

    assert plan.steps[0].include_imagery is True  # type: ignore[union-attr]


def test_a_non_visual_plan_does_not_gain_imagery() -> None:
    """Deterministic questions must not start paying for a picture."""

    execute = plan_with(include_imagery=False, visual=False).steps[0]

    assert isinstance(execute, ExecuteQueryParams)
    assert execute.include_imagery is False


def test_a_non_visual_plan_may_still_request_imagery_explicitly() -> None:
    """Wanting to SEE a scene is a separate, legitimate request."""

    execute = plan_with(include_imagery=True, visual=False).steps[0]

    assert isinstance(execute, ExecuteQueryParams)
    assert execute.include_imagery is True


def test_an_ndwi_plan_requests_imagery_for_display() -> None:
    intent = SatQueryIntent.model_validate(
        {
            "location_query": "Chennai",
            "temporal_mode": "single",
            "time_windows": [{"start_date": "2025-01-01", "end_date": "2025-01-31"}],
            "modalities": [S2],
            "task": "visualize",
        }
    )
    plan = AgentPlan(
        steps=[
            ExecuteQueryParams(
                tool="execute_query", intent=intent, include_imagery=False
            ),
            NdwiParams(tool="ndwi_statistics"),
        ]
    )

    assert plan.steps[0].include_imagery is True  # type: ignore[union-attr]


def test_the_executor_check_is_still_independent() -> None:
    """Defence-in-depth: a plan asking for imagery does not guarantee any.

    Discovery can legitimately return a window with no imagery, so the executor
    must still refuse rather than assume the plan's request was honoured.
    """

    ex, _, analyst = build_executor(query=FakeQuery(execution([window(img=None)])))
    outcome = run(ex, plan_with(include_imagery=True))

    assert analyst.calls == []
    assert outcome.steps[1].status == "rejected"
