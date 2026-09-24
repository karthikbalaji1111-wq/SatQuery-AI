"""M5.5 - the provider-independent natural-language workflow.

    question -> interpret() -> AgentPlan -> AgentExecutor -> StandardReport
             -> validate_answer -> AgentResult

Four layers, tested in order:

A. **Interpretation** - which analysis, where, when, which sensor, and the
   clarifications for everything the question does not state.
B. **Execution** - the interpretation becomes the EXISTING contracts: the
   ``SatQueryIntent`` reaches the geospatial resolver through the real
   ``QueryExecutionService``, and the analysis flags reach ``AnalysisService``.
C. **Report** - the answer is fixed sentences over engine values, and it passes
   the unchanged grounding validator.
D. **Provider independence** - the HTTP route answers supported questions with
   no key configured, and never builds a provider unless one is named.

Hand-written recording fakes, as elsewhere in this suite. No test contacts a
model, Nominatim, STAC or imagery.
"""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any

import pytest
from app.api.routes import query as query_routes
from app.core.errors import NotFoundError
from app.main import create_app
from app.services.agent.executor import AgentExecutor
from app.services.agent.interpretation import (
    ClarificationRequiredError,
    QueryInterpretation,
    interpret,
)
from app.services.agent.schemas import (
    AgentQuestionRequest,
    AgentResult,
    ExecuteQueryParams,
    SarBackscatterParams,
    SpectralIndicesParams,
    TemporalNdwiParams,
)
from app.services.agent.service import AgentService
from app.services.agent.standard import (
    ABSTENTION,
    StandardPlanner,
    StandardReport,
)
from app.services.analysis.schemas import (
    AnalysisRequest,
    AnalysisResult,
    Measurement,
    SarBackscatterResult,
)
from app.services.geospatial.schemas import BoundingBox
from app.services.query.execution import QueryExecutionService
from app.services.query.schemas import (
    ExecutedWindow,
    QueryExecutionRequest,
    QueryExecutionResult,
    ResolvedQueryPlan,
    SatQueryIntent,
    TemporalComparison,
    TimeRange,
)
from app.services.satellite.schemas import QueryEcho, Scene, SceneSearchResponse
from fastapi.testclient import TestClient

TODAY = date(2026, 9, 24)
BBOX = BoundingBox(west=80.27, south=13.03, east=80.29, north=13.07)
S2 = "sentinel-2-optical"
S1 = "sentinel-1-sar"


def read(question: str) -> QueryInterpretation:
    return interpret(question, today=TODAY)


def ask_for(question: str) -> Any:
    with pytest.raises(ClarificationRequiredError) as raised:
        interpret(question, today=TODAY)
    return raised.value.clarification


def window(start: str, end: str) -> TimeRange:
    return TimeRange(start_date=date.fromisoformat(start), end_date=date.fromisoformat(end))


# =========================================================================== #
# A. Interpretation
# =========================================================================== #


@pytest.mark.parametrize(
    ("question", "analyses"),
    [
        ("Show vegetation around Chennai in January 2025", ("ndvi",)),
        ("What is the NDVI of Anna Nagar, Chennai in January 2025?", ("ndvi",)),
        ("Find water around Puzhal Lake in January 2025", ("ndwi",)),
        ("What is the NDWI of Marina Beach, Chennai in January 2025?", ("ndwi",)),
        ("Show built-up areas around Hyderabad in December 2024", ("ndbi",)),
        ("Analyse the urban area around Pune in March 2024", ("ndbi",)),
        ("Analyze radar backscatter around Chennai in January 2025", ("sar_backscatter",)),
        ("Analyze SAR backscatter around Chennai in January 2025", ("sar_backscatter",)),
        ("Show Sentinel-1 VV and VH over Kochi in June 2024", ("sar_backscatter",)),
        ("Show satellite imagery of Chennai in January 2025", ("imagery",)),
    ],
)
def test_each_supported_analysis_is_routed_to_its_own_engine(
    question: str, analyses: tuple[str, ...]
) -> None:
    assert read(question).analyses == analyses


def test_vegetation_is_ndvi_and_never_ndwi() -> None:
    """Mutation target 1: NDVI routed to NDWI."""

    plan = read("Show vegetation around Chennai in January 2025").plan()

    spectral = [s for s in plan.steps if isinstance(s, SpectralIndicesParams)]
    assert [s.indices for s in spectral] == [["ndvi"]]


def test_radar_is_sentinel_1_and_never_optical() -> None:
    """Mutation target 2: SAR routed to optical."""

    interpretation = read("Analyze SAR backscatter around Chennai in January 2025")
    plan = interpretation.plan()

    assert interpretation.modalities == [S1]
    discovery = plan.steps[0]
    assert isinstance(discovery, ExecuteQueryParams)
    assert discovery.intent.modalities == [S1]
    assert [type(s) for s in plan.steps[1:]] == [SarBackscatterParams]


def test_optical_analyses_select_sentinel_2_only() -> None:
    for question in (
        "Show vegetation around Chennai in January 2025",
        "Show water around Chennai in January 2025",
        "Show built-up area around Chennai in January 2025",
    ):
        assert read(question).modalities == [S2]


def test_optical_and_radar_together_select_both_sensors() -> None:
    interpretation = read("Show NDVI and SAR backscatter around Chennai in January 2025")

    assert interpretation.analyses == ("ndvi", "sar_backscatter")
    assert interpretation.modalities == [S2, S1]
    assert len(interpretation.plan().steps) == 3


def test_a_comparison_is_two_periods_and_never_one() -> None:
    """Mutation target 3: temporal query treated as single-date."""

    interpretation = read(
        "Compare water at Marina Beach, Chennai between January and March 2025"
    )
    plan = interpretation.plan()
    discovery = plan.steps[0]

    assert interpretation.comparison is True
    assert isinstance(discovery, ExecuteQueryParams)
    assert discovery.intent.temporal_mode == "compare"
    assert discovery.intent.time_windows == TemporalComparison(
        baseline=window("2025-01-01", "2025-01-31"),
        target=window("2025-03-01", "2025-03-31"),
    )
    assert [type(s) for s in plan.steps[1:]] == [TemporalNdwiParams]


@pytest.mark.parametrize(
    "question",
    [
        "How has water changed at Marina Beach, Chennai from January 2024 to January 2025?",
        "Compare NDWI at Marina Beach, Chennai in January 2024 and January 2025",
        "Water at Marina Beach, Chennai: January 2024 vs January 2025",
    ],
)
def test_comparison_phrasings_all_reach_temporal_ndwi(question: str) -> None:
    interpretation = read(question)

    assert interpretation.analyses == ("temporal_ndwi",)
    assert interpretation.windows == (
        window("2024-01-01", "2024-01-31"),
        window("2025-01-01", "2025-01-31"),
    )


@pytest.mark.parametrize(
    ("question", "place"),
    [
        ("Show vegetation around Chennai in January 2025", "Chennai"),
        ("What is the NDWI of Marina Beach, Chennai in January 2025?", "Marina Beach, Chennai"),
        ("Show NDVI at Marina Beach in Chennai in January 2025", "Marina Beach, Chennai"),
        ("What is the NDVI of the area near Anna Nagar in 2024?", "Anna Nagar"),
        ("Is the water index high over Puzhal Lake in May 2025?", "Puzhal Lake"),
        ("NDVI around 13.05, 80.28 in January 2025", "13.05, 80.28"),
        ("Show water around Forest Hill in January 2025", "Forest Hill"),
        ("Show water in Chennai's Marina Beach in January 2025", "Chennai's Marina Beach"),
        ("What is the NDVI of vegetation in Chennai in 2024?", "Chennai"),
    ],
)
def test_the_place_is_extracted_verbatim_and_never_geocoded(question: str, place: str) -> None:
    """Mutation target 4 (location silently discarded) - and the place text,
    not coordinates: the interpreter never resolves it."""

    interpretation = read(question)
    discovery = interpretation.plan().steps[0]

    assert interpretation.location_query == place
    assert isinstance(discovery, ExecuteQueryParams)
    assert discovery.intent.location_query == place


def test_a_place_word_is_not_an_analysis_request() -> None:
    """ "Forest Hill" is a place; "forest" inside it does not ask for NDVI."""

    assert read("Show water around Forest Hill in January 2025").analyses == ("ndwi",)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("NDVI around Chennai in January 2025", window("2025-01-01", "2025-01-31")),
        ("NDVI around Chennai in Jan 2025", window("2025-01-01", "2025-01-31")),
        ("NDVI around Chennai in February 2024", window("2024-02-01", "2024-02-29")),
        ("NDVI around Chennai on 15 January 2025", window("2025-01-15", "2025-01-15")),
        ("NDVI around Chennai on January 15, 2025", window("2025-01-15", "2025-01-15")),
        ("NDVI around Chennai on 2025-01-15", window("2025-01-15", "2025-01-15")),
        ("NDVI around Chennai between January and March 2025", window("2025-01-01", "2025-03-31")),
        ("NDVI around Chennai from January 2025 to March 2025", window("2025-01-01", "2025-03-31")),
        ("NDVI around Chennai from 10 to 20 January 2025", window("2025-01-10", "2025-01-20")),
        ("NDVI around Chennai in 2024.", window("2024-01-01", "2024-12-31")),
        ("NDVI around Chennai in May 2025", window("2025-05-01", "2025-05-31")),
    ],
)
def test_explicit_dates_become_exactly_the_stated_period(
    question: str, expected: TimeRange
) -> None:
    """Mutation target 5: dates silently discarded."""

    interpretation = read(question)
    discovery = interpretation.plan().steps[0]

    assert interpretation.windows == (expected,)
    assert isinstance(discovery, ExecuteQueryParams)
    assert discovery.intent.time_windows == [expected]


def test_may_the_verb_is_not_may_the_month() -> None:
    interpretation = read("Where water may collect, show NDWI around Chennai in 2024")
    assert interpretation.windows == (window("2024-01-01", "2024-12-31"),)


@pytest.mark.parametrize(
    ("question", "reason"),
    [
        ("Show vegetation around Chennai", "date_missing"),
        ("Show vegetation around Chennai in January", "date_ambiguous"),
        ("Show water around Chennai last month", "date_ambiguous"),
        ("Show water around Chennai today", "date_ambiguous"),
        ("Show water around Chennai during monsoon 2024", "date_ambiguous"),
        ("Show water around Chennai in early 2024", "date_ambiguous"),
        ("Show NDVI around Chennai in January 2024 and March 2025", "date_ambiguous"),
        ("Show NDVI around Chennai in December 2026", "date_invalid"),
        ("Show NDVI around Chennai in March 2010", "date_invalid"),
        ("Show NDVI around Chennai on 31 February 2025", "date_invalid"),
        ("Compare water at Marina Beach, Chennai in January 2025", "comparison_incomplete"),
        ("Compare water at Marina Beach, Chennai between January 2025 and 2025", "date_invalid"),
    ],
)
def test_no_date_is_invented(question: str, reason: str) -> None:
    """Never today, never a default window, never a season's boundaries."""

    assert ask_for(question).reason == reason


def test_a_missing_date_keeps_what_was_understood() -> None:
    clarification = ask_for("Show vegetation around Chennai")

    assert clarification.understood_location == "Chennai"
    assert clarification.understood_analyses == ["vegetation (NDVI)"]
    assert "January 2025" in clarification.message  # an example, not an assumption


@pytest.mark.parametrize(
    ("question", "reason"),
    [
        ("I want to know something about Chennai", "analysis_missing"),
        ("Analyze this", "analysis_missing"),
        ("Tell me about Chennai in January 2025", "analysis_missing"),
        ("Show change", "analysis_missing"),
        ("What is NDWI?", "analysis_missing"),
        ("Find water around this reservoir in January 2025", "location_missing"),
        ("Show NDVI in January 2025", "location_missing"),
        # Two separate names and nothing deciding between them: asked back.
        # ("Chennai NDVI January 2025" names ONE place and now resolves.)
        ("Chennai NDVI Pune January 2025", "location_missing"),
    ],
)
def test_an_unstated_analysis_or_place_is_asked_for(question: str, reason: str) -> None:
    assert ask_for(question).reason == reason


def test_the_analysis_question_offers_only_supported_analyses() -> None:
    clarification = ask_for("I want to know something about Chennai")

    assert clarification.understood_location == "Chennai"
    assert clarification.options == [
        "vegetation (NDVI)",
        "water (NDWI)",
        "built-up area (NDBI)",
        "SAR backscatter (Sentinel-1 VV/VH)",
        "water change between two periods (NDWI)",
    ]


def test_show_change_asks_what_to_compare() -> None:
    clarification = ask_for("Show change")

    assert clarification.message.startswith("What should be compared?")


def test_a_deictic_place_is_named_back_to_the_user() -> None:
    clarification = ask_for("Find water around this reservoir in January 2025")

    assert "'this reservoir'" in clarification.message
    assert clarification.understood_analyses == ["water (NDWI)"]


@pytest.mark.parametrize(
    "question",
    [
        "Show flooding in Chennai in December 2023",
        "Detect ships near Chennai port in January 2025",
        "Count the buildings in Anna Nagar in January 2025",
        "Classify land cover around Chennai in January 2025",
        "Identify objects around Chennai in January 2025",
        "Show vegetation change around Chennai between January 2024 and January 2025",
        "How has urban area grown in Chennai between 2020 and 2024?",
        "Compare SAR backscatter around Chennai in January 2024 and January 2025",
    ],
)
def test_an_unsupported_request_is_refused_not_approximated(question: str) -> None:
    """Mutation target 6 (part 1): an unsupported intent is never executed."""

    assert ask_for(question).reason == "analysis_unsupported"


def test_a_supported_analysis_beside_an_unsupported_one_is_still_refused() -> None:
    """Running the supported half would answer a different question."""

    assert ask_for(
        "Show NDVI and count the ships near Chennai in January 2025"
    ).reason == "analysis_unsupported"


def test_detect_beside_a_supported_analysis_is_a_measurement() -> None:
    """ "Detect water" can be answered with NDWI statistics - reported as index
    statistics, never as a detection."""

    assert read("Detect water around Chennai in January 2025").analyses == ("ndwi",)


def test_a_visual_question_needs_an_ai_model() -> None:
    clarification = ask_for("Is there visible water at Marina Beach, Chennai in January 2025?")

    assert clarification.reason == "requires_ai_model"
    assert "AI model" in clarification.message


@pytest.mark.parametrize(
    "question",
    [
        "Show vegetation using radar around Chennai in January 2025",
        "Show NDWI from Sentinel-1 around Chennai in January 2025",
        "Show SAR backscatter from Sentinel-2 around Chennai in January 2025",
    ],
)
def test_a_sensor_that_cannot_produce_the_analysis_is_questioned(question: str) -> None:
    assert ask_for(question).reason == "conflicting_request"


def test_a_refused_index_is_not_computed() -> None:
    interpretation = read(
        "Show NDVI and NDBI of Anna Nagar, Chennai in January 2025, not NDWI"
    )

    assert interpretation.analyses == ("ndvi", "ndbi")


def test_every_interpretation_validates_as_the_existing_contracts() -> None:
    """The plan is the SAME typed plan an AI planner must propose."""

    for question in (
        "Show vegetation around Chennai in January 2025",
        "Analyze SAR backscatter around Chennai in January 2025",
        "Compare water at Marina Beach, Chennai between January and March 2025",
        "Show NDVI and SAR backscatter around Chennai in January 2025",
        "Show satellite imagery of Chennai in January 2025",
    ):
        plan = read(question).plan()
        discovery = plan.steps[0]
        assert isinstance(discovery, ExecuteQueryParams)
        SatQueryIntent.model_validate(discovery.intent.model_dump())
        # Never change detection or object identification - both unimplemented.
        assert discovery.intent.task == "visualize"


# =========================================================================== #
# B. Execution through the existing services
# =========================================================================== #


def make_scene(scene_id: str, when: str, collection: str = "sentinel-2-l2a") -> Scene:
    return Scene(
        id=scene_id,
        datetime=when,
        bbox=BBOX,
        geometry=None,
        cloud_cover=3.0,
        collection=collection,
        platform=None,
        processing_level="L2A",
        thumbnail_url=None,
        assets=[],
    )


def execution_for(request: QueryExecutionRequest, *, selected: bool = True) -> QueryExecutionResult:
    """A result shaped by the request it answers - one scene per window."""

    intent = request.intent
    windows = []
    ranges = (
        [("baseline", intent.time_windows.baseline), ("target", intent.time_windows.target)]
        if isinstance(intent.time_windows, TemporalComparison)
        else [("single", intent.time_windows[0])]
    )
    for modality in intent.modalities:
        for label, time_range in ranges:
            collection = "sentinel-1-rtc" if modality == S1 else "sentinel-2-l2a"
            scene = make_scene(
                f"{'S1A' if modality == S1 else 'S2B'}_{label}_{time_range.start_date:%Y%m%d}",
                f"{time_range.start_date.isoformat()}T05:15:13Z",
                collection,
            )
            windows.append(
                ExecutedWindow(
                    modality=modality,
                    label=label,
                    time_range=time_range,
                    scene_count=1 if selected else 0,
                    scenes=[scene] if selected else [],
                    selected_scene_id=scene.id if selected else None,
                )
            )
    return QueryExecutionResult(
        plan=ResolvedQueryPlan(intent=intent, bbox=BBOX),
        executed_modalities=list(intent.modalities),
        skipped_modalities=[],
        windows=windows,
        catalog="https://earth-search.aws.element84.com/v1",
    )


class RecordingQueryExecution:
    def __init__(self, *, selected: bool = True, error: Exception | None = None) -> None:
        self.calls: list[QueryExecutionRequest] = []
        self._selected = selected
        self._error = error

    async def execute(self, request: QueryExecutionRequest, **_: Any) -> QueryExecutionResult:
        self.calls.append(request)
        if self._error is not None:
            raise self._error
        return execution_for(request, selected=self._selected)


class EngineAnalysis:
    """Returns the values an engine would, keyed by what was requested."""

    NDVI_MEAN = -0.0613608
    NDWI_MEAN = 0.146391
    VV_MEAN = -5.44372

    def __init__(self) -> None:
        self.calls: list[AnalysisRequest] = []

    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        self.calls.append(request)
        measurements: list[Measurement] = []
        # Like the real service: an index needs an optical window that selected
        # a scene, and none is produced without one.
        optical = any(
            w.modality == S2 and w.selected_scene_id for w in request.execution.windows
        )
        for index in request.indices if optical else []:
            mean = {"ndvi": self.NDVI_MEAN, "ndwi": self.NDWI_MEAN, "ndbi": 0.0118356}[index]
            measurements += [
                Measurement(name=f"{index}_valid_pixel_count", value=33524.0, unit="pixels"),
                Measurement(name=f"{index}_mean", value=mean, unit="index"),
            ]
        sar = None
        sar_window = next(
            (w for w in request.execution.windows
             if w.modality == S1 and w.selected_scene_id),
            None,
        )
        if request.include_sar_backscatter and sar_window is not None:
            sar = SarBackscatterResult(
                scene_id=sar_window.selected_scene_id or "",
                window_label=sar_window.label,
                acquired_at=None,
                measurements=[
                    Measurement(name="vv_mean_db", value=self.VV_MEAN, unit="dB"),
                    Measurement(name="vh_mean_db", value=-17.8496, unit="dB"),
                    Measurement(name="vv_minus_vh_mean_db", value=12.4059, unit="dB"),
                ],
            )
        return AnalysisResult(
            status="ok",
            task="visualize",
            answer="Retrieved.",
            windows_considered=[],
            measurements=measurements,
            sar_backscatter=sar,
        )


def standard_service(
    query: Any | None = None, analysis: Any | None = None
) -> tuple[AgentService, Any, Any]:
    query = query or RecordingQueryExecution()
    analysis = analysis or EngineAnalysis()
    service = AgentService(
        planner=StandardPlanner(),
        executor=AgentExecutor(query_execution_service=query, analysis_service=analysis),
        synthesizer=StandardReport(),
    )
    return service, query, analysis


def answer(question: str, **kwargs: Any) -> tuple[AgentResult, Any, Any]:
    service, query, analysis = standard_service(**kwargs)
    result = asyncio.run(service.answer(AgentQuestionRequest(question=question)))
    return result, query, analysis


def test_the_intent_reaches_the_existing_query_execution_contract() -> None:
    _, query, _ = answer("Show vegetation around Chennai in January 2025")

    [request] = query.calls
    assert isinstance(request, QueryExecutionRequest)
    assert request.intent.location_query == "Chennai"
    assert request.intent.time_windows == [window("2025-01-01", "2025-01-31")]
    assert request.intent.modalities == [S2]


def test_the_requested_index_reaches_the_analysis_service() -> None:
    _, _, analysis = answer("Show vegetation around Chennai in January 2025")

    [request] = analysis.calls
    assert request.indices == ["ndvi"]
    assert not request.include_sar_backscatter
    assert not request.include_temporal_ndwi


def test_sar_reaches_the_backscatter_engine_on_a_sentinel_1_window() -> None:
    result, query, analysis = answer("Analyze SAR backscatter around Chennai in January 2025")

    assert query.calls[0].intent.modalities == [S1]
    [request] = analysis.calls
    assert request.include_sar_backscatter
    assert request.indices == []
    assert result.status == "ok"


def test_a_temporal_question_reaches_the_comparison_engine() -> None:
    _, query, analysis = answer(
        "Compare water at Marina Beach, Chennai between January and March 2025"
    )

    assert query.calls[0].intent.temporal_mode == "compare"
    [request] = analysis.calls
    assert request.include_temporal_ndwi
    assert request.indices == []


class RecordingGeocodingQueryService:
    """The geospatial entry point QueryExecutionService grounds through."""

    def __init__(self, error: Exception | None = None) -> None:
        self.places: list[str] = []
        self._error = error

    async def build_plan(self, intent: SatQueryIntent) -> ResolvedQueryPlan:
        self.places.append(intent.location_query)
        if self._error is not None:
            raise self._error
        return ResolvedQueryPlan(intent=intent, bbox=BBOX)


class EmptyCatalog:
    async def search(self, request: Any) -> SceneSearchResponse:
        return SceneSearchResponse(
            query=QueryEcho(
                collections=[request.collection or "sentinel-2-l2a"],
                bbox=[BBOX.west, BBOX.south, BBOX.east, BBOX.north],
                datetime="x",
                max_cloud_cover=None,
                limit=request.limit,
                filter=None,
            ),
            scene_count=0,
            scenes=[],
            catalog="https://earth-search.aws.element84.com/v1",
        )


def real_execution(geocoder: RecordingGeocodingQueryService) -> QueryExecutionService:
    return QueryExecutionService(
        query_service=geocoder,  # type: ignore[arg-type]
        satellite_service=EmptyCatalog(),  # type: ignore[arg-type]
    )


def test_the_place_is_resolved_by_the_existing_geospatial_service() -> None:
    """Mutation target 4 (part 2): the place text reaches the geocoder intact,
    through the REAL QueryExecutionService - the interpreter resolved nothing."""

    geocoder = RecordingGeocodingQueryService()
    result, _, _ = answer(
        "What is the NDWI of Marina Beach, Chennai in January 2025?",
        query=real_execution(geocoder),
    )

    assert geocoder.places == ["Marina Beach, Chennai"]
    # Nothing matched in the (empty) catalog, so nothing is claimed.
    assert result.status == "ok"
    assert result.answer == ABSTENTION


def test_a_place_the_geocoder_cannot_find_is_a_clarification() -> None:
    geocoder = RecordingGeocodingQueryService(
        error=NotFoundError("No matching location was found.")
    )
    result, _, analysis = answer(
        "Show vegetation around Qwxzt in January 2025", query=real_execution(geocoder)
    )

    assert result.status == "needs_clarification"
    assert result.clarification is not None
    assert result.clarification.reason == "location_not_found"
    assert result.clarification.understood_location == "Qwxzt"
    assert "<" not in result.clarification.message
    assert result.answer is None
    assert analysis.calls == []


def test_a_place_too_large_to_measure_is_a_clarification() -> None:
    """The analysis gate refuses the area after geocoding and before any catalog
    search; the user is asked for a smaller place in the gate's own words."""

    from app.services.analysis.validation import AnalysisRequestRejectedError

    refusal = AnalysisRequestRejectedError(
        "aoi_too_large",
        "plan.bbox",
        "this area is about 20.9 x 42.4 km. Measurements are read on the "
        "sensor's native 10 m grid and are never downsampled, which limits an "
        "analysis area to roughly 20 km across. Choose a smaller area.",
    )
    result, _, analysis = answer(
        "Show vegetation around Chennai in January 2025",
        query=RecordingQueryExecution(error=refusal),
    )

    assert result.status == "needs_clarification"
    assert result.clarification is not None
    assert result.clarification.reason == "area_too_large"
    assert result.clarification.understood_location == "Chennai"
    assert "about 20.9 x 42.4 km" in result.clarification.message
    assert "plan.bbox" not in result.clarification.message
    # A real example of the form, never a placeholder the user must decode.
    assert "<" not in result.clarification.message
    assert "park or landmark in Chennai together with the city name" in (
        result.clarification.message
    )
    assert analysis.calls == []
    # The refused step stays in the trace: nothing is hidden.
    assert result.trace.steps[0].status == "failed"


def test_an_unsupported_question_executes_nothing() -> None:
    """Mutation target 6 (part 2): refused before any service is called."""

    result, query, analysis = answer("Detect ships near Chennai port in January 2025")

    assert result.status == "needs_clarification"
    assert result.clarification is not None
    assert result.clarification.reason == "analysis_unsupported"
    assert query.calls == []
    assert analysis.calls == []
    assert result.trace.steps == []
    assert result.evidence.items == []


def test_a_clarification_never_carries_an_answer_or_a_failure() -> None:
    result, _, _ = answer("Show vegetation around Chennai")

    assert result.status == "needs_clarification"
    assert result.answer is None
    assert result.failure is None
    with pytest.raises(ValueError, match="clarification"):
        AgentResult.model_validate(
            {**result.model_dump(), "answer": "The mean NDVI was 0.5 index."}
        )
    with pytest.raises(ValueError, match="clarification"):
        AgentResult.model_validate({**result.model_dump(), "status": "ok",
                                    "answer": "x"})


# =========================================================================== #
# C. The report states engine values and passes the unchanged grounding
# =========================================================================== #


def test_the_answer_is_the_measured_value_not_a_substitute() -> None:
    """Mutation target 8: a fake or mock result substituted for real execution."""

    result, _, _ = answer("Show vegetation around Chennai in January 2025")

    assert result.status == "ok"
    assert result.answer == (
        "The mean NDVI was -0.06136 index. "
        "Scene S2B_single_20250101 was selected. "
        "The scene was acquired on 2025-01-01."
    )
    assert "ndvi.ndvi_mean" in result.trace.evidence_refs
    assert result.evidence.items  # the evidence the sentence came from


def test_every_report_passes_every_grounding_check() -> None:
    for question in (
        "Show vegetation around Chennai in January 2025",
        "What is the NDWI of Marina Beach, Chennai in January 2025?",
        "Show built-up area around Chennai in January 2025",
        "Analyze SAR backscatter around Chennai in January 2025",
        "Show NDVI and SAR backscatter around Chennai in January 2025",
        "Show satellite imagery of Chennai in January 2025",
    ):
        result, _, _ = answer(question)
        validation = result.trace.answer_validation
        assert result.status == "ok", question
        assert validation is not None
        assert (validation.numeric_grounding, validation.evidence_refs,
                validation.forbidden_terms) == ("pass", "pass", "pass"), question


def test_sar_is_reported_in_decibels_per_polarization() -> None:
    result, _, _ = answer("Analyze SAR backscatter around Chennai in January 2025")

    assert result.answer is not None
    assert "The mean VV was -5.444 dB." in result.answer
    assert "The mean VH was -17.85 dB." in result.answer
    assert "The VV minus VH difference was 12.41 dB." in result.answer


def test_nothing_measured_is_an_abstention_never_a_number() -> None:
    result, _, _ = answer(
        "Show vegetation around Chennai in January 2025",
        query=RecordingQueryExecution(selected=False),
    )

    assert result.status == "ok"
    assert result.answer == ABSTENTION


def test_the_temporal_report_uses_the_comparison_sentences() -> None:
    from app.services.agent.schemas import AgentEvidence, EvidenceItem

    evidence = AgentEvidence(items=[
        EvidenceItem(id="temporal_ndwi.first.ndwi_mean", source="temporal_ndwi",
                     measurement=Measurement(name="ndwi_mean", value=0.02665, unit="index")),
        EvidenceItem(id="temporal_ndwi.second.ndwi_mean", source="temporal_ndwi",
                     measurement=Measurement(name="ndwi_mean", value=0.1464, unit="index")),
        EvidenceItem(id="temporal_ndwi.difference.mean_ndwi_difference",
                     source="temporal_ndwi",
                     measurement=Measurement(name="mean_ndwi_difference", value=0.119742,
                                             unit="index")),
    ])
    draft = asyncio.run(StandardReport().synthesize("q", evidence))

    assert draft.summary == (
        "The earlier mean NDWI was 0.02665 index. The later mean NDWI was 0.1464 "
        "index. The mean NDWI difference was 0.1197 index."
    )


# =========================================================================== #
# D. Provider independence, through the HTTP route
# =========================================================================== #


@pytest.fixture
def no_providers(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every way a provider could be built, armed to fail and recorded.

    conftest already clears every provider credential. The fakes below replace
    the network-facing services, so a supported question is proven to run with
    no key AND with no provider constructed at all.
    """

    touched: list[str] = []

    def forbid(name: str) -> Any:
        def boom(*args: Any, **kwargs: Any) -> Any:
            touched.append(name)
            raise AssertionError(f"a provider was built via {name}")

        return boom

    monkeypatch.setattr(query_routes, "get_agent_providers", forbid("get_agent_providers"))
    monkeypatch.setattr(query_routes, "get_intent_parser", forbid("get_intent_parser"))
    monkeypatch.setattr(query_routes, "QueryExecutionService", RecordingQueryExecution)
    monkeypatch.setattr(query_routes, "AnalysisService", EngineAnalysis)
    return touched


def post(question: str, **extra: Any) -> Any:
    client = TestClient(create_app(), raise_server_exceptions=False)
    return client.post("/api/v1/query/agent", json={"question": question, **extra})


@pytest.mark.parametrize(
    "question",
    [
        "Show vegetation around Chennai in January 2025",
        "Find water around Puzhal Lake in January 2025",
        "Show built-up areas around Chennai in December 2024",
        "Analyze SAR backscatter around Chennai in January 2025",
    ],
)
def test_supported_questions_run_with_no_ai_provider(
    no_providers: list[str], question: str
) -> None:
    """Mutation target 7: an external AI provider made mandatory."""

    response = post(question)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["answer"]
    assert body["trace"]["answer_validation"]["numeric_grounding"] == "pass"
    assert no_providers == []


def test_a_clarification_is_a_200_with_no_provider(no_providers: list[str]) -> None:
    response = post("I want to know something about Chennai")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "needs_clarification"
    assert body["clarification"]["reason"] == "analysis_missing"
    assert no_providers == []


def test_ai_interpretation_is_used_only_when_named(no_providers: list[str]) -> None:
    """Non-vacuity: the tripwire does fire when a provider IS asked for."""

    response = post("Show vegetation around Chennai in January 2025", provider="gemini")

    assert no_providers == ["get_agent_providers"]
    assert response.status_code == 500


def test_parse_without_a_provider_needs_no_key() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/api/v1/query/parse",
        json={"prompt": "Compare water at Marina Beach, Chennai between January and March 2025"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["temporal_mode"] == "compare"
    assert body["location_query"] == "Marina Beach, Chennai"


def test_parse_reports_a_clarification_as_a_422() -> None:
    client = TestClient(create_app())

    response = client.post("/api/v1/query/parse", json={"prompt": "Show change"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "clarification_required"


def test_the_standard_modules_import_no_provider() -> None:
    """The workflow cannot reach a model: no provider module is importable
    from it, so none can be called."""

    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "app" / "services" / "agent"
    for name in ("interpretation.py", "standard.py"):
        tree = ast.parse((root / name).read_text())
        modules = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        } | {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        assert not any("providers" in m or "google" in m or "anthropic" in m
                       or "httpx" in m for m in modules), (name, modules)
