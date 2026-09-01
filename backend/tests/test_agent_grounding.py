"""Phase 15 Commit 3 - grounding validator tests.

Grounding is **containment, not proof**. These tests pin exactly what the
validator can mechanically establish - that every number a generated answer
states is traceable to evidence, that its citations resolve, and that it uses
no forbidden vocabulary - and they also pin, explicitly, what it cannot: the
semantic truth of prose.

The validator is pure. No provider, no network, no filesystem, no service, no
mutation of its inputs.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Any

import pytest
from app.services.agent import grounding as grounding_mod
from app.services.agent.grounding import (
    FORBIDDEN_PHRASES,
    DraftAnswer,
    validate_answer,
)
from app.services.agent.schemas import AgentEvidence, AnswerValidation
from app.services.analysis.schemas import AnalysisResult
from app.services.geospatial.schemas import BoundingBox
from app.services.query.schemas import (
    ExecutedWindow,
    QueryExecutionResult,
    ResolvedQueryPlan,
    SatQueryIntent,
    TimeRange,
)
from app.services.satellite.schemas import Scene

DEFAULT_BBOX = BoundingBox(west=80.10, south=12.90, east=80.30, north=13.20)
CATALOG = "https://earth-search.aws.element84.com/v1"
S2 = "sentinel-2-optical"
SCENE_ID = "S2B_44PLV_20241026_0_L2A"


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def intent_dict(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "location_query": "Chennai",
        "temporal_mode": "single",
        "time_windows": [{"start_date": "2024-01-01", "end_date": "2024-01-31"}],
        "modalities": [S2],
        "task": "visualize",
    }
    body.update(overrides)
    return body


def make_execution(**overrides: Any) -> QueryExecutionResult:
    intent = SatQueryIntent.model_validate(intent_dict(**overrides))
    scene = Scene(
        id=SCENE_ID,
        datetime="2024-01-15T05:00:00Z",
        bbox=DEFAULT_BBOX,
        geometry=None,
        cloud_cover=1.0,
        collection="sentinel-2-l2a",
        platform=None,
        processing_level="L2A",
        thumbnail_url=None,
        assets=[],
    )
    window = ExecutedWindow(
        modality=S2,
        label="single",
        time_range=TimeRange.model_validate(
            {"start_date": "2024-01-01", "end_date": "2024-01-31"}
        ),
        scene_count=3,
        scenes=[scene],
        selected_scene_id=scene.id,
        imagery=None,
        imagery_error=None,
    )
    return QueryExecutionResult(
        plan=ResolvedQueryPlan(intent=intent, bbox=DEFAULT_BBOX),
        executed_modalities=[S2],
        skipped_modalities=[],
        windows=[window],
        catalog=CATALOG,
    )


def measurement_item(
    item_id: str, name: str, value: float, unit: str
) -> dict[str, Any]:
    return {
        "id": item_id,
        "source": "ndwi",
        "measurement": {"name": name, "value": value, "unit": unit},
        "produced_by": "analysis.engines.compute_ndwi_measurements",
    }


def make_evidence(
    *,
    items: list[dict[str, Any]] | None = None,
    execution: QueryExecutionResult | None = None,
    analysis: AnalysisResult | None = None,
) -> AgentEvidence:
    payload: dict[str, Any] = {
        "items": (
            items
            if items is not None
            else [
                measurement_item("ndwi.ndwi_mean", "ndwi_mean", 0.2777, "index"),
                measurement_item(
                    "ndwi.ndwi_valid_pixel_count",
                    "ndwi_valid_pixel_count",
                    1234567.0,
                    "pixels",
                ),
            ]
        )
    }
    payload["execution"] = (
        execution if execution is not None else make_execution()
    ).model_dump(mode="json")
    if analysis is not None:
        payload["analysis"] = analysis.model_dump(mode="json")
    return AgentEvidence.model_validate(payload)


def draft(summary: str, refs: list[str] | None = None) -> DraftAnswer:
    return DraftAnswer(
        summary=summary,
        evidence_refs=refs if refs is not None else ["ndwi.ndwi_mean"],
    )


def check(summary: str, **kwargs: Any) -> AnswerValidation:
    refs = kwargs.pop("refs", None)
    evidence = kwargs.pop("evidence", None) or make_evidence(**kwargs)
    return validate_answer(draft(summary, refs), evidence)


# =========================================================================== #
# A. Numeric grounding
# =========================================================================== #


def test_exact_evidence_value_is_accepted() -> None:
    assert check("The mean NDWI was 0.2777 index.").numeric_grounding == "pass"


def test_rounded_representation_of_an_evidence_value_is_accepted() -> None:
    """0.2777 presented as 0.28 is the same measurement, shown to 2 dp."""

    assert check("The mean NDWI was about 0.28.").numeric_grounding == "pass"


def test_rounding_is_matched_at_the_precision_the_answer_used() -> None:
    for text in ("0.3", "0.28", "0.278", "0.2777"):
        assert check(f"Mean NDWI {text}.").numeric_grounding == "pass", text


def test_a_materially_different_number_is_rejected() -> None:
    assert check("The mean NDWI was 0.85.").numeric_grounding == "fail"


def test_a_plausible_but_wrong_rounding_is_rejected() -> None:
    """0.27 is not what 0.2777 rounds to at 2 dp."""

    assert check("The mean NDWI was 0.27.").numeric_grounding == "fail"


def test_multiple_grounded_measurements_are_accepted() -> None:
    result = check(
        "Mean NDWI was 0.2777 index over 1234567 valid pixels.",
        refs=["ndwi.ndwi_mean", "ndwi.ndwi_valid_pixel_count"],
    )
    assert result.numeric_grounding == "pass"


def test_thousands_separators_are_understood() -> None:
    assert check("1,234,567 valid pixels were analysed.").numeric_grounding == "pass"


def test_a_fabricated_number_among_grounded_ones_is_rejected() -> None:
    result = check("Mean NDWI 0.2777 across 42 lakes.")
    assert result.numeric_grounding == "fail"


def test_percentages_match_at_their_stated_precision() -> None:
    evidence = make_evidence(
        items=[
            measurement_item(
                "ndwi.pct", "ndwi_percent_above_index_threshold_0.3", 44.03472, "%"
            )
        ]
    )
    assert (
        validate_answer(draft("44.0% of valid pixels.", ["ndwi.pct"]), evidence)
        .numeric_grounding
        == "pass"
    )


def test_negative_values_are_grounded_like_any_other() -> None:
    evidence = make_evidence(
        items=[measurement_item("ndwi.ndwi_mean", "ndwi_mean", -0.4200, "index")]
    )
    assert (
        validate_answer(draft("Mean NDWI was -0.42.", ["ndwi.ndwi_mean"]), evidence)
        .numeric_grounding
        == "pass"
    )
    assert (
        validate_answer(draft("Mean NDWI was -0.99.", ["ndwi.ndwi_mean"]), evidence)
        .numeric_grounding
        == "fail"
    )


def test_an_answer_with_no_numbers_is_trivially_grounded() -> None:
    assert check("Optical imagery was retrieved for the area.").numeric_grounding == "pass"


def test_grounding_is_not_naive_substring_matching() -> None:
    """0.2 is a substring of "0.2777" but is not what it rounds to at 1 dp."""

    assert check("The mean NDWI was 0.2.").numeric_grounding == "fail"


# =========================================================================== #
# B. Date grounding - narrowly tied to the validated intent
# =========================================================================== #


def test_iso_dates_from_the_intent_window_are_allowed() -> None:
    result = check("Imagery from 2024-01-01 to 2024-01-31 was retrieved.")
    assert result.numeric_grounding == "pass"


def test_the_intent_year_is_allowed() -> None:
    assert check("The 2024 acquisition was used.").numeric_grounding == "pass"


def test_a_month_and_year_from_the_intent_window_is_allowed() -> None:
    assert check("Imagery from January 2024 was used.").numeric_grounding == "pass"


def test_an_unrelated_fabricated_iso_date_is_rejected() -> None:
    assert check("Imagery from 2019-05-17 was used.").numeric_grounding == "fail"


def test_an_unrelated_year_is_rejected() -> None:
    assert check("The 1998 acquisition was used.").numeric_grounding == "fail"


def test_scene_identifiers_do_not_become_numeric_claims() -> None:
    """A scene id is full of digits; none of them is a measurement."""

    result = check(f"Scene {SCENE_ID} was selected.")
    assert result.numeric_grounding == "pass"


def test_a_scene_id_not_present_in_the_evidence_is_not_silently_allowed() -> None:
    result = check("Scene S2A_99XYZ_20190517_9_L2A was selected.")
    assert result.numeric_grounding == "fail"


def test_compare_mode_windows_are_all_allowlisted() -> None:
    execution = make_execution(
        temporal_mode="compare",
        time_windows={
            "baseline": {"start_date": "2024-01-01", "end_date": "2024-01-31"},
            "target": {"start_date": "2024-06-01", "end_date": "2024-06-30"},
        },
    )
    evidence = make_evidence(execution=execution)
    result = validate_answer(
        draft("Compared 2024-01-01 with 2024-06-30.", ["ndwi.ndwi_mean"]), evidence
    )
    assert result.numeric_grounding == "pass"


def test_dates_are_not_allowlisted_when_there_is_no_execution() -> None:
    evidence = AgentEvidence.model_validate(
        {"items": [measurement_item("ndwi.ndwi_mean", "ndwi_mean", 0.2777, "index")]}
    )
    result = validate_answer(draft("Imagery from 2024-01-01.", ["ndwi.ndwi_mean"]), evidence)
    assert result.numeric_grounding == "fail"


# =========================================================================== #
# C. Evidence references
# =========================================================================== #


def test_resolving_references_pass() -> None:
    result = check(
        "Mean NDWI 0.2777.",
        refs=["ndwi.ndwi_mean", "ndwi.ndwi_valid_pixel_count"],
    )
    assert result.evidence_refs == "pass"


def test_a_nonexistent_reference_fails() -> None:
    result = check("Mean NDWI 0.2777.", refs=["ndwi.does_not_exist"])
    assert result.evidence_refs == "fail"


def test_an_empty_reference_list_passes_vacuously() -> None:
    assert check("No numbers here.", refs=[]).evidence_refs == "pass"


def test_reference_checking_uses_the_evidence_id_set() -> None:
    """Uniqueness is the contract's job; grounding only resolves."""

    evidence = make_evidence()
    assert evidence.ids() == {"ndwi.ndwi_mean", "ndwi.ndwi_valid_pixel_count"}
    assert validate_answer(draft("x", ["ndwi.ndwi_mean"]), evidence).evidence_refs == "pass"


# =========================================================================== #
# D. Forbidden phrases - the Phase 14 protection, unweakened
# =========================================================================== #


def test_the_phase_14_phrase_list_is_fully_covered() -> None:
    """The production list must not drift below the Phase 14 test list."""

    from tests.test_temporal_ndwi import FORBIDDEN_PHRASES as PHASE_14

    assert set(PHASE_14) <= set(FORBIDDEN_PHRASES)


@pytest.mark.parametrize("phrase", sorted(FORBIDDEN_PHRASES))
def test_every_forbidden_phrase_fails_validation(phrase: str) -> None:
    result = check(f"This is a {phrase} result.")
    assert result.forbidden_terms == "fail"


def test_forbidden_phrases_are_matched_case_insensitively() -> None:
    assert check("This is CHANGE DETECTION.").forbidden_terms == "fail"


def test_a_normal_grounded_answer_passes_the_phrase_scan() -> None:
    result = check("Mean NDWI was 0.2777 index for the selected scene.")
    assert result.forbidden_terms == "pass"


def test_the_word_change_alone_is_not_forbidden() -> None:
    """Phase 14.1: the word is fine; the mischaracterisation is not."""

    assert check("No change in acquisition mode was needed.").forbidden_terms == "pass"


# =========================================================================== #
# E. Suppression
# =========================================================================== #


def test_a_suppressed_difference_cannot_be_cited() -> None:
    """differences == [] means no difference value exists to ground against."""

    evidence = make_evidence(
        items=[
            measurement_item(
                "temporal_ndwi.first.ndwi_mean", "ndwi_mean", 0.2000, "index"
            ),
            measurement_item(
                "temporal_ndwi.second.ndwi_mean", "ndwi_mean", 0.5000, "index"
            ),
        ]
    )
    result = validate_answer(
        draft(
            "The mean NDWI difference was 0.3000 index.",
            ["temporal_ndwi.first.ndwi_mean"],
        ),
        evidence,
    )
    assert result.numeric_grounding == "fail"


def test_an_emitted_difference_can_be_cited() -> None:
    evidence = make_evidence(
        items=[
            measurement_item(
                "temporal_ndwi.difference.mean_ndwi_difference",
                "mean_ndwi_difference",
                0.3000,
                "index",
            )
        ]
    )
    result = validate_answer(
        draft(
            "The mean NDWI difference was 0.3000 index.",
            ["temporal_ndwi.difference.mean_ndwi_difference"],
        ),
        evidence,
    )
    assert result.numeric_grounding == "pass"


def test_suppression_is_enforced_by_absence_not_by_a_special_rule() -> None:
    """No suppression-specific branch: the value simply is not in evidence."""

    source = pathlib.Path(grounding_mod.__file__).read_text()
    assert "differences" not in source
    assert "temporal_comparison" not in source


# =========================================================================== #
# F. Result contract
# =========================================================================== #


def test_the_result_is_the_existing_answer_validation_contract() -> None:
    result = check("Mean NDWI 0.2777.")
    assert isinstance(result, AnswerValidation)
    assert set(AnswerValidation.model_fields) == {
        "numeric_grounding",
        "forbidden_terms",
        "evidence_refs",
    }


def test_all_three_checks_are_always_reported() -> None:
    result = check("Mean NDWI 0.2777.")
    for outcome in (
        result.numeric_grounding,
        result.forbidden_terms,
        result.evidence_refs,
    ):
        assert outcome in {"pass", "fail"}
        assert outcome != "not_run"


def test_checks_are_independent() -> None:
    """A forbidden phrase must not mask an otherwise-grounded number."""

    result = check("Mean NDWI 0.2777 shows change detection.")
    assert result.numeric_grounding == "pass"
    assert result.forbidden_terms == "fail"


def test_no_reasoning_field_exists_on_the_result() -> None:
    for banned in (
        "reasoning",
        "thoughts",
        "thinking",
        "chain_of_thought",
        "rationale",
        "explanation",
    ):
        assert banned not in AnswerValidation.model_fields
        assert banned not in DraftAnswer.model_fields


def test_draft_answer_is_minimal() -> None:
    assert set(DraftAnswer.model_fields) == {"summary", "evidence_refs"}


def test_draft_answer_refuses_extra_fields() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DraftAnswer.model_validate(
            {"summary": "x", "evidence_refs": [], "reasoning": "because"}
        )


# =========================================================================== #
# G. Determinism and purity
# =========================================================================== #


def test_validation_is_deterministic() -> None:
    evidence = make_evidence()
    answer = draft("Mean NDWI 0.2777 over 1,234,567 pixels.")
    results = [validate_answer(answer, evidence) for _ in range(5)]
    assert all(r == results[0] for r in results)


def test_validation_does_not_mutate_its_inputs() -> None:
    evidence = make_evidence()
    answer = draft("Mean NDWI 0.2777.")
    before = (evidence.model_dump(mode="json"), answer.model_dump())

    validate_answer(answer, evidence)

    assert (evidence.model_dump(mode="json"), answer.model_dump()) == before


# =========================================================================== #
# H. Security and boundaries
# =========================================================================== #


def _tree() -> ast.Module:
    return ast.parse(pathlib.Path(grounding_mod.__file__).read_text())


def test_grounding_uses_no_dynamic_execution_primitive() -> None:
    forbidden = {
        "eval",
        "exec",
        "getattr",
        "setattr",
        "compile",
        "__import__",
        "globals",
        "locals",
        "vars",
        "open",
        "input",
    }
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in forbidden, f"grounding.py calls {node.func.id}()"


def test_grounding_imports_nothing_impure() -> None:
    roots: set[str] = set()
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])

    for forbidden in (
        "google",
        "genai",
        "httpx",
        "rasterio",
        "numpy",
        "fastapi",
        "subprocess",
        "importlib",
        "os",
        "pathlib",
        "random",
        "datetime",
        "time",
    ):
        assert forbidden not in roots, f"{forbidden!r} must not be imported"
    assert roots <= {"__future__", "collections", "re", "typing", "pydantic", "app"}


def test_grounding_depends_on_no_service() -> None:
    source = pathlib.Path(grounding_mod.__file__).read_text()
    for banned in ("QueryExecutionService", "AnalysisService", "ImageryService"):
        assert banned not in source


# =========================================================================== #
# I. Documented limits - what grounding does NOT establish
# =========================================================================== #


def test_qualitative_prose_without_numbers_is_not_challenged() -> None:
    """A stated limit: unquantified claims pass the numeric check.

    "the index rose" contains no number, so numeric grounding has nothing to
    check. This is containment, not proof, and the module says so.
    """

    result = check("The index rose noticeably across the period.")
    assert result.numeric_grounding == "pass"


def test_the_module_documents_containment_rather_than_proof() -> None:
    doc = grounding_mod.__doc__ or ""
    assert "containment" in doc.lower()
    assert "not proof" in doc.lower() or "not a proof" in doc.lower()
