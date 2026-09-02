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


# =========================================================================== #
# Post-Phase-15 hardening
#
# MEDIUM-3: only the REQUESTED windows were allowlisted, so an answer citing
# the date a scene was actually acquired - a fact the evidence carries - was
# withheld as ungrounded.
#
# MEDIUM-4: the number pattern could not match scientific notation, so such a
# value was skipped entirely rather than checked. A skipped number is an
# unchecked claim, which is the one thing this validator exists to prevent.
# =========================================================================== #


def test_an_actual_acquisition_date_is_accepted() -> None:
    """The scene was acquired 2024-01-15; the request asked for 01-01..01-31."""

    result = check("The scene was acquired on 2024-01-15.")
    assert result.numeric_grounding == "pass"


def test_an_acquisition_date_outside_the_evidence_is_still_rejected() -> None:
    """Widened for real acquisitions only - not for arbitrary dates."""

    assert check("The scene was acquired on 2024-07-04.").numeric_grounding == "fail"


def test_the_acquisition_month_and_year_are_accepted() -> None:
    assert check("Acquired in January 2024.").numeric_grounding == "pass"


def test_acquisition_dates_come_from_the_evidence_not_from_the_answer() -> None:
    """With no execution there is no acquisition to allowlist."""

    evidence = AgentEvidence.model_validate(
        {"items": [measurement_item("ndwi.ndwi_mean", "ndwi_mean", 0.2777, "index")]}
    )
    result = validate_answer(
        draft("The scene was acquired on 2024-01-15.", ["ndwi.ndwi_mean"]), evidence
    )
    assert result.numeric_grounding == "fail"


@pytest.mark.parametrize("literal", ["1.5e10", "2E5", "-3.2e-4", "1e3"])
def test_scientific_notation_is_extracted_rather_than_skipped(literal: str) -> None:
    """Whatever the verdict, the number must be CHECKED, not ignored."""

    from app.services.agent.grounding import _NUMBER

    assert [m.group(0) for m in _NUMBER.finditer(f"value {literal} here")] == [
        literal
    ]


@pytest.mark.parametrize("literal", ["1.5e10", "2E5", "-3.2e-4"])
def test_an_ungrounded_scientific_number_is_rejected(literal: str) -> None:
    assert check(f"The value was {literal}.").numeric_grounding == "fail"


def test_a_grounded_scientific_number_is_accepted() -> None:
    """1234567 written as 1.234567e6 is the same measurement."""

    result = check(
        "1.234567e6 valid pixels were analysed.",
        refs=["ndwi.ndwi_valid_pixel_count"],
    )
    assert result.numeric_grounding == "pass"


def test_scientific_notation_matches_at_its_own_precision() -> None:
    evidence = make_evidence(
        items=[measurement_item("ndwi.count", "ndwi_valid_pixel_count", 1234567.0, "pixels")]
    )
    # 1.2e6 -> 1200000 at that precision; 1234567 rounds to 1200000. Accepted.
    assert (
        validate_answer(draft("1.2e6 pixels.", ["ndwi.count"]), evidence)
        .numeric_grounding
        == "pass"
    )
    # 9.9e6 does not.
    assert (
        validate_answer(draft("9.9e6 pixels.", ["ndwi.count"]), evidence)
        .numeric_grounding
        == "fail"
    )


def test_ordinary_prose_containing_e_is_unaffected() -> None:
    """The widened pattern must not start matching words."""

    assert check("5 elephants were not counted.").numeric_grounding == "fail"
    assert check("No numeric claim here at all.").numeric_grounding == "pass"


# =========================================================================== #
# A-1: a number glued to a unit must never be silently skipped.
#
# The trailing ``(?![A-Za-z_])`` guard meant "12km2", "500m" and "0.99x"
# produced NO regex match at all, so the number was never checked and an
# unsupported quantitative claim passed as grounded. A skipped number is an
# unchecked claim - the single failure mode this validator exists to prevent.
#
# The invariant: every number in generated prose is either checked against the
# evidence or fails closed. It is never ignored because a letter follows it.
#
# Note what this does NOT do: the unit itself is never interpreted. "500m" is
# grounded iff 500 is in the evidence; grounding has no idea what a metre is,
# and deliberately no measurement ontology is introduced.
# =========================================================================== #


@pytest.mark.parametrize(
    "summary",
    [
        "The lake covers 12km2.",
        "The lake covers 12km2 of the scene.",
        "The shoreline changed by 500m.",
        "Mean NDWI was 0.99x the baseline.",
        "The value increased by 3.5x.",
        "The area is 12km².",
    ],
)
def test_an_unsupported_number_with_a_unit_fails_closed(summary: str) -> None:
    """Evidence holds only 0.2777 and 1234567.0 - none of these is supported."""

    assert check(summary).numeric_grounding == "fail"


@pytest.mark.parametrize(
    "summary,literal",
    [
        ("The lake covers 12km2.", "12"),
        ("The shoreline changed by 500m.", "500"),
        ("Mean NDWI was 0.99x the baseline.", "0.99"),
        ("The area is 12km².", "12"),
    ],
)
def test_the_numeric_component_is_actually_extracted(
    summary: str, literal: str
) -> None:
    """Not merely failing - failing because the number was SEEN and checked."""

    from app.services.agent.grounding import _ungrounded_claims

    assert literal in _ungrounded_claims(summary, make_evidence())


@pytest.mark.parametrize(
    "summary,value",
    [
        ("The shoreline changed by 500m.", 500.0),
        ("The lake covers 12km2.", 12.0),
        ("Mean NDWI was 0.99x the baseline.", 0.99),
    ],
)
def test_a_supported_number_with_a_unit_is_accepted(
    summary: str, value: float
) -> None:
    """The unit is irrelevant; only the number must be traceable."""

    evidence = make_evidence(
        items=[measurement_item("ndwi.v", "measured_value", value, "index")]
    )
    result = validate_answer(draft(summary, ["ndwi.v"]), evidence)
    assert result.numeric_grounding == "pass"


def test_a_number_glued_to_a_word_is_still_checked() -> None:
    """The A-2 skip class disappears with the same fix."""

    from app.services.agent.grounding import _ungrounded_claims

    assert "2" in _ungrounded_claims("There were 2eggs.", make_evidence())


# --- A-1 must not regress the protections it sits behind ------------------- #


def test_a_real_scene_id_still_yields_no_numeric_claims() -> None:
    """Scene ids are masked BEFORE numbers are read - not by the letter guard."""

    from app.services.agent.grounding import _ungrounded_claims

    assert _ungrounded_claims(f"Scene {SCENE_ID} was selected.", make_evidence()) == []


def test_a_scene_id_is_safe_even_without_masking() -> None:
    """Defence in depth: the leading boundary alone blocks its digits.

    Every digit run in ``S2B_44PLV_20241026_0_L2A`` is preceded by a letter or
    an underscore, so the leading guard rejects it regardless of masking. This
    is why removing the TRAILING guard cannot weaken scene-id handling.
    """

    from app.services.agent.grounding import _NUMBER

    assert [m.group(0) for m in _NUMBER.finditer(SCENE_ID)] == []


@pytest.mark.parametrize("token", ["L2A", "S2B", "B04", "EPSG"])
def test_identifier_fragments_are_not_read_as_numbers(token: str) -> None:
    from app.services.agent.grounding import _NUMBER

    assert [m.group(0) for m in _NUMBER.finditer(f"the {token} value")] == []


def test_an_invented_scene_id_is_still_reported() -> None:
    assert (
        check("Scene S2A_99XYZ_20190517_9_L2A was selected.").numeric_grounding
        == "fail"
    )


@pytest.mark.parametrize(
    "literal", ["1.5e10", "2E5", "-3.2e-4", "1.25E+06", "1e3"]
)
def test_scientific_notation_still_extracts_whole(literal: str) -> None:
    """The exponent must not be split off now that letters may follow."""

    from app.services.agent.grounding import _NUMBER

    assert [m.group(0) for m in _NUMBER.finditer(f"value {literal} here")] == [
        literal
    ]


def test_scientific_notation_grounding_is_unchanged() -> None:
    assert check("1.234567e6 valid pixels.").numeric_grounding == "pass"
    assert check("The value was 9.9e6.").numeric_grounding == "fail"


def test_the_acquisition_date_fix_still_holds() -> None:
    """dcac153's MEDIUM-3 fix must survive the A-1 change."""

    assert check("The scene was acquired on 2024-01-15.").numeric_grounding == "pass"
    assert check("The scene was acquired on 2024-07-04.").numeric_grounding == "fail"


def test_ordinary_prose_is_unchanged() -> None:
    assert check("5 elephants were not counted.").numeric_grounding == "fail"
    assert check("3 exabytes of nothing.").numeric_grounding == "fail"
    assert check("the e value is irrelevant.").numeric_grounding == "pass"


# =========================================================================== #
# G. Platform identifiers are not quantitative claims
# =========================================================================== #
#
# Found by live Gemini validation, not by a unit test. A factually correct
# answer was withheld because "Sentinel-2" contributed the literal "2", which
# no measurement could account for. The digit belongs to a platform name; it
# claims nothing.
#
# The trap was self-inflicted: ``AnalysisService`` mandates an NDWI disclaimer
# naming Sentinel-2, so any answer faithfully repeating the system's own
# evidence text failed grounding. Every correct answer was rejected.
#
# The fix masks platform identifiers before numbers are read - the same
# treatment scene ids and allowed dates already receive - and leaves ``_NUMBER``
# untouched, so A-1 detection is unaffected.


@pytest.mark.parametrize(
    "summary",
    [
        "The imagery is from Sentinel-2.",
        "The imagery is from Sentinel-1.",
        "The imagery is from Sentinel-2A.",
        "The modality was sentinel-2-optical.",
        "The imagery is from Sentinel-1 SAR.",
        "Sentinel-2 and Sentinel-1 were both considered.",
        "sentinel-1-sar and sentinel-2-optical are distinct modalities.",
    ],
)
def test_platform_identifiers_are_not_numeric_claims(summary: str) -> None:
    assert check(summary).numeric_grounding == "pass"


def test_the_exact_live_failure_is_grounded() -> None:
    """The real rejected answer from live Gemini validation.

    Reproduces the mechanism verbatim: the measurements are the ones the engine
    actually produced, and the closing sentence is the disclaimer
    ``AnalysisService`` itself appends. Before the fix this returned "fail" on
    the single literal "2".
    """

    evidence = make_evidence(
        items=[
            measurement_item(
                "ndwi.ndwi_mean", "ndwi_mean", 0.1463908465206975, "index"
            ),
            measurement_item(
                "ndwi.ndwi_min", "ndwi_min", -0.7808971620384498, "index"
            ),
            measurement_item(
                "ndwi.ndwi_max", "ndwi_max", 0.9729119638826185, "index"
            ),
        ]
    )
    summary = (
        "In January 2024, the mean NDWI of Marina Beach, Chennai is "
        "0.1463908465206975, with a minimum of -0.7808971620384498 and a "
        "maximum of 0.9729119638826185. Note that NDWI values are a spectral "
        "index computed from raw Sentinel-2 digital numbers and are not a "
        "validated water or flood classification."
    )
    assert validate_answer(draft(summary), evidence).numeric_grounding == "pass"


def test_the_production_ndwi_disclaimer_grounds_clean() -> None:
    """The exact warning string ``AnalysisService`` emits must not self-reject."""

    disclaimer = (
        "NDWI values are a spectral index computed from raw Sentinel-2 "
        "digital numbers; they are not a validated water or flood "
        "classification."
    )
    assert check(disclaimer).numeric_grounding == "pass"


def test_a_realistic_mixed_answer_passes() -> None:
    assert (
        check(
            "In January 2024, the mean NDWI is 0.2777. "
            "The imagery is from Sentinel-2."
        ).numeric_grounding
        == "pass"
    )


# --- the mask must not become a smuggling route ---------------------------- #


def test_masking_a_platform_name_does_not_hide_other_claims() -> None:
    """Only the identifier is blanked; the rest of the sentence is still read."""

    assert (
        check(
            "The imagery is from Sentinel-2 and the area increased by 37%."
        ).numeric_grounding
        == "fail"
    )


def test_a_standalone_two_is_still_checked() -> None:
    """Masking is span-local: an unrelated bare 2 is not excused by it."""

    assert (
        check("Sentinel-2 imagery showed 2 lakes.").numeric_grounding == "fail"
    )


@pytest.mark.parametrize(
    "summary",
    [
        "Sentinel-12345 covered the area.",
        "Sentinel-3 was used.",
        "Sentinel-25 was used.",
    ],
)
def test_only_real_sentinel_platform_numbers_are_excused(summary: str) -> None:
    """Fail closed: anything outside the known platform shape stays a claim."""

    assert check(summary).numeric_grounding == "fail"


# --- A-1 must survive the platform mask ------------------------------------ #


@pytest.mark.parametrize(
    "summary",
    [
        "Sentinel-2 imagery shows the lake covers 12km2.",
        "Sentinel-2 imagery shows the shoreline changed by 500m.",
        "Sentinel-1 data gives 0.99x the baseline.",
        "Sentinel-2 shows a 3.5x increase.",
        "Sentinel-2 imagery shows the area is 12km².",
        "Sentinel-2 imagery shows a 37% increase.",
    ],
)
def test_unit_attached_claims_are_still_caught_beside_a_platform_name(
    summary: str,
) -> None:
    """The A-1 fix from 5d15b56 is untouched by the platform mask."""

    assert check(summary).numeric_grounding == "fail"


def test_scientific_notation_still_checked_beside_a_platform_name() -> None:
    assert (
        check("Sentinel-2 imagery shows 1.2e-3 change.").numeric_grounding
        == "fail"
    )


def test_grounded_values_still_pass_beside_a_platform_name() -> None:
    assert (
        check(
            "Sentinel-2 imagery gives a mean NDWI of 0.2777 index."
        ).numeric_grounding
        == "pass"
    )
