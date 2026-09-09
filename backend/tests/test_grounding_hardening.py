"""Claim/evidence regression tests: a correct claim survives, mutations do not."""

from __future__ import annotations

import pytest
from app.services.agent.grounding import _NUMBER, DraftAnswer, validate_answer
from app.services.agent.schemas import AgentEvidence, EvidenceItem, VisualObservation
from app.services.analysis.schemas import AnalysisResult, Measurement, TemporalIndexComparison

from tests.test_agent_grounding import SCENE_ID, make_execution
from tests.test_temporal_ndwi import compat, index_result


def measured(
    name: str = "ndwi_mean",
    value: float = 0.2777,
    unit: str = "index",
    source: str = "ndwi",
    item_id: str | None = None,
):
    return EvidenceItem(
        id=item_id or f"{source}.{name}",
        source=source,
        measurement=Measurement(name=name, value=value, unit=unit),
    )


def validate(text: str, items: list[EvidenceItem], refs: list[str] | None = None, **context):
    return validate_answer(
        DraftAnswer(
            summary=text, evidence_refs=refs if refs is not None else [i.id for i in items]
        ),
        AgentEvidence(items=items, **context),
    )


def accepted(result) -> bool:
    return result.numeric_grounding == result.evidence_refs == result.forbidden_terms == "pass"


@pytest.mark.parametrize("literal", [".987", "-.987", "+.987", "−.987", ".987e-2"])
def test_leading_dot_decimal_is_never_skipped(literal: str) -> None:
    assert [m.group() for m in _NUMBER.finditer(f"NDWI is {literal}.")] == [literal]
    result = validate(f"NDWI is {literal}.", [])
    assert result.numeric_grounding == "fail"
    assert not accepted(result)


@pytest.mark.parametrize(
    "literal,value",
    [
        (".28", 0.2777),
        ("-.28", -0.2777),
        ("+.28", 0.2777),
        ("−.28", -0.2777),
        (".28e-2", 0.002777),
    ],
)
def test_supported_leading_dot_and_rounding_survive(literal: str, value: float) -> None:
    assert accepted(validate(f"Mean NDWI is {literal} index.", [measured(value=value)]))


@pytest.mark.parametrize("literal", ["1,23", "1,,000", "1e9999", "1e309"])
def test_malformed_or_unbounded_numeric_forms_fail_closed(literal: str) -> None:
    assert not accepted(validate(f"Mean NDWI is {literal}.", [measured(value=1)]))


def test_scene_count_cannot_authorize_ndvi_even_when_cited() -> None:
    count = measured("scene_count", 1, "count", "execution")
    assert accepted(validate("1 scenes were reported.", [count]))
    result = validate("NDVI is 1.", [count])
    assert result.numeric_grounding == "fail"


@pytest.mark.parametrize("mutation", ["metric", "statistic", "unit", "source", "id", "value"])
def test_one_field_evidence_mutations_break_support(mutation: str) -> None:
    text = "Mean NDWI is 0.28 index."
    assert accepted(validate(text, [measured()]))
    options = {
        "metric": dict(name="ndvi_mean", source="ndvi"),
        "statistic": dict(name="ndwi_max"),
        "unit": dict(unit="pixels"),
        "source": dict(source="model"),
        "id": dict(item_id="ndvi.ndvi_mean"),
        "value": dict(value=0.987),
    }
    assert validate(text, [measured(**options[mutation])]).numeric_grounding == "fail"


def test_measurement_unit_must_be_valid_even_when_prose_omits_it() -> None:
    assert accepted(validate("Mean NDWI is .28.", [measured()]))
    assert validate("Mean NDWI is .28.", [measured(unit="pixels")]).numeric_grounding == "fail"


@pytest.mark.parametrize(
    "text",
    [
        "Mean NDVI is 0.28.",
        "Maximum NDWI is 0.28.",
        "NDWI is 0.28%.",
        "Mean NDWI is 0.28x.",
        "Mean NDWI is 0.28m.",
        "The mean NDWI difference is 0.28.",
        "Mean NDWI change is 0.28.",
    ],
)
def test_one_claim_mutation_cannot_reuse_a_mean(text: str) -> None:
    assert validate(text, [measured()]).numeric_grounding == "fail"


def test_correct_multimetric_answer_and_swapped_values() -> None:
    items = [measured(), measured("ndvi_mean", 0.7, source="ndvi")]
    assert accepted(validate("Mean NDWI is .28 and mean NDVI is .7.", items))
    assert not accepted(validate("Mean NDWI is .7 and mean NDVI is .28.", items))
    assert not accepted(validate("The mean is .28.", items))  # identity is ambiguous


def test_reference_presence_and_wrong_existing_reference() -> None:
    items = [measured(), measured("ndvi_mean", 0.2777, source="ndvi")]
    text = "Mean NDWI is .28."
    assert accepted(validate(text, items, [items[0].id]))
    for refs in ([], [items[1].id], ["missing"]):
        result = validate(text, items, refs)
        assert result.numeric_grounding == "fail"
        assert result.evidence_refs == "fail"


@pytest.mark.parametrize(
    "text",
    [
        "Water is visible.",
        "The index rose.",
        "Vegetation is healthy.",
        "No water is visible.",
        "No imagery was retrieved.",
    ],
)
def test_substantive_prose_requires_matching_evidence(text: str) -> None:
    assert validate(text, []).evidence_refs == "fail"
    assert validate(text, [measured()]).evidence_refs == "fail"


def test_supported_number_cannot_launder_a_qualitative_conclusion() -> None:
    for text in (
        "NDWI is .28 so the area is flooded.",
        "NDWI is .28. Water is visible.",
        "NDWI is not .28.",
    ):
        assert validate(text, [measured()]).evidence_refs == "fail"


def visual(statement: str = "Water is visible along the shoreline.") -> EvidenceItem:
    return EvidenceItem(
        id="model.visual.SCENE",
        source="model",
        visual=VisualObservation(
            statement=statement, provider="gemini", model="test-model", scene_id="SCENE"
        ),
    )


def test_qualitative_claim_must_match_its_cited_observation() -> None:
    item = visual()
    assert accepted(validate(item.visual.statement, [item]))
    assert accepted(validate("Water is visible along the shoreline!", [item]))
    for text in (
        "No water is visible along the shoreline.",
        "The area is flooded.",
        "Water is visible along the shoreline. Vegetation is healthy.",
    ):
        assert validate(text, [item]).evidence_refs == "fail"
    assert validate(item.visual.statement, [item], []).evidence_refs == "fail"
    result = validate("NDWI is .28.", [measured(), item], ["ndwi.ndwi_mean"])
    assert accepted(result)
    assert result.visual_claims == "not_run"


def test_value_wrapper_preserves_measurement_identity_checks() -> None:
    text = "The mean NDWI value was 0.2777 index."
    assert accepted(validate(text, [measured()]))
    for items in (
        [],
        [measured(name="ndvi_mean", source="ndvi")],
        [measured(name="ndwi_max")],
        [measured(unit="pixels")],
        [measured(name="scene_count", value=1, unit="count", source="execution")],
        [measured(source="model")],
    ):
        assert not accepted(validate(text, items))
    for refs in ([], ["missing"]):
        assert not accepted(validate(text, [measured()], refs))
    assert not accepted(validate(text + " The area is flooded.", [measured()]))


def test_averaged_is_only_a_mean_statistic_alias() -> None:
    text = "NDWI averaged 0.2777 index."
    assert accepted(validate(text, [measured()]))
    for item in (measured(name="ndwi_max"), measured(name="ndvi_mean", source="ndvi"),
                 measured(unit="pixels"), measured(source="model")):
        assert not accepted(validate(text, [item]))
    assert not accepted(validate(text, []))
    assert not accepted(validate(text, [measured()], []))


def test_yes_wrapper_requires_the_exact_cited_observation() -> None:
    item = visual()
    text = "Yes, " + item.visual.statement
    assert accepted(validate(text, [item]))
    for items, refs in (([], []), ([item], []), ([item], ["missing"])):
        assert not accepted(validate(text, items, refs))
    for unsupported in (
        "Yes, no water is visible along the shoreline.",
        text + " The area is flooded.",
        "Yes, definitely, water is visible along the shoreline.",
        "Yes, vegetation is healthy.",
    ):
        assert not accepted(validate(unsupported, [item]))
    uncertain = visual("Water may be visible along the shoreline.")
    assert accepted(validate("Yes, " + uncertain.visual.statement, [uncertain]))
    assert not accepted(validate(text, [uncertain]))


def test_yes_wrapper_does_not_authorize_model_numbers() -> None:
    item = visual("Water covers .28 percent.")
    result = validate("Yes, " + item.visual.statement, [item, measured()])
    assert result.numeric_grounding == "fail"


def test_model_estimate_cannot_borrow_a_deterministic_number() -> None:
    item = visual("Water covers .28 percent.")
    result = validate(item.visual.statement, [item, measured()])
    assert result.numeric_grounding == "fail"


def test_model_coverage_estimate_cannot_borrow_a_threshold_percentage() -> None:
    item = visual("Water covers 44 percent.")
    statistic = measured("ndwi_percent_above_index_threshold_0.3", 44, "%")
    assert accepted(validate("44 percent of valid pixels.", [statistic]))
    assert validate(item.visual.statement, [item, statistic]).numeric_grounding == "fail"


def test_qualified_visual_observation_cannot_lose_uncertainty() -> None:
    item = visual("Water may be visible along the shoreline.")
    assert accepted(validate(item.visual.statement, [item]))
    assert not accepted(validate("Water is visible along the shoreline.", [item]))


def test_cited_limitation_is_supported_without_inventing_a_claim() -> None:
    item = EvidenceItem(
        id="compatibility.limitation.0",
        source="compatibility",
        text="The observations may cover different ground.",
    )
    assert accepted(validate(item.text, [item]))
    assert not accepted(validate("The observations cover identical ground.", [item]))


def test_narrow_abstention_remains_available_without_evidence() -> None:
    assert accepted(validate("Insufficient evidence to answer the question.", []))
    assert not accepted(
        validate("Insufficient evidence to answer the question, but water is visible.", [])
    )
    assert not accepted(
        validate("Insufficient evidence to answer the question. Water is visible.", [])
    )


@pytest.mark.parametrize("date", ["2024-01-15", "2024-01-01"])
def test_measurement_date_is_acquisition_not_just_requested_window(date: str) -> None:
    result = validate(
        f"Mean NDWI on {date} was .28 index.", [measured()], execution=make_execution()
    )
    assert accepted(result) == (date == "2024-01-15")


def test_measurement_cannot_move_to_another_discovered_scene() -> None:
    execution = make_execution()
    other = execution.windows[0].scenes[0].model_copy(update={"id": "S2_OTHER_20240115"})
    execution.windows[0].scenes.append(other)
    assert accepted(
        validate(
            f"Mean NDWI for scene {SCENE_ID} was .28 index.", [measured()], execution=execution
        )
    )
    result = validate(
        f"Mean NDWI for scene {other.id} was .28 index.", [measured()], execution=execution
    )
    assert result.numeric_grounding == "fail"


def test_known_calendar_year_cannot_authorize_an_index_value() -> None:
    result = validate("NDWI is 2024.", [measured()], execution=make_execution())
    assert result.numeric_grounding == "fail"
    assert not accepted(result)


def test_pixel_count_equal_to_a_year_is_still_a_measurement() -> None:
    item = measured("ndwi_valid_pixel_count", 2024, "pixels")
    assert accepted(
        validate("2024 valid pixels were analysed.", [item], execution=make_execution())
    )
    assert (
        validate(
            "2024 valid pixels were analysed.", [measured()], execution=make_execution()
        ).numeric_grounding
        == "fail"
    )


def test_calendar_year_is_bound_to_the_measured_observation() -> None:
    execution = make_execution()
    execution.windows[0].scenes[0].datetime = "2025-01-15T05:00:00Z"
    assert not accepted(validate("Mean NDWI in 2024 was .28.", [measured()], execution=execution))
    assert accepted(validate("Mean NDWI in 2025 was .28.", [measured()], execution=execution))


@pytest.mark.parametrize(
    "text",
    [
        "NDWI is .28 maximum.",
        "Mean NDWI .28 minimum NDWI .28.",
        "NDWI is .28 mean minimum.",
        "Mean NDWI is .28. No evidence was collected.",
    ],
)
def test_matching_number_cannot_hide_a_second_claim(text: str) -> None:
    assert not accepted(validate(text, [measured()]))


def test_optical_index_cannot_be_attributed_to_sar() -> None:
    assert accepted(validate("Sentinel-2 imagery gives a mean NDWI of .28.", [measured()]))
    assert not accepted(validate("Sentinel-1 imagery gives a mean NDWI of .28.", [measured()]))


def test_temporal_reference_identity_is_not_chosen_by_matching_value() -> None:
    first = measured(source="temporal_ndwi", item_id="temporal_ndwi.first.ndwi_mean", value=0.2)
    second = measured(source="temporal_ndwi", item_id="temporal_ndwi.second.ndwi_mean", value=0.5)
    analysis = AnalysisResult(
        status="ok",
        task="visualize",
        answer="",
        windows_considered=[],
        temporal_comparison=TemporalIndexComparison(
            first=index_result(window_label="target", mean=0.2),
            second=index_result(window_label="baseline", mean=0.5),
            compatibility=compat(),
        ),
    )
    items = [first, second]
    assert accepted(
        validate("First mean NDWI is .2 and second mean NDWI is .5.", items, analysis=analysis)
    )
    assert accepted(
        validate("Target mean NDWI is .2 and baseline mean NDWI is .5.", items, analysis=analysis)
    )
    for text in ("First mean NDWI is .5.", "Baseline mean NDWI is .2.", "Mean NDWI is .2."):
        assert validate(text, items, analysis=analysis).numeric_grounding == "fail"


def test_aggregate_and_paired_change_are_distinct_measurements() -> None:
    aggregate = measured(
        "mean_ndwi_difference",
        0.3,
        source="temporal_ndwi",
        item_id="temporal_ndwi.difference.mean_ndwi_difference",
    )
    paired = measured(
        "ndwi_change_mean",
        0.3,
        source="temporal_ndwi",
        item_id="temporal_ndwi.change.ndwi_change_mean",
    )
    assert accepted(validate("Mean NDWI difference is .3.", [aggregate]))
    assert accepted(validate("Mean NDWI change is .3.", [paired]))
    assert not accepted(validate("Mean NDWI difference is .3.", [paired]))
    assert not accepted(validate("Mean NDWI change is .3.", [aggregate]))
