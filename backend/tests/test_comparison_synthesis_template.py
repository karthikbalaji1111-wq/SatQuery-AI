"""The comparison answer the synthesiser is told to write is one grounding accepts.

Observed live: for a two-date water question NVIDIA followed the single-scene
template literally - "The mean NDWI was 0.02665 index. The mean NDWI was 0.1464
index." - and grounding correctly refused it, because a bare "mean NDWI" is
ambiguous when two observations exist. The instruction now carries an explicit
comparison template; this pins that every sentence in it is groundable, so the
prompt can never ask for an answer the validator must withhold.
"""

from __future__ import annotations

from app.services.agent.prompts import _SYNTHESIS_INSTRUCTION
from app.services.analysis.schemas import AnalysisResult, TemporalIndexComparison

from tests.test_grounding_hardening import accepted, measured, validate
from tests.test_temporal_ndwi import compat, index_result

TEMPLATE = (
    '"The earlier mean NDWI was <value> index.", "The later mean\n'
    '  NDWI was <value> index." and "The mean NDWI difference was <value> index."'
)


def _comparison():  # type: ignore[no-untyped-def]
    items = [
        measured(source="temporal_ndwi", item_id="temporal_ndwi.first.ndwi_mean", value=0.2),
        measured(source="temporal_ndwi", item_id="temporal_ndwi.second.ndwi_mean", value=0.5),
        measured(
            "mean_ndwi_difference",
            0.3,
            source="temporal_ndwi",
            item_id="temporal_ndwi.difference.mean_ndwi_difference",
        ),
    ]
    analysis = AnalysisResult(
        status="ok",
        task="visualize",
        answer="",
        windows_considered=[],
        temporal_comparison=TemporalIndexComparison(
            first=index_result(window_label="baseline", mean=0.2),
            second=index_result(window_label="target", mean=0.5),
            compatibility=compat(),
        ),
    )
    return items, analysis


def test_the_instruction_carries_the_comparison_template() -> None:
    assert TEMPLATE in _SYNTHESIS_INSTRUCTION
    assert "never a shortened prefix" in _SYNTHESIS_INSTRUCTION


def test_the_template_answer_is_grounded() -> None:
    items, analysis = _comparison()
    text = (
        "The earlier mean NDWI was 0.2 index. The later mean NDWI was 0.5 index. "
        "The mean NDWI difference was 0.3 index."
    )
    assert accepted(validate(text, items, analysis=analysis))


def test_the_single_scene_form_is_refused_for_a_comparison() -> None:
    """Why the template exists: with two observations this is ambiguous."""

    items, analysis = _comparison()
    assert not accepted(validate("The mean NDWI was 0.2 index.", items, analysis=analysis))


def test_the_template_cannot_swap_the_two_observations() -> None:
    items, analysis = _comparison()
    assert not accepted(validate("The earlier mean NDWI was 0.5 index.", items, analysis=analysis))
