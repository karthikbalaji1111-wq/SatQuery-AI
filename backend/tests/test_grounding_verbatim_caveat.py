"""A cited caveat repeated verbatim is the engine's statement, not a claim.

Observed live with the local model: asked for NDBI, it gave the correct mean
AND repeated the engine's own resolution caveat word for word - "NDBI uses a
20 m band, so it is sampled on the 10 m grid but resolves detail no finer than
20 m." The prose check accepted that sentence as a repetition of cited
evidence; the numeric check read 20, 10 and 20 as unsupported measurements and
withheld the whole, correct answer. The two checks disagreed about one
sentence the system itself wrote.

The allowance is deliberately narrow, and each test below pins one edge of it:
a WHOLE sentence, of a CITED caveat (a warning or limitation from a
deterministic source), with every figure unaltered. Model observations never
qualify, and nothing here makes a number citable anywhere else.
"""

from __future__ import annotations

from typing import Any

from app.services.agent.grounding import validate_answer
from app.services.agent.schemas import AnswerValidation

from tests.test_agent_grounding import SCENE_ID, draft, make_evidence, measurement_item

CAVEAT = (
    "NDBI uses a 20 m band, so it is sampled on the 10 m grid but resolves "
    "detail no finer than 20 m."
)
MEAN = "The mean NDBI was 0.01184 index."
REFS = ["ndbi.ndbi_mean", "execution.warning.0"]


def caveat(text: str = CAVEAT, item_id: str = "execution.warning.0") -> dict[str, Any]:
    return {
        "id": item_id,
        "source": item_id.split(".")[0],
        "text": text,
        "produced_by": "analysis.service",
    }


def validate(
    summary: str,
    *,
    refs: list[str] | None = None,
    items: list[dict[str, Any]] | None = None,
) -> AnswerValidation:
    evidence = make_evidence(
        items=[
            measurement_item("ndbi.ndbi_mean", "ndbi_mean", 0.011835606151453555, "index"),
            caveat(),
            *(items or []),
        ]
    )
    return validate_answer(draft(summary, REFS if refs is None else refs), evidence)


def shown(result: AnswerValidation) -> bool:
    return (result.numeric_grounding, result.forbidden_terms, result.evidence_refs) == (
        "pass",
        "pass",
        "pass",
    )


def test_the_live_ndbi_answer_with_its_caveat_is_shown() -> None:
    """The exact shape observed live: correct mean, caveat copied, both cited."""

    assert shown(validate(f"{MEAN} {CAVEAT}"))


def test_the_caveat_alone_is_shown_when_cited() -> None:
    assert shown(validate(CAVEAT, refs=["execution.warning.0"]))


def test_every_engines_caveat_gets_the_same_treatment() -> None:
    """The SAR caveat carries the same trap: the 10 of 10*log10 read as a claim."""

    text = (
        "Backscatter is averaged in linear gamma-naught power and the mean is "
        "then converted with 10*log10."
    )
    item = caveat(text, "sar_backscatter.warning.0")
    assert shown(validate(text, refs=[item["id"]], items=[item]))


def test_a_caveat_with_one_figure_altered_is_still_checked() -> None:
    altered = CAVEAT.replace("no finer than 20 m", "no finer than 30 m")
    assert validate(f"{MEAN} {altered}").numeric_grounding == "fail"


def test_a_changed_sign_is_a_different_sentence() -> None:
    """Prose support compares words only; this allowance keeps every sign."""

    kept = "Pixels with NDWI below -0.2 were excluded."
    item = caveat(kept, "temporal_ndwi.warning.0")
    refs = [item["id"]]
    assert shown(validate(kept, refs=refs, items=[item]))
    flipped = kept.replace("-0.2", "0.2")
    assert validate(flipped, refs=refs, items=[item]).numeric_grounding == "fail"


def test_a_caveat_copied_without_citing_it_is_still_checked() -> None:
    result = validate(f"{MEAN} {CAVEAT}", refs=["ndbi.ndbi_mean"])
    assert result.numeric_grounding == "fail"


def test_a_caveat_folded_into_a_longer_sentence_is_still_checked() -> None:
    result = validate(f"The mean NDBI was 0.01184 index, and {CAVEAT}")
    assert result.numeric_grounding == "fail"


def test_the_caveats_figures_become_citable_nowhere_else() -> None:
    result = validate(f"{MEAN} {CAVEAT} The mean NDBI was 20 index.")
    assert result.numeric_grounding == "fail"


def test_an_invented_value_beside_a_verbatim_caveat_is_still_caught() -> None:
    assert validate(f"The mean NDBI was 0.5 index. {CAVEAT}").numeric_grounding == "fail"


def test_a_model_observation_repeated_verbatim_never_authorises_its_figures() -> None:
    statement = "About 40 percent of the scene is water."
    observation = {
        "id": f"model.visual.{SCENE_ID}",
        "source": "model",
        "visual": {
            "statement": statement,
            "provider": "local",
            "model": "qwen3-vl:4b-instruct",
            "scene_id": SCENE_ID,
        },
    }
    result = validate(statement, refs=[observation["id"]], items=[observation])
    assert result.numeric_grounding == "fail"


def test_a_failure_note_is_not_a_caveat() -> None:
    """The executor's failure notes relay an error message, not an engine's caveat."""

    note = caveat(
        "The analysis did not complete: 42 bands were unreadable.",
        "execution.analysis_failure",
    )
    result = validate(note["text"], refs=[note["id"]], items=[note])
    assert result.numeric_grounding == "fail"
