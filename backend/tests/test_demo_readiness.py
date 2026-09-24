"""Demo readiness: places without lead words, and clarifications worth reading.

A. A place named without "around / in / at" is recognised when the question
   leaves ONE unambiguous name-like run - and asked back otherwise. The
   geocoder stays the authority on whether that run is a real place.
B. A clarification asks for everything missing at once, in plain words, with
   no placeholder and no internal name, and offers complete questions that ask
   only for what the user already said.
C. Rule gaps the v2 intent-model evaluation exposed: a real place whose name
   begins like a pronoun, auxiliaries read as names, change verbs, forecasting,
   degree questions ("how leafy is") and SAR's own quantity names.

The scientific pipeline is untouched: these are reader and wording changes.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

import pytest
from app.services.agent.interpretation import (
    ClarificationRequiredError,
    classifier_text,
    interpret,
)
from app.services.agent.schemas import AgentClarification
from pydantic import ValidationError

TODAY = date(2026, 9, 24)


def place(question: str) -> str:
    return interpret(question, today=TODAY).location_query


def ask(question: str) -> AgentClarification:
    with pytest.raises(ClarificationRequiredError) as raised:
        interpret(question, today=TODAY)
    return raised.value.clarification


# =========================================================================== #
# A. Places without lead words
# =========================================================================== #


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        # short queries
        ("NDVI Chennai January 2025", "Chennai"),
        ("water Dal Lake January 2025", "Dal Lake"),
        # location at the beginning / at the end
        ("Chennai NDVI January 2025", "Chennai"),
        ("NDVI in January 2025 Kochi", "Kochi"),
        # multi-word, commas, connectors
        ("NDVI Marina Beach, Chennai January 2025", "Marina Beach, Chennai"),
        ("Isle of Man NDVI 2024", "Isle of Man"),
        ("Rio de Janeiro NDVI January 2025", "Rio de Janeiro"),
        # apostrophes
        ("NDVI Chennai's Marina Beach January 2025", "Chennai's Marina Beach"),
        # all lowercase
        ("ndvi chennai january 2025", "chennai"),
        # coordinates
        ("NDVI 13.05, 80.28 January 2025", "13.05, 80.28"),
    ],
)
def test_a_place_without_a_lead_word_is_recognised(question: str, expected: str) -> None:
    assert place(question) == expected


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Show vegetation near Anna Nagar in March 2024", "Anna Nagar"),
        ("Show vegetation around Marina Beach, Chennai in January 2025",
         "Marina Beach, Chennai"),
        ("Show vegetation in Kochi in January 2025", "Kochi"),
        ("Show vegetation at Cubbon Park, Bengaluru in January 2025",
         "Cubbon Park, Bengaluru"),
        ("Show NDVI of Anna Nagar, Chennai in January 2025", "Anna Nagar, Chennai"),
    ],
)
def test_lead_words_still_work(question: str, expected: str) -> None:
    assert place(question) == expected


def test_a_short_query_keeps_its_place_while_asking_for_the_date() -> None:
    clarification = ask("vegetation Chennai")
    assert clarification.reason == "date_missing"
    assert clarification.understood_location == "Chennai"
    assert clarification.understood_analyses == ["vegetation (NDVI)"]


def test_analysis_words_never_leak_into_the_place() -> None:
    """The protections from M5.5 hold for the new pass too."""

    for question in ("built up Hyderabad in December 2024",
                     "Chennai NDVI January 2025",
                     "NDVI around Chennai NDVI January 2025",
                     "Isle of Man NDVI 2024"):
        location = place(question)
        assert not re.search(r"(?i)\b(ndvi|built|up|water|vegetation)\b", location), location
    assert classifier_text("Chennai NDVI January 2025") == "PLACE NDVI DATE"


def test_two_separate_names_are_asked_back_not_guessed() -> None:
    clarification = ask("Chennai NDVI Pune January 2025")
    assert clarification.reason == "location_missing"
    assert "'Chennai'" in clarification.message and "'Pune'" in clarification.message


def test_a_name_beside_a_capitalised_analysis_word_is_confirmed_not_cut() -> None:
    """Without a lead word, "Forest Hill" cannot be told from "forest" + "Hill"."""

    clarification = ask("water Forest Hill January 2025")
    assert clarification.reason == "location_missing"
    assert "'Forest Hill'" in clarification.message
    # With a lead word it is unambiguous, as before.
    assert place("Show water around Forest Hill in January 2025") == "Forest Hill"


def test_a_generic_noun_alone_is_not_a_place() -> None:
    assert ask("water lake January 2025").reason == "location_missing"


def test_a_deictic_place_is_still_asked_back() -> None:
    clarification = ask("Find water around this reservoir in January 2025")
    assert "'this reservoir'" in clarification.message


# =========================================================================== #
# B. Clarifications worth reading
# =========================================================================== #


def test_analyze_a_place_asks_what_to_analyse_there() -> None:
    clarification = ask("Analyze Chennai")

    assert clarification.reason == "analysis_missing"
    assert clarification.message == "What would you like to analyse at Chennai?"
    assert clarification.options == [
        "vegetation (NDVI)",
        "water (NDWI)",
        "built-up area (NDBI)",
        "SAR backscatter (Sentinel-1 VV/VH)",
        "water change between two periods (NDWI)",
    ]
    assert clarification.option_questions == [
        "Show vegetation (NDVI) around Chennai",
        "Show water (NDWI) around Chennai",
        "Show built-up area (NDBI) around Chennai",
        "Analyze SAR backscatter around Chennai",
        "Compare water (NDWI) around Chennai",
    ]


def test_a_suggested_question_carries_only_what_the_user_said() -> None:
    clarification = ask("Analyze Chennai in January 2025")
    assert clarification.option_questions[0] == (
        "Show vegetation (NDVI) around Chennai in January 2025"
    )


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Analyze Chennai in January 2025", ["ndvi", "ndwi", "ndbi", "sar_backscatter"]),
    ],
)
def test_every_suggested_question_reads_back_as_its_option(
    question: str, expected: list[str]
) -> None:
    """A suggestion is only useful if the interpreter reads it the way it says."""

    suggestions = ask(question).option_questions
    for suggestion, key in zip(suggestions, expected, strict=False):
        assert interpret(suggestion, today=TODAY).analyses == (key,), suggestion
    # The comparison option still needs a second period - and asks exactly that.
    assert ask(suggestions[-1]).reason == "comparison_incomplete"


def test_show_vegetation_asks_for_the_place_and_the_date_together() -> None:
    clarification = ask("Show vegetation")
    assert clarification.reason == "location_missing"
    assert "Which place" in clarification.message
    assert "month and year" in clarification.message


def test_compare_water_asks_for_the_place_and_both_periods() -> None:
    clarification = ask("Compare water")
    assert "Which place" in clarification.message
    assert "two periods" in clarification.message


def test_compare_water_at_a_place_asks_for_the_two_periods() -> None:
    clarification = ask("Compare water around Pune")
    assert clarification.reason == "date_missing"
    assert clarification.message.startswith("Which two periods should be compared?")


def test_a_missing_date_names_what_it_is_for() -> None:
    clarification = ask("Vegetation around Chennai")
    assert clarification.reason == "date_missing"
    assert clarification.message.startswith(
        "For which date or period should vegetation (NDVI) be measured?"
    )


@pytest.mark.parametrize(
    "question",
    [
        "Analyze Chennai", "Show vegetation", "Compare water", "Vegetation around Chennai",
        "Compare water around Pune", "Find water around this reservoir",
        "Chennai NDVI Pune January 2025", "water Forest Hill January 2025",
        "Count ships in Chennai harbor", "Show change", "Show vegetation around Chennai in May",
        "Is there visible water at Marina Beach, Chennai in January 2025?",
    ],
)
def test_no_clarification_shows_a_placeholder_or_an_internal_name(question: str) -> None:
    clarification = ask(question)
    visible = " ".join([clarification.message, *clarification.options,
                        *clarification.option_questions])
    assert "<" not in visible and ">" not in visible
    for internal in ("analysis_missing", "location_missing", "date_missing",
                     "temporal_ndwi", "sar_backscatter", "TEMPORAL_NDWI", "TRUE_COLOR",
                     "CLARIFICATION", "UNSUPPORTED"):
        assert internal not in visible, (question, internal)


def test_option_questions_must_match_the_options() -> None:
    with pytest.raises(ValidationError):
        AgentClarification(
            reason="analysis_missing", message="?", options=["a", "b"],
            option_questions=["only one"],
        )
    ok: Any = AgentClarification(reason="analysis_missing", message="?", options=["a"])
    assert ok.option_questions == []


# =========================================================================== #
# C. Rule gaps found by the v2 evaluation
# =========================================================================== #


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        # a name that merely BEGINS like "my", "it" or "here" is not deictic
        ("NDVI around Itanagar in January 2025", "Itanagar"),
        ("Water near Hereford in March 2024", "Hereford"),
        ("water analysis for Mysuru's Kukkarahalli lake in Feb 2025",
         "Mysuru's Kukkarahalli lake"),
        # an auxiliary opening the question is not a second candidate
        ("Did the Ujani backwaters shrink from January to May 2024", "Ujani"),
    ],
)
def test_names_are_not_mistaken_for_pronouns_or_auxiliaries(question: str, expected: str) -> None:
    from app.services.agent.interpretation import read_question

    assert read_question(question).place.text == expected


def test_a_real_deictic_is_still_a_deictic() -> None:
    from app.services.agent.interpretation import read_question

    for question in ("NDVI for my farm in 2024", "NDVI of this area in January 2025"):
        reading = read_question(question)
        assert reading.place.text is None and reading.place.deictic, question


def test_a_visual_phrase_ends_the_place() -> None:
    clarification = ask("Tell me what the image of Pune looks like in January 2025")
    assert clarification.reason == "requires_ai_model"
    assert clarification.understood_location == "Pune"


@pytest.mark.parametrize(
    "question",
    [
        "Has the water at Lonar lake risen between 2021 and 2024",
        "Water body shrinkage at Bellandur lake between 2018 and 2024",
        "Has water at Pulicat declined from January 2024 to January 2025",
        "Did the water at Chembarambakkam lake drop between 2023 and 2024",
    ],
)
def test_a_change_verb_over_water_is_a_comparison(question: str) -> None:
    try:
        result = interpret(question, today=TODAY)
    except ClarificationRequiredError as raised:
        # Anything but a single-period measurement is acceptable here.
        assert raised.clarification.reason != "analysis_missing", question
        return
    assert result.comparison and list(result.analyses) == ["temporal_ndwi"], question


def test_a_change_verb_over_vegetation_is_refused_not_measured_once() -> None:
    clarification = ask("Has the vegetation of Kodagu declined from 2020 to 2024?")
    assert clarification.reason == "analysis_unsupported"
    assert "water (NDWI) only" in clarification.message


@pytest.mark.parametrize(
    "question",
    [
        "Predict the crop yield near Bathinda for 2025",
        "Forecast vegetation around Pune for 2025",
        "What will the crop yield of Guntur be in 2024",
    ],
)
def test_yield_and_forecasting_are_refused(question: str) -> None:
    clarification = ask(question)
    assert clarification.reason == "analysis_unsupported"
    assert "forecasting are not implemented" in clarification.message


@pytest.mark.parametrize(
    "question",
    [
        "How leafy is Marina Beach, Chennai in January 2025",
        "How lush is Thekkady in September 2024",
        "Foliage around Cubbon Park, Bengaluru in January 2025",
    ],
)
def test_vegetation_adjectives_measure_ndvi(question: str) -> None:
    assert list(interpret(question, today=TODAY).analyses) == ["ndvi"]


def test_a_definition_is_still_not_a_request() -> None:
    from app.services.agent.interpretation import _ANALYSIS_TERMS
    from app.services.agent.plan_completion import requested_matches

    assert requested_matches("What is vegetation?", _ANALYSIS_TERMS["ndvi"]) == []
    assert requested_matches("How NDVI is computed", _ANALYSIS_TERMS["ndvi"]) == []
    assert requested_matches("How leafy is Chennai", _ANALYSIS_TERMS["ndvi"])


@pytest.mark.parametrize(
    "question",
    [
        "Sigma naught near Ennore creek in 2024",
        "gamma0 around Chennai port in January 2025",
    ],
)
def test_sar_quantity_names_measure_backscatter(question: str) -> None:
    assert list(interpret(question, today=TODAY).analyses) == ["sar_backscatter"]
