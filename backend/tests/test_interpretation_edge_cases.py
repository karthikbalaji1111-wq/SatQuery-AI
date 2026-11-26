"""Reader defects found by a fresh, unseen question set (production audit).

Both were found by running 41 newly written questions through the full
operation decision (model + rules). Neither executed a wrong analysis; both
produced a worse answer than the rules intend.
"""

from __future__ import annotations

from datetime import date

import pytest
from app.services.agent.interpretation import ClarificationRequiredError, interpret

TODAY = date(2026, 11, 26)


@pytest.mark.parametrize(
    "question",
    [
        "water extent at Hussain Sagar, Hyderabad: March 2023 versus March 2024",
        "NDWI at Hussain Sagar, Hyderabad; March 2024",
    ],
)
def test_trailing_punctuation_never_reaches_the_geocoder(question: str) -> None:
    assert interpret(question, today=TODAY).location_query == "Hussain Sagar, Hyderabad"


@pytest.mark.parametrize(
    "question",
    [
        "How deep is Ulsoor Lake in January 2025?",
        "how deep is the Malampuzha dam in 2024",
    ],
)
def test_a_depth_question_is_refused_not_read_as_a_definition(question: str) -> None:
    with pytest.raises(ClarificationRequiredError) as caught:
        interpret(question, today=TODAY)
    assert caught.value.clarification.reason == "analysis_unsupported"


def test_a_real_definition_question_is_still_a_definition() -> None:
    # "What is NDVI?" defines the index; it must not run it.
    with pytest.raises(ClarificationRequiredError) as caught:
        interpret("What is NDVI?", today=TODAY)
    assert caught.value.clarification.reason != "analysis_unsupported"
