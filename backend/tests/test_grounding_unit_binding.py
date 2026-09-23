"""A unit claim binds to its measurement wherever it appears in the clause.

The defect: `_supporting_measurements` split the clause at the numeric literal
and read only the text AFTER it. A unit stated ahead of the number was invisible,
so all three of these were accepted against `index` or `dB` evidence they
contradicted:

    "Mean NDWI in dB is 0.2777."
    "Mean NDWI in dB is 0.2777 index."
    "Mean VV in index is -4.392."

Every number in them is real and traceable. What was false was the UNIT - a
reader is told the water index is a decibel figure, or that backscatter is
unitless. Checking one side of the number checked half the claim.

Note the trailing forms ("0.2777 dB NDWI", "-4.392 index VV") were already
refused; this file pins both sides so neither can regress.

The expected values are stated literally rather than read from the evidence
objects, so the assertions cannot drift with the fixtures.
"""

from __future__ import annotations

import pytest

from tests.test_agent_sar import accepted, measured, sar_item, validate

NDWI = measured("ndwi_mean", 0.2777, "index", "ndwi")
VV = sar_item("vv_mean_db", -4.392, "dB")


# --------------------------------------------------------------------------- #
# Wrong unit, whichever side of the number it is stated on.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "claim",
    [
        # Leading "in <unit>" - the three that regressed.
        "Mean NDWI in dB is 0.2777.",
        "Mean NDWI in dB is 0.2777 index.",
        "Mean NDWI expressed in dB is 0.2777.",
        "Mean NDWI measured in decibels is 0.2777.",
        # Trailing - already refused, pinned so it stays that way.
        "Mean NDWI is 0.2777 dB.",
        "0.2777 dB NDWI",
        # A length unit against an index value.
        "Mean NDWI is 0.2777 metres.",
        "Mean NDWI in metres is 0.2777.",
    ],
)
def test_an_index_value_is_never_authorised_as_another_unit(claim: str) -> None:
    assert not accepted(validate(claim, [NDWI]))


@pytest.mark.parametrize(
    "claim",
    [
        "Mean VV in index is -4.392.",
        "Mean VV expressed in index is -4.392.",
        "Mean VV is -4.392 index.",
        "-4.392 index VV",
        "Mean VV is -4.392 metres.",
        "Mean VV in metres is -4.392.",
    ],
)
def test_a_decibel_value_is_never_authorised_as_another_unit(claim: str) -> None:
    assert not accepted(validate(claim, [VV]))


@pytest.mark.parametrize(
    "claim",
    [
        # Wrong leading, right trailing. Refused even without the
        # contradiction rule, because the leading unit is then adopted and
        # fails the evidence comparison.
        "Mean VV in index is -4.392 dB.",
        # Right leading, WRONG trailing. This one needs the contradiction rule
        # specifically: adopting the (correct) leading unit would match the
        # evidence and let the false trailing unit through unexamined.
        "Mean VV in dB is -4.392 index.",
        "Mean NDWI in index is 0.2777 dB.",
    ],
)
def test_a_clause_naming_two_different_units_for_one_value_is_refused(
    claim: str,
) -> None:
    """Self-contradictory: it cannot be both, so neither reading is accepted."""

    items = [VV] if "VV" in claim else [NDWI]
    assert not accepted(validate(claim, items))


# --------------------------------------------------------------------------- #
# Correct claims must keep passing. Without these, refusing everything would
# satisfy every case above while withholding perfectly grounded answers.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("claim", "items"),
    [
        ("Mean NDWI is 0.2777.", [NDWI]),
        ("Mean NDWI is 0.2777 index.", [NDWI]),
        ("The mean NDWI is 0.2777 index.", [NDWI]),
        ("Mean VV is -4.392 dB.", [VV]),
        ("The mean VV is -4.392 dB.", [VV]),
        # The unit stated ahead of the number, and CORRECT.
        ("Mean VV in dB is -4.392.", [VV]),
        ("Measured in dB, the mean VV is -4.392.", [VV]),
        ("Mean NDWI in index is 0.2777.", [NDWI]),
    ],
)
def test_a_correct_claim_still_passes(claim: str, items: list[object]) -> None:
    assert accepted(validate(claim, items))  # type: ignore[arg-type]


def test_an_unrecognised_word_after_in_is_not_read_as_a_unit() -> None:
    """"in the bay" states no unit; a unit must not be invented from prose.

    This clause is refused for a DIFFERENT, pre-existing reason - grounding
    fails closed on vocabulary it does not recognise - so the assertion here is
    about the unit reader specifically: it must return None rather than treat
    "the" or "bay" as a unit claim.
    """

    from app.services.agent.grounding import _leading_unit

    assert _leading_unit("Mean NDWI in the bay is ") is None
    assert _leading_unit("Mean NDWI in the surveyed area is ") is None
    assert _leading_unit("Mean NDWI in dB is ") == "dB"
    assert _leading_unit("Mean NDWI in index is ") == "index"
