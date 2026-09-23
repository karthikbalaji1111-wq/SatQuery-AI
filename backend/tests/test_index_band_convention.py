"""The band mapping and sign convention of NDVI, NDWI and NDBI.

WHY THIS FILE EXISTS. A mutation audit found that swapping `high_band` and
`low_band` on NDWI - which negates every NDWI value ever reported - passed the
ENTIRE suite (1658 tests) without a single failure. The existing tests compute
their expectations from the same registry entry they are exercising, so they
verify the arithmetic is applied consistently but say nothing about which band
belongs on top. A sign inversion is the worst kind of silent error here: the
magnitudes stay plausible, the statistics stay well-formed, and a water body
would simply be reported as the opposite of what it is.

These tests deliberately do NOT read `high_band`/`low_band` to build their
expectations. They assert the mapping literally, and then assert the SIGN that
the physics requires on synthetic scenes whose correct answer is known
independently of any code in this repository:

  * water reflects strongly in green and absorbs in NIR  -> NDWI > 0
  * healthy vegetation reflects strongly in NIR, absorbs red -> NDVI > 0
  * built-up/bare ground is brighter in SWIR than NIR   -> NDBI > 0

Any implementation that satisfies all three cannot have a pair swapped.
"""

from __future__ import annotations

import numpy as np
import pytest
from app.services.analysis.engines import _normalised_difference_values
from app.services.analysis.indices import NDBI, NDVI, NDWI, resolve_index

from tests.test_analysis import band

# --------------------------------------------------------------------------- #
# The literal mapping, stated once, in the direction the science defines it.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("index", "high", "low"),
    [
        # NDWI = (Green - NIR) / (Green + NIR)   McFeeters 1996
        (NDWI, "green", "nir"),
        # NDVI = (NIR - Red) / (NIR + Red)       Rouse 1974
        (NDVI, "nir", "red"),
        # NDBI = (SWIR - NIR) / (SWIR + NIR)     Zha 2003
        (NDBI, "swir16", "nir"),
    ],
)
def test_the_band_on_top_of_the_numerator_is_the_one_the_definition_names(
    index, high: str, low: str
) -> None:
    assert index.high_band == high, f"{index.label} numerator band is wrong"
    assert index.low_band == low, f"{index.label} subtracted band is wrong"
    assert index.high_band != index.low_band


def test_the_registry_is_the_closed_set_of_three() -> None:
    assert {resolve_index(k).key for k in ("ndwi", "ndvi", "ndbi")} == {
        "ndwi",
        "ndvi",
        "ndbi",
    }


# --------------------------------------------------------------------------- #
# The sign, from physics. This is the half a band swap cannot survive.
# --------------------------------------------------------------------------- #


#: Reflectance of each Sentinel-2 band over one surface, in raw DN. These are
#: statements about the WORLD, not about this codebase: they say how a surface
#: behaves, and the index definitions are what must agree with them.
_SURFACES: dict[str, dict[str, float]] = {
    # Open water: strong green return, near-total NIR and SWIR absorption.
    "water": {"green": 2400, "red": 1500, "nir": 600, "swir16": 300},
    # Healthy vegetation: the NIR plateau, the red chlorophyll trough.
    "vegetation": {"green": 900, "red": 500, "nir": 3800, "swir16": 1500},
    # Built-up / bare ground: brighter in SWIR than in NIR.
    "built_up": {"green": 1700, "red": 1900, "nir": 1800, "swir16": 2600},
}


def _value(index, surface: str) -> float:
    """The index over one surface, resolved through the registry as production does.

    `AnalysisService` looks each band up BY NAME - `bands[index.high_band]` and
    `bands[index.low_band]` - so this mirrors that exactly. The caller names a
    surface, never an argument order, which is what makes these assertions bind
    to the MAPPING: swap `high_band` and `low_band` on an index and the value
    computed here flips sign, exactly as the shipped analysis would.
    """

    reflectance = _SURFACES[surface]
    high = band([reflectance[index.high_band]] * 16)
    low = band([reflectance[index.low_band]] * 16)
    values = _normalised_difference_values(high, low, index.label)
    return float(np.asarray(values).mean())


def test_water_gives_a_positive_ndwi() -> None:
    assert _value(NDWI, "water") > 0


def test_dense_vegetation_gives_a_negative_ndwi() -> None:
    """The counter-case: vegetation is bright in NIR, so NDWI must go negative.

    Without this, an implementation that returned |value| would still pass.
    """

    assert _value(NDWI, "vegetation") < 0


def test_dense_vegetation_gives_a_positive_ndvi() -> None:
    assert _value(NDVI, "vegetation") > 0


def test_bare_soil_gives_a_negative_ndvi() -> None:
    assert _value(NDVI, "built_up") < 0


def test_built_up_gives_a_positive_ndbi() -> None:
    assert _value(NDBI, "built_up") > 0


def test_vegetation_gives_a_negative_ndbi() -> None:
    assert _value(NDBI, "vegetation") < 0


# --------------------------------------------------------------------------- #
# Cross-index: no two indices may collapse onto the same band pair.
# --------------------------------------------------------------------------- #


def test_the_three_indices_use_three_distinct_band_pairs() -> None:
    """A copy-paste that duplicated one entry would otherwise be invisible.

    NIR appears in all three, which is exactly why the PAIR - and its order -
    is what has to be distinct, not merely the set of bands used.
    """

    pairs = [(i.high_band, i.low_band) for i in (NDWI, NDVI, NDBI)]
    assert len(set(pairs)) == 3
    # And no pair is another pair reversed, which would make one index the
    # exact negation of another.
    assert not any(
        (a, b) == (d, c) for (a, b) in pairs for (c, d) in pairs if (a, b) != (c, d)
    )


# --------------------------------------------------------------------------- #
# Which physical Sentinel-2 band each asset key actually is.
#
# The tests above pin the ORDER of the band pair. They say nothing about whether
# the key "green" is really B03 - and an index computed from the wrong physical
# band is wrong however correctly its arithmetic is arranged.
#
# The expectations below are the mission's own, not this repository's: the band
# numbers and centre wavelengths are properties of Sentinel-2's MSI instrument.
# They were confirmed against Earth Search's `eo:bands` metadata on scene
# S2C_44PMV_20250129_0_L2A, which reports for each asset the band name, the
# common name and the centre wavelength in micrometres:
#
#     visual -> B04/B03/B02 (0.665 / 0.560 / 0.490)   10 m
#     red    -> B04 (0.665)                            10 m
#     green  -> B03 (0.560)                            10 m
#     nir    -> B08 (0.842)                            10 m
#     swir16 -> B11 (1.610)                            20 m
#
# Asserting the wavelength ORDERING rather than the numbers themselves keeps
# this a statement about physics: green is shorter than red, red shorter than
# NIR, NIR shorter than SWIR. No band substitution can satisfy all three.
# --------------------------------------------------------------------------- #

#: Centre wavelength in micrometres, from the Sentinel-2 MSI specification.
_CENTRE_UM = {
    "blue": 0.490,
    "green": 0.560,
    "red": 0.665,
    "nir": 0.842,
    "swir16": 1.610,
}


def test_the_asset_keys_are_ordered_by_wavelength_as_the_spectrum_requires() -> None:
    """Blue < green < red < NIR < SWIR. A swapped key breaks the ordering."""

    keys = ["blue", "green", "red", "nir", "swir16"]
    centres = [_CENTRE_UM[k] for k in keys]

    assert centres == sorted(centres)


@pytest.mark.parametrize(
    ("index", "high_um", "low_um"),
    [
        # NDWI subtracts NIR from green: the numerator's positive term is the
        # SHORTER wavelength. That is what makes water - which absorbs NIR -
        # come out positive.
        (NDWI, _CENTRE_UM["green"], _CENTRE_UM["nir"]),
        # NDVI subtracts red from NIR: positive term is the LONGER wavelength,
        # because vegetation's NIR plateau sits above its red trough.
        (NDVI, _CENTRE_UM["nir"], _CENTRE_UM["red"]),
        # NDBI subtracts NIR from SWIR: longer again.
        (NDBI, _CENTRE_UM["swir16"], _CENTRE_UM["nir"]),
    ],
)
def test_each_index_uses_the_wavelengths_its_definition_requires(
    index, high_um: float, low_um: float
) -> None:
    assert _CENTRE_UM[index.high_band] == high_um, index.label
    assert _CENTRE_UM[index.low_band] == low_um, index.label


def test_ndbi_is_the_only_index_reaching_beyond_the_10_m_bands() -> None:
    """SWIR is delivered at 20 m, which is why NDBI is limited to 20 m.

    Pinned because the limiting resolution and the band choice have to move
    together: an NDBI quietly switched to a 10 m band would keep reporting a
    20 m limit it no longer had.
    """

    assert NDBI.low_band == "nir"
    assert NDBI.high_band == "swir16"
    assert NDBI.limiting_resolution_m == 20.0
    assert NDWI.limiting_resolution_m == 10.0
    assert NDVI.limiting_resolution_m == 10.0


# --------------------------------------------------------------------------- #
# Why raw digital numbers are safe: which transform cancels, and which does not.
#
# `indices.py` used to argue that raw DN was fine because a shared scale AND
# offset cancel in a normalised difference. Half of that is an identity; the
# other half is false, and stating it invited someone to apply the advertised
# offset on the strength of an argument that never covered it.
#
# These assertions are arithmetic on plain numbers - no repository code is
# involved in computing the expectation - so they check the CLAIM, not the
# implementation that relies on it.
# --------------------------------------------------------------------------- #


def _nd(a: float, b: float) -> float:
    return (a - b) / (a + b)


@pytest.mark.parametrize("scale", [0.0001, 1.0, 2.5, 1e6])
def test_a_common_multiplicative_scale_cancels_exactly(scale: float) -> None:
    """The half that IS an identity, over a range of magnitudes."""

    a, b = 2400.0, 600.0

    assert _nd(a * scale, b * scale) == pytest.approx(_nd(a, b), rel=1e-12)


@pytest.mark.parametrize("offset", [-0.1, -1000.0, 250.0])
def test_a_common_additive_offset_does_not_cancel(offset: float) -> None:
    """The half that is NOT. The denominator keeps 2c; the value moves."""

    a, b = 2400.0, 600.0

    assert _nd(a + offset, b + offset) != pytest.approx(_nd(a, b), rel=1e-9)


def test_the_offset_form_matches_the_documented_algebra() -> None:
    """((a+c)-(b+c)) / ((a+c)+(b+c)) == (a-b) / (a+b+2c), exactly."""

    a, b, c = 2400.0, 600.0, -0.1

    assert _nd(a + c, b + c) == pytest.approx((a - b) / (a + b + 2 * c), rel=1e-12)


def test_the_repository_does_not_apply_the_advertised_scale_or_offset() -> None:
    """The decision the note explains, pinned to the flag that records it."""

    from app.services.analysis.engines import STAC_SCALE_OFFSET_APPLIED

    assert STAC_SCALE_OFFSET_APPLIED is False
