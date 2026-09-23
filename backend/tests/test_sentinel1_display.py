"""Sentinel-1 RTC display rendering, provenance wording and refusal messages.

These tests pin the parts of the SAR path a reader is entitled to trust: which
stretch was applied, that the stretch is described accurately, that terrain
correction is attributed to the data provider rather than to SatQuery, and that
the quantitative path still refuses SAR entirely.
"""
from __future__ import annotations

import httpx
import numpy as np
import pytest
from app.core.config import Settings
from app.core.errors import InvalidInputError, UpstreamServiceError
from app.services.satellite.imagery import (
    ImageryService,
    _require_readable_scheme,
)
from app.services.satellite.raster import (
    _SAR_DB_NORMALIZATION,
    _SAR_NORMALIZATION,
    _normalize_sar_band,
    normalize_sar_display,
)
from app.services.satellite.rtc import RTC_COLLECTION, sign_rtc_asset
from app.services.satellite.schemas import ImageryRequest
from rasterio.windows import Window

from tests.test_imagery import bbox_for_window, synthetic_raster
from tests.test_sentinel1_rtc import HREF

# =========================================================================== #
# Which stretch runs, and why
#
# Sentinel-1 RTC gamma naught is LINEAR POWER. Its distribution is heavily
# right-skewed, so a linear percentile stretch collapses the land surface into
# the bottom of the output range. The decibel stretch is a display transform
# over the provider's untouched values - not a calibration.
# =========================================================================== #


def _positive_power(shape: tuple[int, int] = (64, 64)) -> np.ndarray:
    """A synthetic gamma-naught field with the skew real backscatter has.

    Built the way the real quantity is distributed: roughly normal in DECIBELS
    (here mean -14 dB, sd 6 dB, against a live Chennai VV window measured at p2
    -21.8, p50 -14.1, p98 +4.5 dB), which is severely right-skewed once
    converted to linear power. The skew is the whole reason the stretch matters,
    so a synthetic fixture that lacked it would prove nothing.
    """

    decibels = np.random.default_rng(7).normal(-14.0, 6.0, size=shape)
    return 10.0 ** (decibels / 10.0)


def test_strictly_positive_power_is_stretched_in_decibels() -> None:
    out, label = normalize_sar_display(_positive_power(), nodata=None)
    assert label == _SAR_DB_NORMALIZATION
    assert "decibel" in label
    assert out.dtype == np.uint8


def test_a_non_positive_sample_falls_back_to_the_linear_stretch() -> None:
    """A logarithm is undefined at or below zero, so the branch is refused.

    The fallback is not a lesser rendering chosen for convenience: a band
    carrying zero or negative samples is not a power quantity, so calling the
    result decibels would be a false label.
    """

    values = _positive_power()
    values[0, 0] = 0.0
    _, label = normalize_sar_display(values, nodata=None)
    assert label == _SAR_NORMALIZATION
    assert "decibel" not in label

    values[0, 0] = -3.0  # e.g. a band already expressed in dB
    _, label = normalize_sar_display(values, nodata=None)
    assert label == _SAR_NORMALIZATION


def test_the_decibel_stretch_preserves_order() -> None:
    """Monotonic: brighter backscatter is never rendered darker.

    This is what makes the transform a stretch rather than an interpretation -
    it changes contrast and nothing else.
    """

    values = np.sort(_positive_power().ravel()).reshape(64, 64)
    out, label = normalize_sar_display(values, nodata=None)
    assert label == _SAR_DB_NORMALIZATION
    flat = out.ravel()
    assert np.all(np.diff(flat.astype(np.int16)) >= 0)


def test_the_decibel_stretch_is_why_it_was_chosen() -> None:
    """The measured justification, pinned so it cannot silently regress.

    On a live Chennai RTC VV window the linear stretch put 75.1% of pixels in
    the darkest tenth of the output range with a median of 3/255; in decibels
    the same window put 13.8% there with a median of 75/255. That behaviour is
    a property of the skewed distribution, so it reproduces synthetically.
    """

    values = _positive_power((256, 256))
    db, db_label = normalize_sar_display(values, nodata=None)

    # Compare like with like: the SAME numbers down the linear branch, forced
    # there by one zero sample the stretch simply clips.
    mixed = values.copy()
    mixed[0, 0] = 0.0
    linear, linear_label = normalize_sar_display(mixed, nodata=None)
    assert (db_label, linear_label) == (_SAR_DB_NORMALIZATION, _SAR_NORMALIZATION)

    dark = 255 * 0.1
    assert (linear < dark).mean() > 0.5  # linear buries the bulk of the scene
    assert (db < dark).mean() < 0.25  # decibels do not
    assert np.median(db) > 4 * np.median(linear)


def test_invalid_pixels_stay_black_and_out_of_the_decibel_statistics() -> None:
    values = _positive_power()
    values[0, 0] = np.nan
    values[0, 1] = np.inf
    values[1, :] = -9999.0  # nodata, and non-positive
    out, label = normalize_sar_display(values, nodata=-9999.0)

    # The nodata rows are excluded BEFORE the branch is chosen, so a negative
    # nodata sentinel cannot push a genuine power band onto the linear branch.
    assert label == _SAR_DB_NORMALIZATION
    assert out[0, 0] == 0 and out[0, 1] == 0
    assert (out[1, :] == 0).all()
    assert out[2:].max() > 0


def test_a_constant_positive_window_is_still_flat_mid_grey() -> None:
    out, label = normalize_sar_display(np.full((8, 8), 0.05), nodata=None)
    assert label == _SAR_DB_NORMALIZATION
    assert set(np.unique(out).tolist()) == {128}


def test_the_array_only_wrapper_agrees_with_the_labelled_form() -> None:
    values = _positive_power()
    assert np.array_equal(
        _normalize_sar_band(values, nodata=None),
        normalize_sar_display(values, nodata=None)[0],
    )


# =========================================================================== #
# Provenance wording
#
# The provider terrain-corrected these values. SatQuery did not, and the
# response must not let a reader conclude otherwise.
# =========================================================================== #


def _retrieve_rtc(monkeypatch: pytest.MonkeyPatch, asset: str = "vv"):
    from app.services.satellite import raster

    href = HREF.replace("iw-vv", f"iw-{asset}")

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/sign"):
            return httpx.Response(200, json={"href": href + "?sig=test-only-token"})
        return httpx.Response(200, json={"assets": {
            asset: {"href": href, "type": "image/tiff; application=geotiff"},
        }})

    data = _positive_power((20, 20)).astype("float32")
    with synthetic_raster(count=1, dtype="float32", data=data) as mem:
        monkeypatch.setattr(raster, "_open_raster", lambda _url: mem.open())
        return ImageryService(transport=httpx.MockTransport(respond)).retrieve(
            ImageryRequest(
                scene_id="test_rtc", collection=RTC_COLLECTION, asset=asset,
                bbox=bbox_for_window(Window(0, 0, 20, 20)),
            )
        )


def test_rtc_response_attributes_terrain_correction_to_the_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    note = _retrieve_rtc(monkeypatch).normalization
    assert "Provider RTC gamma naught" in note
    assert "terrain-corrected by the data provider, not by SatQuery" in note


def test_rtc_response_claims_no_calibration_it_did_not_perform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    note = _retrieve_rtc(monkeypatch).normalization
    # Still disclaimed, because SatQuery still does none of these.
    assert "calibration" in note
    assert "speckle filtering" in note
    assert "terrain correction of its own" in note
    # But it must NOT deny quantitative backscatter analysis any more. That
    # denial was true when only a display path existed; it is now false, and it
    # rendered directly above the dB statistics in the UI - the response
    # contradicting itself on one screen. Same lesson as the dB-conversion test
    # below: a note may disclaim what the system does not do, never what it does.
    assert "no local calibration, " not in note
    assert "quantitative backscatter analysis is performed" not in note
    # And it must say where the real numbers come from, so the display bytes
    # are never mistaken for the measured values.
    assert "linear power" in note


def test_rtc_response_does_not_deny_the_decibel_step_it_now_performs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wording used to promise "no ... dB conversion" while stretching in dB.

    A display stretch in decibels is legitimate; describing it as absent is
    not. The response must name the step it took.
    """

    note = _retrieve_rtc(monkeypatch).normalization
    assert "no local calibration, dB conversion" not in note
    assert "decibels" in note
    assert "display only" in note


@pytest.mark.parametrize("asset", ["vv", "vh"])
def test_rtc_response_never_leaks_the_signature(
    monkeypatch: pytest.MonkeyPatch, asset: str,
) -> None:
    result = _retrieve_rtc(monkeypatch, asset)
    assert "test-only-token" not in result.model_dump_json()
    assert "sig=" not in result.asset_href


# =========================================================================== #
# Refusals stay refusals, and say something a reader can act on
# =========================================================================== #


def test_the_s3_refusal_names_the_route_that_does_work() -> None:
    """The old message declared bounded Sentinel-1 retrieval unavailable.

    That was true of Earth Search's requester-pays GRD assets and false of the
    system as a whole once RTC landed. A refusal that misstates the system's
    capability sends a reader to fix the wrong thing.
    """

    with pytest.raises(InvalidInputError) as excinfo:
        _require_readable_scheme("vv", "s3://sentinel-s1-l1c/GRD/x/measurement/vv.tiff")
    message = str(excinfo.value)
    assert "s3://" in message
    assert "requester-pays" in message
    assert RTC_COLLECTION in message
    assert "not available" not in message


@pytest.mark.parametrize("href", [
    "s3://bucket/x.tiff",
    "ftp://host/x.tiff",
    "file:///etc/passwd",
    "https://host/a.tif\x00.s3",
    "https://host/a.tif\r\nHost: evil",
])
def test_non_https_and_control_characters_are_still_refused(href: str) -> None:
    with pytest.raises(InvalidInputError):
        _require_readable_scheme("vv", href)


def test_https_is_still_accepted() -> None:
    _require_readable_scheme("vv", HREF)  # must not raise


def test_signing_rate_limit_is_reported_as_itself() -> None:
    """A 429 is recoverable and is not an outage; the message must say so."""

    with pytest.raises(UpstreamServiceError) as excinfo:
        sign_rtc_asset(HREF, settings=Settings(), transport=httpx.MockTransport(
            lambda _: httpx.Response(429, json={"detail": "rate limited"}),
        ))
    message = str(excinfo.value)
    assert "rate-limiting" in message
    assert "retry" in message
    assert "Discovery and scene metadata are unaffected" in message


@pytest.mark.parametrize("status", [401, 403, 500, 503])
def test_other_signing_failures_do_not_claim_a_rate_limit(status: int) -> None:
    with pytest.raises(UpstreamServiceError) as excinfo:
        sign_rtc_asset(HREF, settings=Settings(), transport=httpx.MockTransport(
            lambda _: httpx.Response(status, json={}),
        ))
    assert "rate-limiting" not in str(excinfo.value)


# =========================================================================== #
# No SAR analysis, by construction
# =========================================================================== #


@pytest.mark.parametrize("asset", ["vv", "vh"])
def test_the_quantitative_path_refuses_uncalibrated_grd(asset: str) -> None:
    """Uncalibrated GRD cannot use the RTC quantitative power path."""

    def forbidden(_: httpx.Request) -> httpx.Response:
        pytest.fail("A SAR band read reached the network")

    with pytest.raises(InvalidInputError):
        ImageryService(transport=httpx.MockTransport(forbidden)).read_band(
            scene_id="test_rtc",
            bbox=bbox_for_window(Window(0, 0, 20, 20)),
            asset=asset,
            collection="sentinel-1-grd",
        )
