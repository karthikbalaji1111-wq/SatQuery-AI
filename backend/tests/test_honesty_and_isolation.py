"""Three standing guarantees that are easy to lose and hard to notice.

1. **SAR wording** must not claim processing the system does not perform. The
   RTC product ships `vv`, `vh`, `tilejson` and `rendered_preview` and nothing
   else - there is no layover, shadow, speckle or cloud mask to apply - so any
   sentence implying one would be a false capability claim in the user's face.

2. **Temporal scope** must be stated. Only the FIRST consecutive pair is
   analysed; a reader shown one number must not infer a multi-date trend.

3. **Deterministic isolation.** NDVI, NDWI, NDBI, temporal NDWI and the SAR
   backscatter path must run with no provider at all, so a Gemini quota or an
   NVIDIA outage cannot take the science down with it.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest
from affine import Affine
from app.services.analysis.sar import compute_sar_backscatter
from app.services.satellite.raster import BandWindow

SERVICES = pathlib.Path(__file__).resolve().parents[1] / "app" / "services"

# Phrases that assert processing this system does not do. Each would be a
# specific, checkable lie rather than mere overstatement.
FORBIDDEN = (
    "cloud-free",
    "cloud free",
    "cloud mask",
    "cloud-masked",
    "layover filtered",
    "layover-corrected",
    "shadow filtered",
    "shadow-masked",
    "speckle filtered",
    "speckle-filtered",
    "speckle reduction",
    "despeckled",
    "polarimetric decomposition",
    "radiometrically calibrated by satquery",
)


def _band(values: np.ndarray) -> BandWindow:
    return BandWindow(
        values=values,
        valid=np.ones(values.shape, dtype=bool),
        width=values.shape[1],
        height=values.shape[0],
        crs="EPSG:32644",
        transform=Affine(10.0, 0.0, 399960.0, 0.0, -10.0, 1500000.0),
        resolution=10.0,
        nodata=-32768.0,
        window={"col_off": 0, "row_off": 0, "width": values.shape[1],
                "height": values.shape[0]},
        source_shape=list(values.shape),
    )


def _sar_text() -> str:
    vv = _band(np.full((4, 4), 0.30))
    vh = _band(np.full((4, 4), 0.03))
    result = compute_sar_backscatter(
        vv=vv, vh=vh, scene_id="S1A_TEST", window_label="single"
    )
    return " ".join(result.warnings).lower()


# --------------------------------------------------------------------------- #
# 1. SAR wording
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("phrase", FORBIDDEN)
def test_sar_output_never_claims_processing_it_did_not_perform(
    phrase: str,
) -> None:
    assert phrase not in _sar_text()


def test_sar_output_does_state_what_it_actually_filtered() -> None:
    """The counter-case: silence is not honesty either.

    Refusing every phrase above would be satisfied by emitting nothing at all,
    which would leave the reader with no idea what was excluded.
    """

    text = _sar_text()

    assert "nodata" in text
    assert "quality" in text  # states that no quality mask is available
    assert "provider" in text  # attributes the terrain correction correctly


def test_sar_output_attributes_terrain_correction_to_the_provider() -> None:
    text = _sar_text()

    assert "satquery performs no" in text or "not by satquery" in text


def test_no_optical_cloud_concept_leaks_into_the_sar_result() -> None:
    vv = _band(np.full((4, 4), 0.30))
    result = compute_sar_backscatter(
        vv=vv, vh=None, scene_id="S1A_TEST", window_label="single"
    )

    names = {m.name for m in result.measurements}
    assert not any("cloud" in name for name in names)


# --------------------------------------------------------------------------- #
# 2. Temporal scope is stated, not implied
# --------------------------------------------------------------------------- #


def test_the_service_says_when_further_pairs_went_unanalysed() -> None:
    """Only `pairs[0]` is analysed; that limit must be visible in the output.

    Asserted against the source because the warning is emitted only when three
    or more observations exist, which needs a live multi-scene execution to
    reproduce - while the guarantee itself is that the sentence exists at all.
    """

    source = (SERVICES / "analysis" / "service.py").read_text()

    assert "pairs[0]" in source, "the single-pair limit moved; re-check this"
    assert "further consecutive pair(s) were not analysed" in source


# --------------------------------------------------------------------------- #
# 3. The deterministic science runs without any provider
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("package", ["analysis", "satellite", "geospatial", "query"])
def test_no_deterministic_module_imports_a_provider(package: str) -> None:
    """A Gemini quota or an NVIDIA outage must not reach the measurements.

    Enforced by reading the import graph rather than by convention: a single
    `from app.services.agent...` inside the analysis path would make the whole
    deterministic demo depend on a third-party service being up.
    """

    offenders: list[str] = []
    for path in (SERVICES / package).rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                if any(
                    token in module
                    for token in ("services.agent", "genai", "providers")
                ):
                    offenders.append(f"{path.name}: {module}")

    assert not offenders, f"{package} reaches a provider: {offenders}"


def test_the_sar_engine_computes_with_no_network_and_no_provider() -> None:
    """Executed, not merely inspected: real arithmetic, zero collaborators."""

    vv = _band(np.full((8, 8), 0.25))
    vh = _band(np.full((8, 8), 0.05))

    result = compute_sar_backscatter(
        vv=vv, vh=vh, scene_id="S1A_TEST", window_label="single"
    )
    values = {m.name: m.value for m in result.measurements}

    assert values["vv_mean_db"] == pytest.approx(10 * np.log10(0.25))
    assert values["vh_mean_db"] == pytest.approx(10 * np.log10(0.05))
