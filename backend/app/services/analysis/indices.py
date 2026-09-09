"""The spectral indices this system can compute, and the bands they need.

Every index here is a NORMALISED DIFFERENCE - ``(a - b) / (a + b)`` over two
Sentinel-2 bands. That single shape is why they can share one engine, one set
of numerical guarantees and one evidence contract: the arithmetic is identical
and only the band pair changes.

WHY RAW DIGITAL NUMBERS ARE SAFE HERE
-------------------------------------
The catalog advertises ``scale = 0.0001`` and ``offset = -0.1`` on every
spectral band, and this project deliberately does not apply them (see
``engines.STAC_SCALE_OFFSET_APPLIED`` for the measurement that settled it).
For a normalised difference that decision is exact rather than convenient: with
a COMMON multiplicative scale ``s`` and no offset,

    (a*s - b*s) / (a*s + b*s) == (a - b) / (a + b)

so the scale cancels identically. This holds only while both bands of a pair
share the same scale. Verified against the live catalog (2026-09): ``red``,
``green``, ``nir``, ``swir16`` and ``swir22`` all advertise scale 0.0001 and
offset -0.1, so every pair below cancels. A future band with a DIFFERENT scale
would silently break this, which is why the pairs are declared here rather than
assembled ad hoc at the call site.

These are INDICES, not classifications. A high NDWI is not "water", a high
NDBI is not "a building", and a high NDVI is not "healthy vegetation". The
naming and the reported wording keep that distinction.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpectralIndex:
    """One normalised-difference index and the two bands it is computed from.

    ``high_band`` and ``low_band`` are the numerator's positive and negative
    terms: the index is ``(high - low) / (high + low)``, so a high value means
    the scene is bright in ``high_band`` relative to ``low_band``.
    """

    #: Stable identifier used in the API, evidence ids and measurement names.
    key: str
    #: Display name, for prose and the interface.
    label: str
    high_band: str
    low_band: str
    #: What a HIGH value indicates - phrased as an observation about
    #: reflectance, never as a classification of what is on the ground.
    high_meaning: str
    #: The native ground sample distance the result is limited by, in metres.
    #: For a pair of 10 m bands this is 10; for anything involving a 20 m band
    #: it is 20, because the coarser band caps the real detail no matter what
    #: grid the arithmetic runs on.
    limiting_resolution_m: float


#: NDWI keeps its existing name and band order exactly - this registry describes
#: the shipped behaviour rather than redefining it.
NDWI = SpectralIndex(
    key="ndwi",
    label="NDWI",
    high_band="green",
    low_band="nir",
    high_meaning="water-like spectral response",
    limiting_resolution_m=10.0,
)

NDVI = SpectralIndex(
    key="ndvi",
    label="NDVI",
    high_band="nir",
    low_band="red",
    high_meaning="vegetation-like spectral response",
    limiting_resolution_m=10.0,
)

#: NDBI needs SWIR, which Sentinel-2 delivers at 20 m against NIR's 10 m. The
#: index is therefore reported with a 20 m limiting resolution even though the
#: arithmetic runs on the 10 m grid - see `coregister_to_finer_grid`.
NDBI = SpectralIndex(
    key="ndbi",
    label="NDBI",
    high_band="swir16",
    low_band="nir",
    high_meaning="built-up or bare spectral response",
    limiting_resolution_m=20.0,
)

#: The closed set. A request for anything else is refused rather than guessed.
SPECTRAL_INDICES: dict[str, SpectralIndex] = {
    index.key: index for index in (NDWI, NDVI, NDBI)
}

#: Default when a caller asks for analysis without naming an index. NDWI alone,
#: which is exactly what the pre-multi-index behaviour did.
DEFAULT_INDEX_KEYS: tuple[str, ...] = (NDWI.key,)


def resolve_index(key: str) -> SpectralIndex:
    """Look up an index by key, or raise a caller-facing error.

    Deliberately strict: an unrecognised index is a request the system cannot
    honour, and answering it with a different index would be worse than
    refusing.
    """

    from app.core.errors import InvalidInputError

    index = SPECTRAL_INDICES.get(key.strip().lower())
    if index is None:
        raise InvalidInputError(
            f"Unknown spectral index {key!r}. Supported: "
            f"{', '.join(sorted(SPECTRAL_INDICES))}."
        )
    return index


def bands_for(keys: tuple[str, ...]) -> tuple[str, ...]:
    """Every distinct band the given indices need, in a stable order.

    Used to read each band ONCE for a multi-index run: NIR appears in all three
    indices, so computing NDVI, NDWI and NDBI together costs four band reads
    rather than six.
    """

    seen: list[str] = []
    for key in keys:
        index = resolve_index(key)
        for band in (index.high_band, index.low_band):
            if band not in seen:
                seen.append(band)
    return tuple(seen)
