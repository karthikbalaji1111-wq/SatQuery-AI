# Supported data and scientific policy

What this system will measure, what it will measure with stated caveats, and
what it refuses to measure at all.

The distinction that governs every line below: **computational reproducibility is
not scientific validation.** Every number here is reproducible from the pixels
and cross-checked against an independent implementation. None of it has been
validated against a reference dataset, because this repository contains no
reference dataset. Where that matters, it is said.

---

## Supported

Verified against live data, with the verification named.

| Product | Source | What is computed |
|---|---|---|
| Sentinel-2 L2A optical | Earth Search (Element 84) | NDVI, NDWI, NDBI over raw DN at 10 m |
| Sentinel-1 RTC | Microsoft Planetary Computer | VV / VH gamma naught statistics in decibels |

**Spectral indices run on raw digital numbers.** The STAC metadata advertises
`scale: 0.0001, offset: -0.1` for every band, and the pixels do not behave that
way: applying the advertised offset produced NDVI above 1 — mathematically
impossible for non-negative operands — and drove "reflectance" negative for 55%
and 100% of the probed windows on two scenes from different tiles, dates and
processing baselines. A common multiplicative scale cancels identically in a
normalised difference, so the index is exact without it; an additive offset does
not cancel, and the advertised one is not applied because it does not describe
these pixels. **Absolute surface reflectance is therefore not established** — a
ratio does not need it, and nothing here reports one.

**SAR statistics are computed in linear power, then converted.** The `vv`/`vh`
assets are linear gamma-naught power (float32, `nodata -32768`, no scale or
offset in the product), so `dB = 10·log10(power)` with no amplitude squaring, and
means are `10·log10(mean(power))`. Averaging decibels instead would give a
geometric mean, biased low — seven tests fail if that mutation is applied.
Validated against an independent NumPy computation over the same window: worst
discrepancy **0.000e+00**.

---

## Conditionally supported

Real results, with a condition that travels with them.

| Capability | The condition |
|---|---|
| **NDBI** | SWIR (B11) is 20 m against NIR's 10 m. The grids are exactly 2:1 nested with a shared origin (read from the COG headers), so each 20 m value is assigned whole to the four 10 m pixels it contains. The arithmetic runs at 10 m; **the detail is 20 m**, and every NDBI result says so. A non-nested pair is refused, not resampled. |
| **Temporal NDWI (aggregate)** | Two observations indexed independently; the reported difference is between two *aggregate statistics* over two different sets of pixels. Not per-pixel change. Suppressed entirely when the framing would mislead (no valid pixels, no footprint overlap, same scene twice). |
| **Temporal NDWI (paired-pixel)** | Produced **only** when both reads land on an identical grid — same CRS, dimensions and affine, compared exactly with no tolerance. Otherwise refused with the reason, never approximated. |
| **Index thresholds** | Reported as "% of valid pixels with NDWI > 0.3" — an *index* threshold over valid pixels, never a water or flood classification. The denominator is valid pixels, not the raster's size. |
| **Scene selection** | Deterministic over the scenes the catalog **returned** — one bounded page, not the archive. `scenes_matched` reports how many matched when the catalog says, so "best of 10 examined" is distinguishable from "best of 900 matching". |
| **Cloud** | Scene cloud cover is catalog metadata for the whole tile, not a mask over the analysed area. It is reported as context and warned on above 30%. |
| **Mixed processing baselines** | Two scenes may come from different processing baselines. Nothing here reconciles them, and a comparison across them is not validated. |

---

## Unsupported

Refused outright, with the reason. Each of these is a capability the system does
**not** have, and no query, retry or configuration will produce it.

| Not supported | Why |
|---|---|
| **Sentinel-1 GRD (Earth Search)** | The measurement asset is an `s3://` URI on a requester-pays bucket; the product is in radar geometry (`crs=None`, 210 GCPs) rather than a map projection; the pixels are uncalibrated `uint16` DN whose calibration LUTs are separate XML assets. Refused at the boundary with a message naming the cause. Use the RTC collection. |
| **Cloud and shadow masking** | No `scl` mask is applied. A cloudy pixel inside the AOI is measured like any other. |
| **Co-registration / resampling across grids** | Never performed. Comparisons that would require it are refused. (NDBI's whole-cell 20 m → 10 m assignment is the only regridding anywhere, and it invents no values.) |
| **Land-cover classification** | No classifier exists. A high NDBI is a built-up-*like* reflectance signature, not a detected building. |
| **General change detection** | What exists is a difference between two dated observations of one index. Nothing classifies *what* changed. |
| **SAR calibration, speckle filtering, terrain correction, polarimetric decomposition, optical–SAR fusion** | Not implemented. The terrain correction in RTC products is the **provider's** — never describe it as SatQuery calibration. |
| **Absolute surface reflectance** | See the raw-DN decision above. |
| **Archive-wide scientific validation** | No reference dataset exists here. Reproducibility and an independent cross-check are claimed; agreement with ground truth is not. |

---

## What "verified" means in this repository

* **Cross-checked** — recomputed independently (NumPy) on the same window, with
  the discrepancy reported (0.000e+00 for SAR backscatter; NDWI byte-equal
  before and after the multi-index generalisation).
* **Mutation-checked** — the arithmetic was deliberately broken and the tests
  were observed failing, so a passing test is known to constrain something.
* **Verified live** — run against the real catalogs and real pixels, with the
  scene id and date recorded.

None of those is a claim about agreement with ground measurements.
