# SatQuery AI — Claude Code Instructions

## Project

SatQuery AI is a Smart India Hackathon 2026 system for natural-language interaction with multimodal remote-sensing imagery.

Core intended capabilities:
- Natural-language satellite queries
- Geospatial grounding
- Sentinel-2 optical imagery
- Sentinel-1 SAR imagery
- Multitemporal analysis
- Change detection
- Multimodal reasoning
- Vision-language reasoning
- Geospatial localization
- Explainable results
- Visualization

## Current Implemented Pipeline

1. Geospatial resolution
2. Sentinel-2 STAC scene discovery
3. Bounded Sentinel-2 imagery retrieval
4. Structured query intent
5. Deterministic query-plan resolution
6. Query execution orchestration (intent -> plan -> discovery -> deterministic
   scene selection -> optional bounded imagery)
7. Sentinel-1 SAR discovery + deterministic per-modality scene selection: each
   requested modality executes independently against every temporal window,
   through the existing SatelliteService (collection override).
8. Sentinel-1 VV imagery retrieval for visualization - **IMPLEMENTED BUT NOT
   REACHABLE against live Earth Search.** The ImageryService / raster path
   accepts a single-band VV asset, applies a 2nd-98th percentile display clip
   -> min-max to 8-bit grayscale -> 3 identical bands -> PNG, and returns the
   existing ImageryResponse. Display only, NOT calibrated.

   **Verified 2026-09 against the live catalog, this does not work on real
   data**, for three reasons that are properties of the source, not defects
   here: the `vv` measurement asset is published as an `s3://` URI on a
   requester-pays bucket (this deployment holds no credentials and reads only
   anonymous HTTPS); the GRD product is in radar geometry, so the COG has
   `crs=None` and 210 GCPs instead of a map projection; and the pixels are
   uncalibrated `uint16` DN amplitude (NOT Float32 backscatter - the
   calibration LUTs are separate XML assets). `_require_readable_scheme` in
   `satellite/imagery.py` refuses the request at the boundary with a message
   naming the cause, rather than letting it reach GDAL and surface as an opaque
   credentials error.

   **SUPERSEDED - see step 8b.** The paragraphs above remain accurate about
   Earth Search GRD, which is still refused. They are no longer the whole
   picture: Sentinel-1 imagery now works through a different collection.

8b. Sentinel-1 RTC imagery - **IMPLEMENTED AND VERIFIED LIVE.** Discovery,
   deterministic selection and bounded **VV and VH** retrieval run against
   Microsoft Planetary Computer's Sentinel-1 RTC collection (`satellite/rtc.py`),
   which publishes analysis-ready, **provider terrain-corrected** gamma-naught
   COGs (float32, projected, nodata -32768). Assets return
   `409 PublicAccessNotPermitted` unsigned; the public `/api/sas/v1/sign`
   endpoint signs them without credentials, so no secret is stored for it.
   Rendering is `10*log10` -> 2nd-98th percentile clip -> 8-bit grayscale,
   **display only** - a monotonic transform on the rendered PNG that never
   reaches a quantitative path. The linear stretch used previously put 75% of
   pixels in the darkest tenth of the range (median level 3/255) and was
   effectively black; the decibel stretch gives median 75/255.

8c. Quantitative Sentinel-1 backscatter - **IMPLEMENTED AND VALIDATED.**
   `analysis/sar.py` measures VV/VH gamma naught in decibels. The assets are
   linear POWER (proven from the product: strictly positive pixels, medians
   0.0272 VV / 0.0083 VH, GDAL scale 1.0 / offset 0.0, and the provider's own
   tilejson takes a log of `vv`), so `dB = 10*log10(power)` with NO amplitude
   squaring. **Means average in linear power, then convert** -
   `10*log10(mean(power))`; `mean(10*log10(power))` is a geometric mean, biased
   low, and 7 tests fail if that mutation is applied. nodata, non-finite and
   non-positive samples are excluded and counted, never clamped. Validated
   against independent NumPy on the same window: worst discrepancy 0.000e+00.
   Agent tool `sar_backscatter_statistics`; evidence namespaced
   `sar_backscatter.*` with unit `dB`, bound by grounding so a VV claim cannot
   be satisfied by VH evidence.

   STILL out of scope: radiometric calibration from raw GRD, speckle filtering,
   custom terrain correction, layover/shadow masking, polarimetric
   decomposition, SAR classification, optical-SAR fusion. The terrain
   correction is the PROVIDER'S; never describe it as SatQuery calibration.

9. Analysis boundary (contract only): `POST /api/v1/query/analyze` accepts an
   already-computed `QueryExecutionResult` and returns an `AnalysisResult`
   (status, derived task, deterministic answer, slim per-window traceability,
   warnings, empty measurements). `AnalysisService` is pure - no discovery, no
   STAC, no imagery, no raster I/O, no LLM/VLM. Only `visualize` is answered
   (`status="ok"`, templated summary); `change_detection` and
   `object_identification` return `status="not_implemented"` in a 200 body. No
   analysis engine exists yet.
10. Single-scene Sentinel-2 NDWI: a quantitative raster path
   (`BandWindow` / `read_band_window` / `ImageryService.read_band`) reads raw
   `uint16` bands at native resolution - never through the display path and
   never decimated - and a pure engine returns scalar NDWI `Measurement`s via
   the opt-in `AnalysisRequest.include_ndwi` flag. Index statistics only; not a
   validated water or flood classification.
11. Temporal observation model (domain representation only): `Observation` and
   `ObservationSet` distinguish a *requested* `TimeRange` from an *acquired*
   scene, and are derived from `ExecutedWindow`s as `QueryExecutionResult.
   observations`. Carries acquisition time, scene id, collection, modality,
   footprint, assets and any retrieved imagery so a later phase can establish
   alignment explicitly. No co-registration, no comparison, no resampling -
   observations are NOT assumed to share a CRS, grid or resolution.
12. Observation compatibility reporting (metadata only): given two
   `Observation`s, `app/services/query/compatibility.py` reports what can
   honestly be established from metadata the system already holds - and names
   what cannot. It establishes the COMPATIBILITY BOUNDARY, not an alignment
   mechanism. Pure domain capability: no API route, no frontend, no
   `AnalysisService` involvement.
13. Temporal NDWI Statistics: for ONE deterministic same-modality Sentinel-2
   pair, each observation is indexed **independently** at native 10 m
   resolution and the two summaries are reported side by side with their Phase
   13 `CompatibilityReport`. The single derived value is
   `mean_ndwi_difference = second.ndwi_mean - first.ndwi_mean` - a difference
   between two aggregate statistics over two separate sets of pixels. No pixel
   is compared against another pixel, nothing is aligned or resampled, and the
   value is suppressed when that framing would mislead. Opt-in via
   `AnalysisRequest.include_temporal_ndwi`.

14. Agentic orchestration: `POST /api/v1/query/agent` accepts a free-form
   question. A language model proposes a plan over a CLOSED three-tool
   allowlist; the server validates that plan, executes it through the same
   deterministic services the manual endpoints use, and mechanically validates
   the generated answer against the collected evidence before returning it.
   The model selects; it never computes. No image ever reaches a model, and no
   reasoning is requested, stored or displayed.
15. Provider-independent natural-language interpretation (M5.5): a question
   that names no AI provider is interpreted DETERMINISTICALLY
   (`agent/interpretation.py`) into the same `AgentPlan` an AI planner would
   propose, executed by the unchanged executor, and answered with fixed
   sentences over engine values (`agent/standard.py`) that pass the unchanged
   grounding. No model, no key. Anything it cannot map is `needs_clarification`,
   never a guess. AI interpretation is opt-in per request. See section 24.
16. Local intent model (M5.6): a small TF-IDF + logistic-regression classifier
   (`agent/intent_model.py`, artifact `agent/intent_artifacts/`) names the
   OPERATION; the rules still own places, dates and every guard. It is acted on
   only at >= its calibrated threshold (0.95 for v2; never below 0.90) and only
   when it does not contradict an operation the rules read explicitly
   (`agent/intent_router.py`); otherwise the rules decide or the user is asked.
   Local, numpy-only at runtime, no network. See sections 25 and 26 (v2).

Current HEAD represents the completed Agentic Orchestration phase, plus a
provider abstraction (Gemini + NVIDIA), a MapLibre frontend and a Direction B
UI. Test baselines quoted in the historical sections below are superseded; the
current figures are in section 29 (M6 product completion). Section 23 (the scientific core,
M1-M5) supersedes any statement below that the optical indices are not
cloud-masked, and section 24 supersedes any statement that the typed query box
or `/query/parse` needs an AI provider.

## Architecture Rules

Keep these concepts separate:

SatQueryIntent
→ Geospatial Service
→ BoundingBox
→ ResolvedQueryPlan
→ STAC discovery
→ Imagery retrieval
→ future analysis/reasoning layers

Reuse existing services, schemas and contracts whenever possible.

Do not duplicate:
- Geospatial resolution
- BoundingBox
- STAC discovery
- Imagery retrieval

Inspect existing implementations before creating new abstractions.

## Phase Discipline

Implement only the phase explicitly requested.

Do not prematurely implement future capabilities such as:
- Sentinel-1 processing
- SAR fusion
- Change detection
- Temporal analysis
- VLM
- LLM/NLP parsing
- AI inference
- MapLibre
- spectral indices

unless the current task explicitly authorizes them.

Do not modify working functionality unnecessarily.

## Development Rules

- Inspect before editing.
- Prefer small, focused changes.
- Reuse existing architecture.
- Avoid unnecessary dependencies.
- Never add credentials, secrets or API keys.
- Never download large datasets or imagery unless explicitly requested.
- Maintain backward compatibility with existing functionality.
- Keep backend and frontend contracts synchronized.
- Use typed Pydantic models on the backend.
- Use TypeScript types on the frontend.

## Testing

Before and after significant changes run the project's existing checks.

Backend:
- pytest
- ruff check .

Frontend:
- npm run lint
- npm run typecheck
- npm run test
- npm run build

Also run:

git diff --check

Do not claim completion if checks fail.

## Git

Each implementation phase should have its own focused commit.

Before committing:
- inspect git diff
- run tests/checks
- run git diff --check
- verify no unrelated files changed

Never force-push or rewrite history unless explicitly instructed.

## Important

The repository's existing implementation is authoritative.

Do not assume a file, service, schema, dependency or API exists.

Inspect the repository first and adapt to the actual codebase.

---

# CURRENT PROJECT STATE / NEXT SESSION CHECKPOINT

> Written at the end of the Phase 10 session, after a completed Phase 11
> architecture preflight and a live read-only Sentinel-2 band verification.
> **The next session should START PHASE 11 IMPLEMENTATION from this checkpoint
> and must NOT repeat the architecture investigation or the band verification.**
>
> Everything under **VERIFIED** was directly observed in this repository or from
> live read-only STAC/COG requests. Everything under **PLANNED** is design only
> and is **not implemented**.

## 1. Repository state — VERIFIED

- Branch: `main`
- HEAD: `36eb5cc` — `feat(analysis): add temporal NDWI statistics`
  (this section was written during Phase 11; the lines below describing
  `6036768` as HEAD and Phase 11 work as uncommitted are HISTORICAL and
  no longer true — see sections 13 and 14 for the current state.)
  (Phase 10 baseline was `43f06ee`)
- `main` is in sync with `origin/main`; Phase 10 and the Phase 11 backend are
  **pushed**. The Phase 11 frontend integration and raster integration tests are
  **uncommitted working-tree changes** at the time of writing.

## 2. Phase status — VERIFIED

- **Phases 1–9: complete.** Geospatial grounding → S2 STAC discovery → bounded S2
  imagery → structured intent → query-plan resolution → query execution
  orchestration → S1 discovery + per-modality deterministic selection → S1 VV
  display imagery → S1 collection-aware imagery lookup fix.
- **Phase 10: complete and pushed.** Analysis boundary, contract only.
- **Phase 11: IMPLEMENTED.** Quantitative Sentinel-2 raster access
  (`BandWindow` / `read_band_window` / `ImageryService.read_band`) plus one pure
  single-scene NDWI engine returning scalar `Measurement`s, dispatched from
  `AnalysisService` via the opt-in `AnalysisRequest.include_ndwi` flag, and
  surfaced in the UI by an NDWI checkbox in `QueryPanel`. No other analysis
  engine exists.
- **Phase 12: IMPLEMENTED.** Temporal observation model - `Observation` /
  `ObservationSet` in `query/schemas.py`, exposed additively as the derived
  `QueryExecutionResult.observations`. Domain representation only: no temporal
  analysis, no differencing, no co-registration, no resampling, and no
  assumption that observations are spatially aligned or share a resolution.
- **Phase 13: IMPLEMENTED.** Observation compatibility reporting -
  `CompatibilityReport` / `ObservationPair` / `PairingFailure` /
  `compute_compatibility` / `pair_observations` in
  `query/compatibility.py`. Metadata only: no raster I/O, no resampling, no
  pixel comparison, and co-registration is never claimed. Pure domain
  capability - no route, no frontend, no `AnalysisService` change. See
  section 13.
- **Phase 14: IMPLEMENTED.** Temporal NDWI Statistics - one deterministic
  Sentinel-2 pair, each observation indexed independently, reported side by
  side with its Phase 13 compatibility report plus a single aggregate
  `mean_ndwi_difference`. Opt-in (`include_temporal_ndwi`), additive on the
  existing `/query/analyze` contract. No co-registration, no resampling, no
  pixel comparison. See section 14.
- **Phase 15: IMPLEMENTED.** Agentic orchestration - `services/agent/`
  (contracts, registry, executor, grounding, planner, synthesizer, service,
  `providers/gemini.py`), one additive endpoint `POST /api/v1/query/agent`, and
  a React `AgentPanel`. Additive throughout: the manual `/query/execute` and
  `/query/analyze` paths are unchanged. **Verified against fake provider
  clients only - see the Known Limitations in section 15.**

## 3. Architectural state — VERIFIED

```
NL query -> POST /api/v1/query/parse      -> AiService/GeminiIntentParser -> SatQueryIntent
         -> POST /api/v1/query/build-plan -> QueryService -> GeospatialService -> ResolvedQueryPlan
         -> POST /api/v1/query/execute    -> QueryExecutionService
                                              -> SatelliteService.search (per modality x window)
                                              -> deterministic scene selection
                                              -> ImageryService.retrieve (optional, display PNG)
                                           -> QueryExecutionResult
         -> POST /api/v1/query/analyze    -> AnalysisService -> AnalysisResult
```

- `AnalysisService` is currently **pure**: zero collaborators, no network, no
  model, no raster I/O. `visualize` returns `status="ok"` with a deterministic
  templated summary; `change_detection` and `object_identification` return
  `status="not_implemented"` in a **200** body (deliberate — a 501 error body
  cannot carry `windows_considered`/`warnings`).
- Dependency direction is `analysis -> query -> satellite` and is **acyclic**;
  `services/query` must never import `services/analysis`.
- `MultimodalService`, `TemporalService`, `MapService` remain **unused stubs**
  and are the reserved future homes for fusion / change detection / map tiles.
- `MapPanel` renders real MapLibre GL basemaps and footprints (maplibre-gl
  is a frontend dependency). The line that previously claimed it was a
  placeholder with no MapLibre dependency was stale and is corrected here.

## 4. Test / regression baseline — VERIFIED

Any Phase 11 work must keep these green and must not reduce them:

| Check | Baseline at `43f06ee` (HISTORICAL — current figures in section 14.1) |
| --- | --- |
| `pytest -q` (backend) | **240 passed**, 1 pre-existing StarletteDeprecationWarning |
| `ruff check .` | All checks passed |
| `npm run test` | **44 passed**, 3 test files |
| `npm run lint` / `typecheck` / `build` | clean / clean / builds |
| `git diff --check` | clean |

Test conventions: hand-written recording fakes (no `unittest.mock`),
`app.dependency_overrides` for routes, `asyncio.run` for async services,
`httpx.MockTransport` / synthetic in-memory GeoTIFFs for I/O. No test may
contact Gemini, Nominatim, STAC, or real imagery.

## 5. Phase 11 preflight conclusion — VERIFIED analysis, PLANNED direction

Six directions were compared (analysis foundation / basic optical / SAR
analysis / optical+SAR fusion / multitemporal change detection / object
detection). **Every direction except the analysis foundation is blocked by
infrastructure that does not exist.** Three blockers, all confirmed in-repo:

1. **No quantitative pixel path.** Both branches of `raster._extract_window`
   terminate in 8-bit display bytes: `band_count >= 3` requires `uint8` and
   returns a TCI rendering; `band_count == 1` runs `_normalize_sar_band`
   (2nd–98th percentile clip -> uint8), destroying physical values by design.
2. **No spectral band was reachable.** `SUPPORTED_IMAGERY_ASSETS = ("visual","vv")`.
3. **No pixel -> geographic transform.** `ImageryResponse` carries the source
   CRS, native resolution and the source-relative pixel `window`, but **not the
   source affine origin**, so returned pixels cannot be georeferenced.

**Selected Phase 11 direction (PLANNED):** *Analysis Foundation / quantitative
geospatial raster access*, proven end-to-end by **one honest single-scene
Sentinel-2 NDWI engine reporting scalar statistics**. A single-scene 10 m index
is the only credible real analysis that needs **no co-registration, no
calibration, and no ML model**.

## 6. Verified Sentinel-2 L2A asset mapping — VERIFIED (live STAC)

Earth Search v1 uses **common names, not `B03`/`B08`**. Verified against
`S2B_44PLV_20241026_0_L2A` (tile 44PLV, 2024-10-26, baseline 05.11, EPSG:32644)
and `S2A_43QBA_20230529_0_L2A` (tile 43QBA, 2023-05-29, baseline 05.09,
EPSG:32643).

| Asset key | Band | GSD | dtype | nodata | Phase 11 status |
| --- | --- | --- | --- | --- | --- |
| `green` | B03 | 10 m | `uint16` | `0` | **use** (NDWI) |
| `nir` | B08 | 10 m | `uint16` | `0` | **use** (NDWI) |
| `red` | B04 | 10 m | `uint16` | `0` | available (NDVI) |
| `swir16` | B11 | 20 m | `uint16` | `0` | **deferred** — different grid, needs resampling |
| `scl` | SCL | 20 m | `uint8` | `0` | deferred in Phase 11; **used since M3** as the pixel-quality mask (section 23) |
| `visual` | TCI | 10 m | `uint8` RGB | — | display only (existing behaviour) |

COG characteristics (read from actual headers): 10 m bands are
`10980 x 10980`, transform `(10, 0, 300000, 0, -10, 1500000)`, overviews
`[2,4,8,16]`, block shape `1024 x 1024`; 20 m bands are `5490 x 5490` with
transform `(20, 0, 300000, 0, -20, 1500000)`. In-file GDAL scale/offset are
`(1.0,) / (0.0,)` — i.e. **not set in the file**. Media type for all COGs is
`image/tiff; application=geotiff; profile=cloud-optimized`; the parallel
`-jp2` assets are `image/jp2` and are already correctly rejected by
`ImageryService._resolve_asset_href`.

**Band alignment — VERIFIED.** For one AOI, `green`, `nir` and `red` all resolve
to the *identical* window `(5568, 5271, 437, 446)` with identical CRS and
transform: **pixel-for-pixel aligned, zero resampling required**. `swir16` and
`scl` share their own exact 2:1 20 m grid `(2784, 2635, 219, 224)`.

**Existing code already reaches these bands — VERIFIED.**
`ImageryService._default_fetch_item(scene_id, "sentinel-2-l2a")` plus
`_resolve_asset_href(item, "green"|"nir"|"red"|"swir16"|"scl")` all succeed.
`Scene.assets` and `_USEFUL_ASSET_KEYS` therefore **do not need to change** —
asset resolution happens against the live STAC item.

## 7. CRITICAL scale/offset finding — VERIFIED

STAC `raster:bands` advertises `scale: 0.0001, offset: -0.1` for every spectral
band. **The pixel data does not behave that way.**

Measured over `SCL == 4` (vegetation, 8500 px) on `S2B_44PLV_20241026_0_L2A`:

| Computation | NDVI (vegetation) | NDVI range | NDWI | out of [-1,1] |
| --- | --- | --- | --- | --- |
| **raw DN** | **+0.637** (correct) | +0.25 .. +0.83 | -0.579 | 0 px |
| advertised scale+offset | **+1.300** (impossible) | -36.52 .. +6.50 | -1.139 | 7168 px |

Expected for healthy vegetation is NDVI ~ +0.3..+0.8. NDVI above 1 is
mathematically impossible for non-negative operands, which proves the offset
over-subtracts. Applying it drove green "reflectance" negative for **55 %** of
the probed window on scene 1 and **100 %** on scene 2 (different tile, date and
processing baseline) — so the finding **generalises across both probed scenes**.

**Decision for Phase 11:**

- Compute NDWI on **raw DN**. Do **not** apply the advertised `-0.1` offset.
- This is exact, not a shortcut: for a common multiplicative scale `s` and no
  offset, `(g*s - n*s) / (g*s + n*s) == (g - n) / (g + n)` — **the scale cancels
  identically in a normalized difference**.
- **Absolute surface reflectance is NOT established** by this work. Any future
  need for absolute reflectance (rather than a ratio) must resolve the
  scale/offset question first.
- Keep this decision **isolated in one documented constant/helper** so it can be
  revisited if Element 84 regenerates the collection. Spot-check one further
  scene from a different UTM zone before hardcoding.

## 8. Confirmed architectural traps — VERIFIED, do not repeat

1. **Never call `read_rgb_window()` for quantitative spectral analysis.** A
   single-band `uint16` spectral band has `count == 1`, so it is routed into the
   Sentinel-1 branch and percentile-normalised to `uint8`. Values are destroyed
   **silently — no exception is raised.**
2. **Never compute NDWI (or any index) from display-normalized PNGs.** Both the
   S2 `visual` TCI and the S1 VV PNG are display renderings, not physical values.
3. **Never treat `ImageryResponse.bbox` as the actual pixel coverage.** It echoes
   the *request*; `_clamp_window_to_source` floor/ceils after reprojection, so
   real coverage is larger. The offset is **not a fixed constant**: it depends
   on where the AOI falls on the source grid, and it is compounded by a scale
   error, because assuming the array spans the bbox mis-sizes every pixel.
   Measured on the live Marina Beach AOI against `S2B_44PMV_20250104_0_L2A`:
   origin off by 3.944 m / 0.394 px in x and -3.753 m / -0.375 px in y, plus a
   pixel size of 9.9578 x 9.9757 m instead of 10 x 10. It is worse and
   asymmetric when a window clamps at a scene edge. Georeferencing a detection
   through it is silently wrong - use the window affine (Phase 16) instead.
4. **Never perform change detection before co-registration.** `compare` windows
   are independently reprojected, independently clamped and independently
   decimated; differencing them would paint a false-change border around every
   image. Change detection is invalid until an explicit resampling step exists.

## 9. Phase 11 implementation plan — IMPLEMENTED

Shipped as planned below. Two decisions were settled during implementation:
dispatch is the opt-in `AnalysisRequest.include_ndwi` flag (`QueryTask`,
`SatQueryIntent` and `QueryExecutionResult` all untouched), and the STAC
collection is recovered from the selected `Scene.collection`, falling back to
`None` so `ImageryService` uses its configured default. `max_dimension` and
`max_window_pixels` are enforced as **rejection** bounds, because a quantitative
read is never decimated.

Dependency order as built:

1. `satellite/schemas.py` — add an **analysis-only** band allowlist (e.g.
   `ANALYSIS_BAND_ASSETS = ("green", "nir", "red")`). The public display
   whitelist `SUPPORTED_IMAGERY_ASSETS` stays `("visual", "vv")`.
2. `satellite/raster.py` — **new** `BandWindow` + `read_band_window(...)`:
   preserves values as float, returns transform, CRS, output GSD and a validity
   mask from `nodata`. Reuses the existing window math
   (`transform_bounds` -> `from_bounds` -> `_clamp_window_to_source`) and both
   existing caps (`imagery_max_dimension`, `imagery_max_window_pixels`).
   **`read_rgb_window` and `_normalize_sar_band` must not be modified.**
3. `satellite/imagery.py` — **new** `read_band(...)` reusing
   `_default_fetch_item` + `_resolve_asset_href` + the new raster function.
   `retrieve` unchanged. `ImageryService` remains the sole imagery entry point.
4. `analysis/engines.py` — **pure** NDWI on raw DN, masking `nodata == 0` and
   guarding the denominator (`green + nir == 0`), returning `Measurement`s.
5. `analysis/service.py` — inject `ImageryService` as a keyword argument
   defaulting to a real instance (**zero-argument construction must still work**
   — `tests/test_services.py` requires it). `AnalysisService` stays the
   **dispatcher**; it must not perform pixel arithmetic itself. Its Phase 10
   "no imagery, no raster I/O" docstring must be **explicitly amended**.
6. Tests: synthetic-raster tests for `read_band_window` (float preservation,
   nodata, transform, decimated GSD); engine unit tests on known arrays;
   `AnalysisService` tests with a fake imagery service; full Phase 1–10
   regression.
7. `CLAUDE.md` — pipeline step 10.

**Pixel access mechanism:** pixels stay **server-side**. The engine re-retrieves
through `ImageryService` keyed by `selected_scene_id` + `collection` + asset +
`plan.bbox`. Re-retrieval is correct here — COG windowed reads are cheap and
idempotent, and an NDWI needs only two reads. (Three since M3: `scl` is read
first for pixel quality - section 23.)

**Output:** **scalar `Measurement`s only** (e.g. mean/min/max NDWI, valid-pixel
count, % of valid pixels above a stated index threshold).

**Scoping note:** because Phase 11 reports only scalars, it does **not** need the
`ImageryResponse` georeferencing fix (actual extent / affine / GSD). That fix
becomes mandatory the moment a mask, overlay or detection **location** is
emitted — sequence it immediately before that work, not now.

## 10. Phase 11 MUST NOT introduce

- change detection
- optical/SAR fusion
- co-registration or resampling (including 10 m + 20 m index combinations)
- SAR calibration, speckle filtering, terrain correction
- object detection or localization
- any ML/VLM runtime or model weights (torch, onnx, ultralytics, SAM, ...)
- raw-array transport across the API boundary
- persistence, caching, execution IDs, job queues
- overlays, masks-as-output, GeoJSON or any geometry model
- MapLibre / map work
- VH polarization

## 11. Frontend — VERIFIED

`AnalysisResult.measurements` was already mirrored in
`frontend/src/api/types.ts` and already rendered by `AnalysisView` in
`QueryPanel.tsx`, so the measurement rendering needed no change and was reused
as-is. The Phase 11 backend landed backend-only; a small follow-up made NDWI
reachable from the UI:

- `types.ts` — `AnalysisRequest.include_ndwi?: boolean`.
- `QueryPanel.tsx` — an `includeNdwi` checkbox beside the imagery checkbox. The
  flag is **omitted from the request body when off**, so a non-NDWI analysis
  request stays byte-identical to the pre-NDWI behaviour.
- `query.ts` — documented `include_ndwi`; corrected a stale comment that still
  claimed Sentinel-1 was skipped rather than executed.

## 12. Reporting honesty requirement — IMPLEMENTED (standing rule)

Phase 11 must report **NDWI scalar statistics**, not "water detection" or "flood
detection". A threshold, if reported at all, must be labelled explicitly as an
*index threshold* (e.g. "% of valid pixels with NDWI > 0.3"), never as a
validated water or flood classification. `scl` can later provide an independent
validation reference, but that is deferred. Do not claim an analysis the system
did not perform. (Since M3 `scl` IS used - as a per-pixel QUALITY mask, never as
a water reference or a validation of any index; see section 23.)

## 13. Phase 13 — Observation Compatibility Reporting — IMPLEMENTED

Metadata-only. Establishes the **compatibility boundary**, not an alignment
mechanism. Raster-level co-registration and resampling remain deferred to a
later phase.

**Location:** `backend/app/services/query/compatibility.py` (pure domain module,
sibling of the `Observation`/`ObservationSet` models it reports over). Owned by
the query domain deliberately: compatibility is a relationship between
observations, independent of any particular analysis, so future analysis modules
**consume** this layer rather than own it. Dependency direction `analysis ->
query` is unchanged; this module imports nothing from `services/analysis`.

**Public API** (re-exported from `app.services.query`):

```python
MatchStatus          = Literal["same", "different", "unknown"]
BboxOverlapStatus    = Literal["none", "partial", "full", "unknown"]
CoRegistrationStatus = Literal["not_evaluated", "not_supported_cross_modal"]

class CompatibilityReport(BaseModel)   # 8 fields, listed below
class ObservationPair(BaseModel)       # first, second
class PairingFailure(BaseModel)        # modality: Modality | None, reason: str

def compute_compatibility(first: Observation, second: Observation) -> CompatibilityReport
def pair_observations(observations: ObservationSet)
    -> tuple[list[ObservationPair], list[PairingFailure]]
```

`CompatibilityReport` fields: `same_modality`, `temporal_separation_days`,
`bbox_overlap`, `crs_match`, `resolution_match`, `processing_level_match`,
`limitations`, `co_registration_status`. No intersection geometry is emitted.

**Co-registration is NEVER claimed.** `co_registration_status` is assigned
*structurally* from modality alone, before any match is evaluated, and no later
code path can upgrade it. There is deliberately no value meaning
"co-registered" — this layer cannot establish that, so it cannot report it.

    same modality != co-registration      same CRS        != co-registration
    same bbox     != co-registration      same resolution != co-registration

- same modality -> `co_registration_status = "not_evaluated"`
- S1 + S2       -> `co_registration_status = "not_supported_cross_modal"`

**Unknown must never become "different".** Every `MatchStatus` helper returns
`"unknown"` on a missing input before reaching its equality branch.

**What the metadata actually supports — VERIFIED.** `_normalize_scene`
(`satellite/service.py`) keeps only `datetime`, `bbox`, `geometry`,
`eo:cloud_cover`, `collection`, `platform` and `processing:level`; the STAC
item's `proj:epsg` and `gsd` are **dropped**. The only in-repo source of a CRS
or resolution is `ImageryResponse`, which exists solely when bounded imagery was
retrieved. Therefore **with `include_imagery=False` — the common case —
`crs_match` and `resolution_match` are always `"unknown"`.** That is the correct
report, not a gap to work around. Separately, `processing_level` may be *derived
from the collection name* rather than read from the item, so a `"same"` verdict
says nothing about the processing baseline; this is stated in `limitations`
whenever the field is not unknown.

**`bbox_overlap`** is a coarse WGS84 relation between whole-scene footprints:
`full` = one contains the other (identical included), `partial` = positive-area
intersection with neither containing, `none` = no positive-area intersection (a
shared edge has zero area and is `none`), `unknown` = a footprint is absent. No
area and no percentage — degrees are not an equal-area unit.

**Pairing — deterministic, same-modality, consecutive.** Within each modality,
observations are ordered by **acquisition time** ascending (unknown times last,
ties broken by window label then scene id) and paired consecutively, so *n*
observations yield *n - 1* pairs. Modalities are visited in order of first
appearance. Fewer than two observations in a modality yields a `PairingFailure`;
an empty set yields one failure with `modality=None`.

`first`/`second` mean *acquired earlier* / *acquired later*. They do **not** mean
baseline/target: `TemporalComparison` does not require the baseline window to
precede the target, so an inverted comparison pairs as `first=target`. Requested
roles remain readable as `Observation.window_label`.

**Cross-modal pairs are NOT produced by `pair_observations()`** — exclusion is
structural (pairing only ever zips within one modality group), not a filter.
Proposing an S1/S2 pair would assert that comparing them is coherent, and it is
not without SAR terrain correction, which is out of scope. **Cross-modal
compatibility is still directly reportable** via
`compute_compatibility(s2_obs, s1_obs)`, which returns
`not_supported_cross_modal` — an explicit refusal is information, not a
proposal. Pairing *proposes*; `compute_compatibility` *qualifies*.

**Phase 13 does NOT contain:** raster I/O · rasterio imports · WarpedVRT ·
reprojection · resampling · `GridSpec` · aligned raster reads · pixel overlap ·
pixel differences · co-valid pixel counts · change detection · API endpoint or
route change · frontend change · `AnalysisService` change · `ImageryService`
change · any change to the raster reading path · `include_alignment`.
Enforced by tests that parse the module's imports with `ast` and assert them a
subset of `{__future__, math, datetime, typing, pydantic, app}`.

**Files:** `app/services/query/compatibility.py` (new),
`tests/test_compatibility.py` (new, 64 tests),
`app/services/query/__init__.py` (exports only — no behaviour change).

**Baseline after Phase 13:** `pytest -q` **376 passed** (was 312), `ruff check .`
clean, `git diff --check` clean; frontend untouched (**50 passed**, 3 files).
(Superseded by Phase 14 and 14.1 — see section 14.1 for the current figures.)
Note the pre-Phase-13 checkpoint above records a stale backend baseline of 240;
the real figure at `185320a` was 312.

### Known issue — NOT fixed in Phase 13

`ObservationSet.ordered_by_acquisition()` (`query/schemas.py`) raises
`TypeError: can't compare offset-naive and offset-aware datetimes` when one
scene's `datetime` carries a `Z` suffix and another does not. Reproduced live.
It is dormant because Earth Search is consistently `Z`-suffixed. This is
**Phase 12 code and was deliberately left untouched**; it remains a separate
known issue. `pair_observations()` is unaffected — it uses its own guarded sort
key that anchors a naive datetime to UTC **for ordering only**. Note the
deliberate asymmetry: `temporal_separation_days` refuses that assumption and
returns `None` for a mixed pair, because a *reported measurement* must not
invent a time zone.

## 14. Phase 14 — Temporal NDWI Statistics — IMPLEMENTED

The first capability in SatQuery that **reasons across two observations**. It
does so without any raster alignment, because it never needs one: each
observation is indexed on its own pixels and only the resulting *scalars* are
placed side by side.

**Objective.** For ONE deterministic same-modality Sentinel-2 pair, compute NDWI
statistics independently per observation at native 10 m resolution, and report
them together with the pair's Phase 13 `CompatibilityReport`.

**The one derived value:**

```
mean_ndwi_difference = second.ndwi_mean - first.ndwi_mean
```

It is a difference between two **aggregate statistics**, each summarising a
different set of pixels. It is NOT per-pixel change detection, NOT spatial
change detection, NOT detected physical change, NOT a change mask, and NOT
evidence of land-cover change. User-facing terminology is **"Temporal NDWI
Statistics"** and **"Mean NDWI Difference"**. A test scans every user-facing
string against a forbidden-phrase list; the disclaimers are worded so they never
need those phrases themselves.

**Suppression.** The difference is withheld entirely - not annotated - when the
framing would mislead: either observation has no valid pixels, the footprints do
not overlap (`bbox_overlap == "none"`), or both observations resolved to the same
scene. A number a reader can see is a number a reader will use.

**Pair selection.** Optical observations only (SAR is filtered out *before*
pairing - NDWI is optical-only and S1 is not comparable without terrain
correction). Then Phase 13's `pair_observations`, read-only:

| Eligible optical observations | Behaviour |
| --- | --- |
| 0 or 1 | Warning, `temporal_comparison = None`, **zero band reads** |
| exactly 2 | Those two are compared |
| 3+ | `pairs[0]` only, with a warning naming the unanalysed pairs |

`first`/`second` mean *acquired earlier* / *acquired later*, never
baseline/target - roles stay readable via `window_label`.

**Contract** (additive on the existing `/query/analyze`; **no new endpoint**):

```python
AnalysisRequest.include_temporal_ndwi: bool = False        # opt-in
AnalysisResult.temporal_comparison: TemporalIndexComparison | None = None

class ObservationIndexResult:   # window_label, scene_id, acquired_at,
                                # cloud_cover, measurements
class TemporalIndexComparison:  # first, second, compatibility,
                                # differences, warnings
```

`Measurement` and `CompatibilityReport` are reused verbatim.

**Warning split.** `TemporalIndexComparison.warnings` carries the comparison's
own qualifications (aggregate framing, suppression reason, partial overlap,
cloud, tiny sample). `AnalysisResult.warnings` carries orchestration outcomes
(no pair formed, a band read failed, further pairs not analysed). No duplication.

**Cloud.** `Scene.cloud_cover` is reported as context and warned on above 30%.
HISTORICAL: in Phase 14 the index was never cloud-masked. **Since M3 it is**:
each observation is masked by its own Sentinel-2 SCL before any statistic, and
the paired change uses only pixels usable on both dates (section 23). Unknown
scene-level cloud cover is still reported as unknown, never assumed clear.

**Reads.** Exactly four per comparison in Phase 14: `green` + `nir` per
observation, through the **unchanged** `ImageryService.read_band`. **Six since
M3**: `scl`, `green`, `nir` per observation (section 23). Deliberately not optimised - no
batching, no caching, no multi-band redesign. `raster.py` and `imagery.py` are
untouched.

**Architecture.** `AnalysisService` orchestrates only: select the pair (Phase 13,
read-only), read the bands (Phase 11, unchanged), delegate every arithmetic and
suppression decision to `compare_ndwi_observations` in `engines.py`. A test
asserts the service source contains no `numpy` and no `np.`.

**Phase 14 does NOT contain:** co-registration · grid alignment · resampling ·
`WarpedVRT` · `rasterio.warp` · per-pixel differencing · change masks · change
detection · cloud masking · NDVI or any second index · SWIR/20 m bands · SAR
analysis · cross-modal comparison · a general time-series framework · caching or
persistence · georeferencing changes · overlays/GeoJSON · MapLibre · ML/VLM ·
new dependencies · new endpoints. `raster.py`, `imagery.py`, `query/schemas.py`,
`query/execution.py` and `query/compatibility.py` are unmodified.

**Backward compatibility.** `include_temporal_ndwi` defaults `False`. When false:
existing response fields retain identical values and semantics, **no temporal
band read is performed**, and `temporal_comparison` is `null`. The only
intentional serialized difference is that new optional field with value `null`
(Option A - deliberately no custom `model_serializer`).

## 14.1 Audit remediation — IMPLEMENTED

An independent audit of the Phase 14 checkpoint (`36eb5cc`) raised ten findings.
Each was verified against the actual code before any change; three were
confirmed and fixed, three were presentation/documentation corrections, and
four were rejected or deliberately deferred with the reason recorded.

### Confirmed and fixed

**AOI coverage was not established (audit finding A) — CONFIRMED.**
`bbox_overlap` compares the two *scene footprints* to each other. It says
nothing about how much of the *requested AOI* each observation actually
analysed: each quantitative read is clamped to its own scene by
`_clamp_window_to_source` and masked by its own nodata, so two observations
over the same bbox can analyse wildly different pixel counts while reporting
`bbox_overlap == "full"`. `ObservationIndexResult` now carries
`window_pixel_count` (the clamped AOI window, width x height) alongside the
existing `ndwi_valid_pixel_count`, and the pure engine emits an explicit
statement of both observations' coverage.

**No threshold was invented.** The repository has no scientifically defensible
basis for a "materially different coverage" cut-off, so the correction states
the measured coverage and stops. The difference is NOT suppressed on a coverage
ratio — replacing an honest report with a fabricated judgement would be worse
than the gap it closes. The existing suppression rules (no valid pixels, no
footprint overlap, same scene) are unchanged.

**BandWindow evidence was discarded (audit finding, section 3) — CONFIRMED.**
`_observation_index` read `crs`, `resolution`, `width` and `height` from the
`BandWindow` and kept only the measurements. `ObservationIndexResult` now
carries `crs` and `resolution` from the actual read, and the engine reports
whether the two reads used the same grid.

`query/compatibility.py` was NOT modified. Phase 13 remains metadata-only and
will still report `crs_match: "unknown"` when bounded display imagery was not
retrieved. That is not a contradiction: the compatibility report and the
observation results are two different evidence sources — STAC metadata versus
the raster read — and Phase 14.1 keeps them clearly separated rather than
feeding raster metadata into the metadata-only layer.

**/query/analyze trust boundary (audit finding C) — CONFIRMED, with corrected
terminology.** A client can submit a fabricated `QueryExecutionResult`; its
`selected_scene_id` and `Scene.collection` reach `ImageryService.read_band` and
are interpolated into the STAC item URL. Verified empirically with `httpx`:

| Class | Reachable? | Evidence |
| --- | --- | --- |
| Arbitrary-host SSRF | **NO** | the host is always `settings.stac_base_url`; no input reaches it |
| Fixed-host path manipulation | **YES** | `collection="../../../search"` rewrote the path to `/search/items/x`; `?`/`#` split off a query/fragment |
| Remote-read resource abuse | **YES** | unbounded, arbitrary identifiers drove outbound requests |

Hardened at the URL-building boundary in `satellite/imagery.py`:
`_validate_stac_identifier` allows `[A-Za-z0-9._-]{1,200}` and refuses the
reserved segments `.` and `..`, applied to `scene_id` and the resolved
`collection` in both `retrieve` and `read_band`, **before** any catalog call.
Real Earth Search identifiers pass unchanged. No authentication was added and
no endpoint changed. An empty `collection` still falls back to the configured
default — a server-controlled value, so it is accepted, not rejected.

### Presentation corrections

- **Number formatting** is now applied at the presentation layer only. The API
  representation and backend values are unrounded and unchanged. Unit-driven:
  `index` -> 4 dp, `%` -> 1 dp, `pixels` -> integer with separators; day
  intervals -> 1 dp; cloud cover -> 1 dp, matching the existing scene list.
- **Unsupported tasks are labelled.** `change_detection` and
  `object_identification` remain selectable (the backend answers them with
  `status: "not_implemented"`), but read "Change Detection (unavailable)" and
  "Object Identification (unavailable)" so Temporal NDWI Statistics can never be
  mistaken for a change-detection result. No change detection was implemented,
  nothing was renamed, and no fake result was added.
- **`ObservationSet.for_modality`** is now reused by `_temporal_ndwi` instead of
  an inline comprehension. Identical behaviour; the domain model already
  exposed exactly this query.

### Rejected or deferred, with reasons

- **Sample-size warning threshold — NOT CHANGED.** `_sample_warnings` fires at
  `0 < count <= 1`. It communicates a degenerate sample (min == max == mean),
  which is exact and needs no constant. Any stronger threshold ("at least N
  pixels for a meaningful mean") would depend on spatial autocorrelation and
  the effective sample size, neither of which this system estimates. Inventing
  one would be an arbitrary scientific constant. **KNOWN LIMITATION:** a small
  but >1-pixel sample is reported without a statistical-power caveat.
- **Performance — NOT CHANGED.** Sequential discovery, four sequential band
  reads per comparison, and no caching are all real. They are correctness-
  neutral and were explicitly out of scope. **FUTURE ENGINEERING WORK.**
- **Mixed aware/naive datetime issue — NOT FIXED**, as documented in section 13.
  Still dormant (Earth Search is consistently `Z`-suffixed) and still confined
  to `ObservationSet.ordered_by_acquisition`; `pair_observations` and
  `compute_compatibility` remain unaffected.

### Baseline after Phase 14.1 — VERIFIED

| Check | Result |
| --- | --- |
| `pytest -q` | **458 passed** (426 at `36eb5cc`) |
| `ruff check .` | clean |
| `git diff --check` | clean |
| `npm run lint` / `typecheck` / `test` / `build` | clean / clean / **58 passed** / builds |

Two existing frontend assertions were updated — not weakened — to the
deliberately changed presentation: the task-option label, and a pixel count now
rendered with thousands separators. No backend test was weakened or removed.

## 15. Phase 15 — Agentic Orchestration — IMPLEMENTED

A language model chooses **which** of the existing deterministic analyses to
run. It does not compute anything, and it is never authoritative: every choice
it makes and every sentence it writes is validated by the server before it has
any effect.

**Architecture — strictly acyclic:**

```
api -> agent -> {analysis, query} -> satellite
                agent/providers/gemini -> google-genai
```

Nothing in `analysis`, `query`, `satellite`, `geospatial` or `core` may import
`services.agent`; the API layer imports it and is the composition point. Agent
core never imports `providers/`. Both directions are enforced by tests.

**Separation of responsibilities** — each boundary owns exactly one thing:

| Component | Responsibility |
| --- | --- |
| `AgentPlanner` (ABC) | proposes a validated `AgentPlan`. Never executes |
| `AgentExecutor` | deterministic execution through the existing services |
| `AnswerSynthesizer` (ABC) | turns evidence into a `DraftAnswer`. Computes nothing |
| `grounding.validate_answer` | pure mechanical checks over draft + evidence |
| `AgentService` | orchestrates the four; owns none of their logic |
| `api/routes/query.py` | HTTP adapter only - one handler, one call |

**SDK isolation.** The GenAI SDK is confined to `agent/providers/gemini.py`.
`schemas.py`, `registry.py`, `executor.py`, `grounding.py`, `planner.py`,
`synthesizer.py` and `service.py` import no SDK - asserted by AST tests that
enumerate the package and require the importer list to be exactly
`['providers/gemini.py']`.

**The tool allowlist is closed.** `execute_query`, `ndwi_statistics`,
`temporal_ndwi_statistics` - and nothing else. `ToolCall` is a Pydantic
discriminated union, so an unrecognised name fails validation before dispatch;
the registry holds inert descriptors with no callables. `retrieve_imagery` and
`compatibility_report` are deliberately absent (imagery is a parameter and
compatibility is an automatic byproduct). `limit` is NOT model-controlled: it
is a server resource budget the executor injects.

> **CORRECTION (post-Phase 15).** The paragraph above said `rs_model_analysis`
> was also absent because "the RS model does not exist". That is no longer
> true: the tool IS registered (`agent/registry.py`) and dispatches the visual
> step to a provider VLM. The allowlist is therefore **four** tools, not three,
> and it is still closed - `ToolCall` remains a discriminated union and an
> unrecognised name still fails validation before dispatch. The tool observes
> ONE already-retrieved Sentinel-2 PNG and its statement is recorded as a
> model observation, never as a measurement: `source="model"` is excluded from
> the grounding numeric authorities, so it cannot authorise a number.

### Gemini structured-output compatibility — the Commit 4 fix

Pydantic emits `discriminator` and `oneOf` for a discriminated union, and
google-genai 2.20.0's own `Schema` model **forbids both**, so sending
`AgentPlan` directly fails at request time. Constructing
`GenerateContentConfig(response_schema=AgentPlan)` succeeds and proves nothing -
the SDK stores the model and translates it later, which is exactly how the
problem hid.

**Implemented fix:** the provider builds an SDK-compatible `types.Schema` that
expresses the same union with **`any_of`** over two concrete branches, using the
SDK's public `Schema.from_json_schema` (never the private `_transformers`). It
is derived from the contracts - tool names from `TOOL_REGISTRY`, the intent
shape from `SatQueryIntent`, step bounds from `AgentPlan` - so it cannot drift.
This is a **generation hint only**: the response is still parsed through
`AgentPlan`, which remains the sole validation authority.

The lesson is pinned in tests: the fake client now performs the *same*
`t_schema` translation the real request path performs, so an untranslatable
schema can never again pass a green suite.

### The four operational statuses

| Status | Meaning | What survives |
| --- | --- | --- |
| `ok` | grounded answer produced | everything |
| `planner_unavailable` | no plan; nothing ran | nothing is claimed |
| `synthesis_unavailable` | tools ran, prose failed | the evidence |
| `answer_withheld` | answer generated, failed validation | the evidence, trace, checks |

All four are **HTTP 200**. Deterministic-fallback semantics:

- evidence is preserved when synthesis or grounding fails;
- planner failure fabricates **no** evidence and no plan - `answer_validation`
  stays `None`, so an unchecked answer can never read as a validated one;
- an answer that fails validation is **withheld**, never presented as
  successful, and never replaced with substitute prose.

The measurements are the product; the sentence is a presentation of them.

**Grounding is containment, not proof.** It establishes that every number in an
answer is traceable to evidence at the precision stated, that citations
resolve, and that no forbidden phrase appears. It does NOT establish
qualitative correctness or causal attribution. It is nonetheless conservative
about prose: a qualitative sentence must repeat a citation to survive, so an
arbitrary unsupported sentence does NOT simply pass. What stays outside its
reach is a claim that is supported by the evidence and still wrong about what
the evidence means. That limit is documented in `grounding.py` rather than
papered over.

**Frontend boundary.** `AgentPanel` posts to `/api/v1/query/agent` and nothing
else. It performs no planning, execution, grounding or provider call, renders
`Plan -> Tools selected -> Execution -> Evidence -> Answer`, and shows **no
reasoning, thoughts, thinking or chain-of-thought** - there is no such field in
any contract to render. Tool labels come from the response; a withheld answer
shows the evidence and an honest statement of absence.

**Phase 15 does NOT contain:** image input to any model · vision-language
reasoning · multimodal fusion · an RS-adapted model · ReAct or any loop ·
multi-agent behaviour · model-generated code or tool names · caching,
persistence or job queues · new dependencies · changes to `/query/execute` or
`/query/analyze` · changes to `api/router.py`, the raster path, or any
analysis/query/satellite module.

### Known limitations

- **Live Gemini execution has now been verified** (at HEAD `09cea4f`, model
  `gemini-3.6-flash`, the shipped default). The `any_of` generation schema is
  accepted by the real endpoint; NDWI returned `status="ok"` 3/3, S1+S2
  multimodal 2/2 with unique modality-namespaced evidence ids and no HTTP 500,
  and temporal execution ran end to end. All three failure statuses were
  observed against the real provider - `planner_unavailable` and
  `synthesis_unavailable` on upstream 503/504, `answer_withheld` on a real
  grounding rejection - with evidence preserved exactly as documented. Two
  fail-closed grounding limits remain: a negated sentence containing
  "co-registered" is still withheld (the phrase scan is polarity-blind, by
  design), and a threshold published only in a measurement *name*
  (`ndwi_percent_above_index_threshold_0.3`) is not in the allowed values, so
  citing it fails. Both over-reject; neither is a bypass.
- The executor produces **no evidence item explaining a discovery failure**, so
  when discovery fails the synthesizer receives empty evidence with no
  indication why. Reported rather than patched - creating that item is the
  executor's responsibility, not the service's.
- `IntentParsingError` is reused for unusable planner/synthesizer output. Its
  code string reads `intent_parse_error`, which is imprecise for a plan or an
  answer; a better-named error would mean editing `core/errors.py`.
- `AgentService` is deliberately **not** zero-argument constructible - its
  collaborators must be injected - so it is absent from the `test_services`
  contract list.

### Baseline after Phase 15 — VERIFIED

| Check | Result |
| --- | --- |
| `pytest -q` | **939 passed** (458 before Phase 15) |
| `ruff check .` | clean |
| `git diff --check` | clean |
| `npm run test` | **80 passed** (58 before Phase 15) |
| `npm run lint` / `typecheck` / `build` | clean / clean / builds |

Every commit was written test-first: the tests were added, observed failing for
the expected reason, and only then satisfied.

---

## 16. Pre-UI hardening sprint — IMPLEMENTED

An audit-and-remediate pass across multitemporal correctness, the multimodal /
SAR boundary, resource safety, API/error contracts, code quality and security.
Every finding below was reproduced independently before it was changed, and
every fix is pinned by a test that was mutation-checked - the mutation was
applied, the test observed failing, and the mutation reverted.

### Confirmed and fixed

**Sentinel-1 display imagery was a false capability (P0).** Documented as
working; non-functional against live Earth Search. See pipeline step 8 - now
corrected to discovery-only, with `_require_readable_scheme` refusing the
`s3://` asset at the boundary and naming the cause.

> HISTORICAL. This paragraph records the state at the time of that sprint and
> remains true of Earth Search GRD only. Sentinel-1 is NO LONGER discovery-only:
> see pipeline steps 8b (RTC VV/VH imagery) and 8c (quantitative backscatter).

**`NdwiTemporalChange` labelled its axes by requested role (P1).**
`TemporalComparison` does not require `baseline` to precede `target`, and
pairing orders by ACQUISITION time, so an inverted request reported the target
scene as the baseline. Reproduced live: requesting baseline=December /
target=June returned `baseline_scene_id` = the June scene, and the same
`change_mean` as the non-inverted request. The arithmetic was right and the
label was wrong, which is the harder kind to notice. Fixed by renaming the
axes to `first_*` / `second_*` (earlier / later), matching the neutral
vocabulary `ObservationPair` and `TemporalIndexComparison` already used for
exactly this reason; the requested roles stay readable in `window_label`, and
the frontend now reads "Earlier" / "Later".

**`time_windows` was unbounded (P0, resource).** `SatQueryIntent` accepted
50,000 windows. Execution runs one catalog search per (modality x window)
sequentially, so one unauthenticated request produced unbounded outbound
requests against a third-party catalog - measured at 1000 upstream searches
for 500 windows x 2 modalities - plus unbounded memory when imagery was on.
Bounded by `MAX_TIME_WINDOWS = 24` at the `SatQueryIntent` boundary, so every
path that builds an intent inherits it.

**An explicit `provider` could not override an unconfigured default (P1).**
FastAPI resolves a dependency before the handler body runs, so building the
configured default decided the run before the request's own `provider` was
read: a deployment holding only an NVIDIA key could not use NVIDIA, and the
502 named Gemini - a provider the caller never asked for. `get_agent_service`
now defers that failure and the handler re-raises it only when the request
names no provider of its own, so the actionable message naming the variable is
preserved for the case it was written for.

**Two different JSON shapes shared status 422 (P2).** Pydantic rendered
`{"detail": [...]}` and `AppError` rendered `{"error": {...}}`; the frontend
reads only the latter and silently discarded which field was wrong. A
`RequestValidationError` handler now uses the one envelope, with a distinct
`validation_error` code so a malformed body stays distinguishable from a
semantically invalid one. The offending VALUE is never echoed.

**An href's scheme was checked, the rest of it was not (P2, security).**
Control characters passed: the scheme in `https://x/a.tif\x00.s3` is a
perfectly good `https`. A NUL truncates the path for any C consumer (GDAL and
curl are C) and CR/LF are the separators of an HTTP request, so either can make
the request sent differ from the URL that was checked. Now refused before the
scheme test.

**The config rail did not mark unimplemented tasks (P2).** `ConfigSummary`
rendered an active "Change detect" chip with no qualifier while the task
selector correctly read "Change Detection (unavailable)" - letting a Temporal
NDWI Statistics result be read as a change-detection result, the one confusion
that layer exists to prevent. Both surfaces now say it, and a test pins both.

### Verified correct, left unchanged

- **Grid identity for paired-pixel change.** `_grids_are_comparable` refuses on
  absent CRS, CRS mismatch, dimension mismatch and any affine inequality, with
  no tolerance. Exercised across all eight branches; a sub-millimetre origin
  shift is refused. Confirmed live: two same-tile scenes shared an identical
  transform and produced a change whose `change_mean` equalled the
  independently computed difference-of-means to six decimal places.
- **The VLM numeric quarantine.** A number can never be authorised by
  model-sourced evidence; `source="model"` is excluded from the numeric
  authorities and the exclusion is keyed on source, not on the absence of a
  measurement.
- **Exactly one image reaches the visual model**, fetched once and encoded
  once - measured, not inferred. The "double fetch" hypothesis was disproved.
- **Provider isolation, now proven in BOTH directions** with vacuity checks on
  each tripwire. Previously only NVIDIA-selected-never-reaches-Gemini was
  tested.
- **Secrets reach neither responses nor logs**, canary-tested through the full
  HTTP stack and mutation-checked.
- **STAC identifier validation** covers both URL-building paths; traversal,
  query/fragment injection and overlong ids are all refused before any call.

### Accepted trade-off, deliberately not changed

`providers/gemini.py` catches `Exception` around its SDK calls where
`providers/nvidia.py` catches only `httpx` errors, so a bug in the Gemini path
is recoded as a 502 rather than surfacing as a 500. The asymmetry is
defensible rather than arbitrary - the google-genai SDK is a large third-party
exception surface and httpx is a narrow one - and narrowing it would route SDK
exception text into `logger.exception`, trading a hidden bug for a possible
credential in a log. **KNOWN LIMITATION**, recorded rather than silently
traded away.

### Baseline after the hardening sprint — VERIFIED

| Check | Result |
| --- | --- |
| `pytest -q` | **1360 passed** (1333 at sprint start) |
| `ruff check .` / `git diff --check` | clean / clean |
| `npm run test` | **211 passed** |
| `npm run lint` / `typecheck` / `build` | clean / clean / builds |

---

## 17. Multi-index spectral analysis — IMPLEMENTED

NDVI and NDBI join the existing NDWI. All three are the same normalised
difference over different Sentinel-2 band pairs, so they share one engine and
therefore one set of numerical guarantees:

    NDVI = (nir    - red) / (nir    + red)     10 m
    NDWI = (green  - nir) / (green  + nir)     10 m   (unchanged)
    NDBI = (swir16 - nir) / (swir16 + nir)     20 m limiting

**NDWI was not rewritten.** `_ndwi_values`/`_ndwi_grid` became thin wrappers
over the shared core, and a test pins that the generic path returns byte-equal
statistics. Live check: `ndwi_mean` over the Marina Beach AOI is 0.075406,
identical to the value recorded before the generalisation.

**Raw DN still cancels.** The decision rests on both bands of a pair sharing
one multiplicative scale. Verified live (2026-09): `red`, `green`, `nir`,
`swir16` and `swir22` all advertise scale 0.0001 / offset -0.1, so every pair
cancels. A test pins the assumption so a future index over a differently
scaled band cannot silently invalidate the arithmetic.

**The 10 m / 20 m problem, and why this one is tractable.** Read from the live
COG headers: `nir` is 10980² at 10 m and `swir16` is 5490² at 20 m, both
EPSG:32644 with the SAME origin (399960, 1500000), and 10980 = 2 x 5490. The
grids are exactly 2:1 nested with no sub-pixel phase offset, so
`coregister_to_finer_grid` assigns each fine pixel the value of the coarse cell
containing its centre. That invents nothing - every number in the output
already existed in the source - which is why this direction was chosen over
averaging NIR down to 20 m, which would synthesise values never measured. It
creates no detail: NDBI is sampled at 10 m and resolved at 20 m, and that is
reported as a warning on every NDBI result. The mapping goes through both
affine transforms rather than assuming a factor, so a non-nested pair produces
out-of-bounds parents and is REFUSED. Different CRS, non-integer ratios and
uncovered pixels are all refused or marked invalid rather than filled.

**Agent integration is ONE tool, deliberately.** `spectral_indices` takes an
`indices` list rather than shipping three sibling tools, because the plan
budget is three steps: a tool per index would leave no room for discovery and a
visual observation in the same run. It is the only analysis tool with a
parameter, and the distinction holds - the model chooses WHICH index answers a
question; the bands, the raw-DN decision and the co-registration rule stay
engine constants. A test asserts the params carry nothing but `tool` and
`indices`.

**Evidence and grounding move together.** Each index attributes its own
measurements (`ndvi.*`, `ndwi.*`, `ndbi.*`) rather than inheriting a single
"ndwi" source, and `EvidenceSource` plus `_NUMERIC_AUTHORITIES` gained both new
sources. Omitting the latter fails silently and in the worst direction - a real
pixel-derived number read as ungrounded, withholding a correct answer - so a
test now pins it and was mutation-checked.

**Live end-to-end.** "What is the vegetation condition… Compute NDVI." →
Gemini planned `execute_query, spectral_indices` → read exactly B08 and B04 →
`accepted=True, numeric=pass, terms=pass, refs=pass`, 5 citations.

**Band reads are shared**: three indices cost four reads, not six, because NIR
is common to all three. Measured. (Five since M3: the `scl` quality layer is read
once per scene and shared by all three - section 23.)

## 18. Auditable evidence export — IMPLEMENTED

`features/agent/evidenceReport.ts` builds the record entirely from state the
browser already holds, so it asks the server for nothing and cannot disagree
with what was displayed. It carries the answer, the validated plan, the
grounding checks, the flattened evidence, the manual-path evidence, every
warning, and - deliberately - the unflattering parts: a withheld answer, a
failed check, a window whose imagery could not be retrieved. A report that
recorded only successes would not be an audit.

Filenames are derived from the geocoded place, which is untrusted for that
purpose: anything outside `[a-z0-9]` becomes a separator, so traversal and
separators cannot survive. A test proves `../../etc/passwd` cannot escape, and
a mutation-checked test proves no credential-shaped material can reach the file.

### Baseline

| Check | Result |
| --- | --- |
| `pytest -q` | **1386 passed** |
| `npm run test` | **226 passed** |
| ruff / eslint / tsc / build / `git diff --check` | clean |

---

## 19. Anthropic (Claude) as a third AI provider — IMPLEMENTED

A third inference backend behind the existing abstractions. Additive
throughout: no route changed, no contract field was added or removed, and the
deterministic half of the system is untouched.

**Not verified against the live Anthropic API.** Every test here runs through a
hand-written recording fake. What is established is selection, isolation,
request shape and error mapping; what is NOT established is that a real Claude
model plans well against this tool allowlist. Treat that the way Phase 15
treated Gemini before its live run.

**Where it lives.** `app/services/agent/providers/anthropic.py` implements all
four provider-neutral roles - `AgentPlanner`, `AnswerSynthesizer`,
`VisualAnalyst` and `IntentParser` - through one shared
`_AnthropicMessagesClient`. The official `anthropic` SDK is a new dependency
and is confined to that single file, exactly as `google-genai` is confined to
`gemini.py`; an AST test asserts the importer list is exactly
`['providers/anthropic.py']`, plus a non-vacuity test that the file really does
import it. The SDK was chosen over raw `httpx` (the NVIDIA route) because
Anthropic publishes a first-party async client; reimplementing its
authentication, retry and error taxonomy would buy nothing.

**Selection is table-driven now.** `AI_PROVIDER_FIELDS` in `core/config.py`
maps each provider to its `Settings` key field, model field and environment
variable, and `SUPPORTED_AI_PROVIDERS` is derived from it. `Settings.
api_key_for()` / `.model_for()` replaced the `if provider == "gemini" else
nvidia` branches in `factory.py` and `api/routes/ai.py`. This is the direct
lesson of the Phase 15 defect that `test_provider_isolation.py` documents -
only the visual step was provider-aware, so planning silently stayed on Gemini
- and a test now pins `set(AI_PROVIDER_FIELDS) == SUPPORTED_AI_PROVIDERS`.
Every credential message is unchanged in wording; it is now formatted from the
table rather than written out per provider.

**Three request-shape decisions, each a fact about the models, not a taste:**

- **No `temperature`.** The Gemini and NVIDIA adapters pin it to 0.0. Sampling
  parameters were REMOVED from the current Claude models and return HTTP 400,
  so sending one would not add determinism - it would fail every request.
- **No `thinking` and no `output_config`.** Claude thinks adaptively by default
  on current models, and `output_config.effort` is rejected by some older ones;
  sending neither keeps the adapter compatible with any model id an operator
  configures, catalogued or not. Reasoning is never *read*: `_message_text`
  takes text blocks only, so no thinking block is parsed, stored, returned or
  rendered and the Phase 15 rule holds unchanged.
- **No server-side refusal fallbacks.** Anthropic can reroute a refused request
  to another model automatically. Deliberately not enabled: a silently
  substituted model would break the attribution that travels with every
  observation, which is the same reason there is no fallback BETWEEN providers.

`stop_reason` is inspected before any parsing: `refusal` becomes an upstream
failure and `max_tokens` becomes an explicit "cut off" error, because truncated
JSON would otherwise be reported as a malformed plan - blaming the model for a
budget this adapter set.

**Catalog.** Three Claude models: `claude-opus-5` (the default),
`claude-sonnet-5`, `claude-haiku-4-5`, all `endpoint_type="anthropic-messages"`.
Every one accepts image input, so unlike the NVIDIA section there is no
text-only entry and no capability refusal to encode - a test asserts the visual
and text model sets are identical.

**Isolation is proven in every direction.** `test_provider_isolation.py` now
arms an Anthropic tripwire (all four role classes plus the shared messages
client) and asserts Gemini- and NVIDIA-selected runs never touch it, with a
vacuity check that the tripwire fires; and an Anthropic-selected run is checked
against the Gemini AND NVIDIA tripwires armed simultaneously. Every provider is
credentialed in that fixture, so an absent touch is never explained by an
absent key. `test_provider_credential_isolation.py` gained an
`ANTHROPIC_SENTINEL` and now also refuses `sk-ant-` and `x-api-key` anywhere in
a response or a log - Anthropic authenticates with `x-api-key`, not
`Authorization`.

**Frontend.** `AiProvider` gained `"anthropic"`; `AgentPanel` and
`ModelSelector` label it **Claude**. Nothing else changed - the browser still
posts to one endpoint and never sees a key.

**Phase 19 does NOT contain:** tool use, structured outputs, prompt caching,
extended-thinking configuration, streaming, batches, the Files API, Managed
Agents, any change to the tool allowlist, grounding, evidence shape, the raster
path, or any analysis/query/satellite module.

### Baseline after Phase 19 — VERIFIED

| Check | Result |
| --- | --- |
| `pytest -q` | **1892 passed** (1836 before) |
| `ruff check .` / `git diff --check` | clean / clean |
| `npm run test` | **268 passed** (267 before) |
| `npm run lint` / `typecheck` / `build` | clean / clean / builds |

---

## 20. Productionization pass — IMPLEMENTED (2026-09-15)

Every item here was reproduced live before it was changed and re-verified live
after: Gemini `gemini-3.6-flash`, NVIDIA
`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`, Earth Search, Planetary
Computer and Nominatim, plus the real UI in a browser.

### Confirmed and fixed

- **NVIDIA planning was non-functional.** On the `json_object` path the
  configured model returned tool names as bare strings
  (`{"steps": ["execute_query", "ndwi_statistics"]}`), so every NVIDIA run was
  `planner_unavailable`. `NvidiaAgentPlanner` now forces one tool call whose
  schema is `AgentPlan.model_json_schema()` and falls back to `json_object` only
  when a model answers without a tool call. `_PLAN_MAX_TOKENS = 4096` (measured
  1,430-2,179 completion tokens per plan). A missing tool call logs
  `finish_reason` only, never content.
- **Planners returned valid but incomplete plans** (both providers).
  `agent/plan_completion.py` adds exactly the step a question explicitly asks
  for: the named index (NDVI/NDWI/NDBI acronym, single-window Sentinel-2), the
  visual observation ("visible", "look like", "can you see"; the user's question
  verbatim, at most 500 chars, never truncated), and temporal NDWI (compare mode
  over Sentinel-2 plus "water"/"NDWI"). The result is re-validated through
  `AgentPlan`, a planner step is never dropped, and `trace.plan` keeps the
  planner's own plan.
- **Synthesis abstained on about half of direct questions** with identical
  evidence. Root cause: the rendered evidence named metric and value but not
  place, window, scene or date. `_render_evidence` now prefixes a non-citable
  CONTEXT block built from `evidence.execution` (6/6 grounded, versus 2/4
  before). Values are shown at four significant figures (`_display_value`);
  stored evidence, the API response and the export keep full precision, and a
  property test proves every displayed value is groundable.
- **Comparison answers were withheld.** Models reused the single-scene template
  ("The mean NDWI was X" twice), which grounding correctly refuses as
  ambiguous. The synthesis instruction now carries an exact comparison template
  (earlier / later / difference, full ids); a test pins it groundable. Live:
  Gemini answered it grounded.
- **NVIDIA synthesis and visual calls died on one 503.** `_retry_transient`
  retries 429/5xx up to three attempts; malformed responses and timeouts are not
  retried.
- **STAC search had no retry.** `_post_search` makes a second attempt after a
  transport error, 429 or 5xx, for both Earth Search and Planetary Computer.
- **A failed step left a silent hole.** The executor now adds
  `execution.discovery_failure`, `execution.analysis_failure` or
  `execution.visual_failure` text items (system-authored message, no
  measurement, never a `visual` field). This RESOLVES the section-15 known
  limitation about discovery failures.
- **An NDVI run could be labelled `object_identification`**, which made
  `analysis.status` read `not_implemented` and the UI highlight "Object ID (not
  implemented)". The planning instruction now states when each task applies.
- **`_WARN_AGGREGATE` said flatly "No pixels were compared against one
  another"** beside the paired-pixel change; it is now scoped to
  `mean_ndwi_difference`.
- **A temporal run's headline showed one observation, unlabelled.** Both
  observations' means are named `ndwi_mean`, so the Analysis Result headline
  set the EARLIER mean large as "ndwi mean" beside an answer about change.
  `headlineMeasurements` now headlines a comparison by `mean_ndwi_difference`
  and `ndwi_change_mean`, or by nothing when those were suppressed.
- **README facts corrected**: the S1 data source, the `/query/parse` provider,
  the retired NVIDIA model, the GRD contradiction, co-registration wording, and
  the cloud / scene-selection rule.

### Verified correct, left unchanged

- S2 NDVI/NDWI/NDBI, temporal NDWI with a same-grid paired change, S1 RTC VV/VH
  backscatter and georeferenced overlays, all live. Overlays align with the OSM
  basemap in Chennai (EPSG:32644) and Bengaluru (EPSG:32643).
- Scene selection: S2 lowest cloud cover, then earliest, then id; S1 earliest,
  then id. Selection itself still uses only scene-level cloud cover. (At the
  time of this pass no per-pixel cloud mask existed; since M3 the optical
  indices are masked per pixel by the Sentinel-2 SCL - section 23.)
- Attribution uses the run snapshot: switching the provider after a run does
  not relabel it. A new query replaces the answer, map, evidence and scenes.

### Known limitations and external constraints

- Free tiers are the practical limit. Gemini returned 429 on first attempts for
  long stretches; NVIDIA's hosted NIM answered about half of sampled requests
  with 503 or a timeout. Both surface honestly as `planner_unavailable` or
  `synthesis_unavailable` with the evidence preserved.
- The NVIDIA model occasionally ignores the forced `tool_choice`
  (`finish_reason=stop`); the planner's three-attempt loop re-samples it.
- Plan completion is keyword-narrow by design. A question that implies an
  analysis without naming it remains the planner's judgement.
- Browser automation: a Chrome tab reporting `visibilityState: hidden` pauses
  `requestAnimationFrame`, so MapLibre never fires `load` and the map looks
  black or stuck at its initial view until an input event forces a frame. This
  is not a product defect; verify maps with a forced frame.
- The UI's raw evidence list shows full-precision floats (the headline metrics
  are formatted). P2.

### Baseline after the productionization pass — VERIFIED

| Check | Result |
| --- | --- |
| `pytest -q` | **2024 passed** (1892 at the start of the pass) |
| `ruff check .` / `git diff --check` | clean / clean |
| `npm run test` | **270 passed** (268 at the start of the pass) |
| `npm run lint` / `typecheck` / `build` | clean / clean / builds |

---

## 21. Local open-weight provider (Ollama + Qwen3-VL) — IMPLEMENTED (2026-09-15)

A fourth AI provider that runs on the developer's machine, so SatQuery can plan,
look and describe without a cloud quota. Additive: the planner, executor,
grounding, evidence shape, geospatial services and frontend architecture are
unchanged; the local model is trusted exactly as little as a cloud one.

**Where it lives.** `app/services/agent/providers/local.py` implements
`AgentPlanner`, `AnswerSynthesizer`, `VisualAnalyst` and `IntentParser` over
Ollama's native `POST /api/chat` with plain `httpx` (no SDK). `format` carries
a JSON Schema that Ollama compiles into a decoding grammar: `DraftAnswer`'s and
`SatQueryIntent`'s own schemas, and for planning a grammar DERIVED from
`AgentPlan`'s (`_plan_grammar`: a required `discovery` slot holding the
`execute_query` step, then `analysis` with up to two other tools, `tool`
required everywhere). The provider only orders the two slots into `steps`; the
Pydantic contract still validates the result, so the closed tool union remains
the authority. Requests
send `think: false`, temperature 0 and `num_ctx` from settings. The visual role
sends the exact PNG the pipeline retrieved, base64 in `images`.

**Configuration is table-driven.** `ProviderFields.api_key` may be `None` for a
keyless provider, which names an `endpoint` field instead;
`Settings.is_configured()` replaced the key-only checks in the factory and the
models route. Variables: `AI_PROVIDER=local`, `LOCAL_AI_BASE_URL`
(`http://127.0.0.1:11434`), `LOCAL_AI_MODEL` (`qwen3-vl:4b-instruct`),
`SATQUERY_LOCAL_AI_TIMEOUT_SECONDS` (300), `SATQUERY_LOCAL_AI_NUM_CTX` (4096).

**No fallback, honest failure.** Unreachable Ollama ->
"Local AI provider is unavailable. Start Ollama and ensure the selected Qwen3-VL
model is installed."; 404 -> "not installed. Install it with: ollama pull
<tag>"; memory -> "not enough free memory"; timeout -> "did not answer within N
seconds"; any other error answer -> one follow-up `GET /api/tags`, and if Ollama
cannot list its models -> "Ollama is running but cannot read its installed
models. If they are stored on an external drive, check that it is connected,
then try again.", otherwise "The local AI provider failed to answer.". Ollama's
own error text is read to classify, never repeated.

**Detection.** `GET /api/v1/ai/models` asks the local endpoint once
(`GET /api/tags`: 2 s to connect, 10 s to answer) and reports each local model
as Ready / Not installed / Ollama not running / Ollama cannot read models
(reachable, but no model list: `ProbeFailure.MODELS_UNREADABLE`). The only
reachability check in the catalog - justified
because the service is on this machine. The frontend badge is green only when
the server says "Ready" (it was green for configured+compatible, i.e. also for
"Not installed").

### Findings - VERIFIED

- **Hardware:** Apple M2 MacBook Air, 8 GB unified memory, 6-12 GB free disk
  (volume 95-97% full), 7-10.5 GB of swap in use with ordinary apps open.
  `qwen3-vl:8b` (6.14 GB of weights) is excluded here; 4B is the default and 2B
  the fallback. Ollama 0.20.2 was already installed and serving.
- **Every plain `qwen3-vl` tag is the thinking variant.** `4b`, `8b`, `2b`
  and `30b` share config and weight digests with their `-thinking` tags. That
  variant ignores `think: false`: a one-word reply cost 220 generated tokens
  (967 characters in `message.thinking`), and a real planning call did not
  finish in 300 s. The catalog therefore lists `-instruct` tags only
  (`2b-instruct`, `4b-instruct`, `8b-instruct`, `30b-a3b-instruct`;
  `30b-instruct` does not exist), and a test pins that.
- **Thinking 4B, measured:** cold 42.0 s (load 19.3 s), warm 17.4 s for a
  trivial prompt; 5.30 GB resident at an 8K context; free memory 38% -> 5% and
  swap 7.1 -> 10.5 GB while loaded.
- **Discovery must be first in the grammar itself.** Decoding under the plain
  `AgentPlan` schema, the 4B instruct model returned
  `{"steps": [{"tool": "ndwi_statistics"}]}` - well-formed steps, discovery
  missing - identically on every re-ask at temperature 0. The "exactly one
  execute_query, first" rule lives in a validator that a schema-derived grammar
  cannot see, and Ollama 0.20.2 ignores `prefixItems`. The two-slot grammar uses
  only constructs Ollama honours (required properties, `oneOf` over `$ref`);
  verified live 3/3 valid plans (NDWI, visual water, temporal compare), 5-17 s
  each.
- **Context:** the largest prompt (temporal synthesis) is about 2.2k tokens, so
  4096 fits every role and halves the 8K KV cache. A prompt that fills the
  window is logged as a warning (Ollama truncates from the start). Measured
  2026-09-16 from Ollama's `/api/ps`: 4.37 / 4.48 / 4.64 GB resident at
  `num_ctx` 2048 / 3072 / 4096; the largest prompt in the live matrix was 1,726
  tokens (+98 generated). 4096 kept: 3072 saves 0.16 GB, 2048 cannot hold a
  2.2k prompt.
- **Network:** the model pulls ran at 0.1-6 MB/s with repeated CDN connection
  resets; one pull was stopped for low memory and resumed. External.

### Verified live

- Detection in the API and in the UI selector (default Local, badge
  "Not installed" before the instruct model was present).
- Honest failure in the API and the UI, with zero cloud calls in the backend
  log, while the model was not installed.
- Switching in the UI: NVIDIA answered "The mean NDWI was 0.1464 index."
  (grounded); Gemini was reached and refused by its free-tier quota (429).

### Verified live on `qwen3-vl:4b-instruct`

- **Smoke test 5/5** (`backend/scripts/local_model_smoke.py`): intent
  (Marina Beach, January 2025); a valid `AgentPlan` (`execute_query` +
  `ndwi_statistics`); the real retrieved 112x300 Sentinel-2 PNG described as "a
  long, narrow strip of sandy beach bordered by dark water on one side and a
  developed coastal area with buildings and vegetation on the other";
  synthesis "The mean NDWI was 0.1464 index." (grounded); a question inviting
  invented figures answered with an abstention; an invented NDWI refused.
- **In the UI with `AI_PROVIDER=local`:** NDWI 0.1464, NDVI -0.06136 and NDBI
  0.01184, each grounded (numeric/refs/terms pass) and attributed "Produced by
  Local · qwen3-vl:4b-instruct"; round trips 94 / 110 / 131 s. Evidence panel
  complete (scene, acquisition, CRS, bounds, 33,600 valid pixels).
- **Detection:** the selector badge reads Ready for the installed tag and "Not
  installed" for the others. It went stale when a model was installed while the
  page was open; the catalog is now re-read on window focus and visibility.
- **Performance:** cold load about 20 s; 4.64 GB resident at `num_ctx` 4096;
  planning 10-82 s, observation about 20 s, synthesis 13-52 s. The same call
  varied three- to eightfold with swap pressure (9.9-10.9 GB of swap in use).
- **Browser automation:** partway through, the Chrome extension's tools began
  failing with "Couldn't determine which page this action targets" while the
  tab was still valid, so the last two checks went through the same
  `POST /api/v1/query/agent` endpoint the UI calls, with `provider: local`:
  - **Temporal NDWI:** "The earlier mean NDWI was 0.02665 index. The later mean
    NDWI was 0.1464 index. The mean NDWI difference was 0.1197 index." -
    grounded, three full-id citations; the local planner chose
    `temporal_ndwi_statistics` itself (S2A 2024-01-15 -> S2B 2025-01-04); 100.6 s.
  - **Visual:** "Yes, there is visible water in the image, appearing as the
    dark blue expanse along the coastline." - an attributed observation of the
    real retrieved scene (`visual_claims: attributed`). The planner chose NDWI
    statistics; plan completion added the observation the question asked for;
    32.4 s.

### Hardening pass (2026-09-16) - VERIFIED

Found live on the reference machine, with the models on a USB hard disk
(`~/.ollama/models -> /Volumes/Expansion/ollama/models`, exFAT):

- **False "Ollama not running" after the drive slept.** macOS spins the disk
  down after 10 idle minutes (`disksleep 10`); Ollama then needed ~4 s to answer
  `/api/tags`, past the probe's single 2 s budget, so a running Ollama was
  reported as not running. The probe now allows 2 s to connect, 10 s to answer.
- **A real USB dropout.** Mid-matrix the drive dropped off (it re-enumerated,
  disk6 -> disk4). Ollama kept running, answered `/api/tags` with 500 and every
  chat with 400 "model is required" in milliseconds; SatQuery said "Ollama not
  running" (catalog) and "The local AI provider failed to answer." (query) -
  honest, wrong about the cause. The probe now returns
  `ProbeFailure.MODELS_UNREADABLE` for a reachable Ollama that gives no usable
  model list (non-200, malformed, read timeout; a connect failure is still "not
  running"), the catalog says "Ollama cannot read models", and an unclassified
  chat error asks `/api/tags` once and names the drive. Ollama recovered without
  a restart once the drive was back (same process), as the dangling-symlink
  simulation predicted; it will not START while the drive is missing.
- **The local plan adapter dropped unknown keys.** `_plan_from_slots` read the
  two slots and silently ignored any other top-level key, so a plan carrying a
  smuggled `"steps": [{"tool": "shell"}]` or `"evidence"` key was ACCEPTED
  (replayed against the old code) - one level above the contract's
  `extra="forbid"`. It now refuses anything but exactly `discovery` (a dict)
  and `analysis` (a list).
- **The NDBI caveat withholding was a genuine grounding mismatch, not
  conservatism.** A sentence repeating a cited engine caveat verbatim passed
  `_prose_supported` but failed numeric grounding on the caveat's own figures
  ("20 m", "10 m"; the SAR caveat's `10*log10` had the same trap).
  `_ungrounded_claims` now skips a sentence that repeats - whitespace, case and
  closing punctuation aside, every digit, sign and decimal exact - a sentence of
  a CITED `*.warning.N` / `*.limitation.N` item from a numeric authority. Model
  observations never qualify (source), executor failure notes never qualify
  (id), and the figures authorise nothing elsewhere.
  `tests/test_grounding_verbatim_caveat.py`: 4 of its tests failed on the old
  code, all 11 pass now.
- **Considered and left alone:** `num_predict` (largest output observed: 192
  tokens; a runaway is bounded by the timeout and fails validation);
  `keep_alive` (documented `OLLAMA_KEEP_ALIVE` instead - holding 4.6 GB of 8 GB
  longer is the operator's call); a raster read retry (a truncated COG tile read
  under a network saturated by a model download rejected the visual step and
  the model abstained - honest, and outside this pass).

| Live check (real Ollama, real Sentinel data) | Result |
| --- | --- |
| A. Ollama running | Ready, green badge |
| B. Ollama stopped | "Ollama not running"; query `planner_unavailable` with the agreed message; 0 cloud requests; relaunch -> Ready |
| C. Model missing | `planner_unavailable`, "...not installed. Install it with: ollama pull qwen3-vl:2b-instruct"; 0 steps; 0 cloud requests |
| D. Drive unavailable | the real dropout above, and a replay: "Ollama cannot read models" / the drive message; recovers without a restart |
| E. Timeout | `SATQUERY_LOCAL_AI_TIMEOUT_SECONDS=1` -> stated failure after 1.0 s |
| F. Malformed answer | non-JSON, list, no message, non-text content, truncated plan -> `intent_parse_error`, executor never called (tests) |
| G. Invalid plan | 13 malicious local outputs refused after one re-ask, never reaching the executor; a URL or path as a place name only ever reaches the geocoder's `q` (tests) |
| H. Secrets | real Gemini and NVIDIA keys loaded, recording proxy in front of Ollama: absent from local traffic, logs, `backend.log` and the response; no Authorization header; 0 cloud calls |

**Verified with the drive physically detached (2026-09-16 09:45).** The real
condition the probe change was written for, not the simulation: Ollama running
(`/api/version` 200) with `/api/tags` 500 because `~/.ollama/models` dangled.
The catalog reported "Ollama cannot read models" for all four local models, a
local query returned `planner_unavailable` with "Ollama is running but cannot
read its installed models. If they are stored on an external drive, check that
it is connected, then try again.", and the run touched 0 steps, 0 evidence and
0 cloud endpoints. Before this pass the same state read "Ollama not running"
and "The local AI provider failed to answer."

**Sentinel-1 after the provider recovered (same session).** Planetary Computer
answered 200 again (its `*.azureedge.net` certificate mismatch had cleared).
With no model involved at all, the deterministic pipeline computed NDVI
-0.0613608, NDWI 0.146391, NDBI 0.0118356 (S2B_44PMV_20250104, 23.7 s) and SAR
VV -5.44372 dB, VH -17.8496 dB, VV-VH 12.4059 dB (S1A ... 20250111, 13.7 s).
Through the agent on NVIDIA: "The mean VV was -5.444 dB. The mean VH was -17.85
dB." and "The mean NDWI was 0.1464 index.", both grounded, citing the
measurement ids, with the provider-RTC caveat carried in the evidence and no
claim that SatQuery performed terrain correction. The earlier SAR failures were
the outage, nothing else.

### Qwen3-VL bake-off (2026-09-16) - 2B and 4B VERIFIED

Protocol, identical for every model: one model loaded at a time (everything
resident is unloaded first), `num_ctx` 4096, never during a model download
(a download saturates the link and truncates COG reads), backend restarted
with `LOCAL_AI_MODEL=<model>`. Tooling outside the repo in
`~/satquery-bakeoff/`: cold T1 intent parse, T2 planner (x2), T10 live
adversarial prompt, T6 vision on ONE cached real scene
(`S2B_44PMV_20250104_0_L2A`, 112x300 PNG, identical bytes and question for
every model, x2), two end-to-end API passes of NDVI / NDWI / NDBI / visual
features / temporal / SAR VV+VH / refusal, 3 API intent parses, memory sampled
every 2 s.

| | 2B | 4B |
| --- | --- | --- |
| Cold first request (model load, USB disk) | 32.2 s (28.2 s) | 62.8 s (40.8 s) |
| Planner median (range) | 5.0 s (3.1-7.7) | 11.6 s (9.6-22.1) |
| Vision probe / synthesis median | 5.6 s / 3.8 s | 4.0 s / 6.3 s |
| End-to-end median, pass 1 / pass 2 | 11.4 / 12.6 s | 24.8 / 22.1 s |
| Resident / free memory min, median | 2.96 GB / 12%, 20% | 4.64 GB / 8%, 13% |
| Swap over the session | 3.33 -> 3.47 GB | 3.68 -> 4.68 GB |
| NDVI, NDWI, NDBI | plan-fail, ok, plan-fail | ok, ok, ok (all grounded) |
| Visual, temporal, refusal | withheld, synthesis timeout, withheld | ok, ok, ok |
| SAR VV+VH | network (both) | network (both) |
| API parses / cloud calls | 3/3 / 0 | 3/3 / 0 |

Failure classification - the model, SatQuery and the network are kept apart:

- **2B NDVI/NDBI - model.** Its raw plans repeat `spectral_indices`
  (`[ndvi]` then `[ndwi]`; `[ndbi]` twice); `AgentPlan` refuses a repeated tool
  by design and the one re-ask repeats it.
- **2B visual - model.** It filled all three plan slots
  (`execute_query`, `ndwi_statistics`, `spectral_indices`), so plan completion
  could not add the observation; completion adds, never removes.
- **2B temporal - model.** Synthesis ran away until the 300 s timeout, twice;
  SatQuery's bound held and reported it. A `num_predict` cap would make such a
  runaway fail in seconds - recommended, not applied mid-bake-off.
- **2B refusal - model, contained.** It wrote an unsupported number; grounding
  withheld the answer.
- **SAR, both models - external.** `planetarycomputer.microsoft.com` served a
  genuine Microsoft certificate for `*.azureedge.net` (hostname mismatch, curl
  exit 60); every Sentinel-1 search failed TLS verification and SatQuery said
  "The satellite catalog is unavailable." Never bypassed.
- **An earlier 4B regression run - network.** Run during the 2B download:
  nearly every raster read truncated; every answer was an honest abstention and
  no number was invented. Re-run on a clean network: all tasks ok.

Vision, identical scene and question ("What water features are visibly present
in this scene?"), deterministic across both runs:

- **2B:** "There is a large body of water along the coast, and there are small,
  scattered patches of water near the beach." (the scattered patches are not
  evident in the scene)
- **4B:** "A large body of dark water, likely the sea or ocean, is visible
  along the right side of the image, bordering a sandy beach. The water appears
  to be a continuous expanse with a clear shoreline."

Browser, 4B default: landing, 3D Earth, Try SatQuery, Local Ready green,
Not installed amber, Ollama-down amber, a real click-and-type NDWI run
(grounded, "Produced by Local · qwen3-vl:4b-instruct", evidence complete), a
selector switch leaving the finished result's attribution untouched, NVIDIA
(one transient 503, then a grounded NDVI in 31 s), Gemini (routed, refused by
its free-tier quota: 429 twice), a new query replacing the old result, and a
temporal headline of the difference, not one scene. The map's bounds and
canvases were correct; its rendering could not be judged because the Chrome
window was hidden (requestAnimationFrame pauses).

### Bake-off status (2026-09-18) — what is and is not measured

Three models were benchmarked: `qwen3-vl:2b-instruct` and `qwen3-vl:4b-instruct`
carry usable numbers; `qwen3-vl:8b-instruct` was measured and **failed
outright**. `qwen3-vl:30b-a3b-instruct` was never installed and has been
abandoned.

**8B does not run on this machine.** Two full passes, 14 E2E attempts, **0
`ok`** — every one returned `planner_unavailable` with "did not answer within
300 seconds". Every auxiliary probe timed out as well: intent ×2, planner ×2,
adversarial, and both vision runs. `backend.log` recorded **zero** "Local model
answered" lines. Pass 1 took ~301–307 s per task; pass 2 degraded to
1021–1672 s as the machine thrashed. The model needs **7.63 GB resident**
(5.11 GB on GPU) on an **8 GB** machine: across 12,109 s the sampler saw free
memory bottom out at **2%**, swap climb 7,565 → 12,611 MB with a **14,590 MB
peak**, and was itself starved for one **1,074 s** gap between nominally 2 s
samples. This is not a tuning problem; there is no headroom to tune.

Two caveats on that run, both against it rather than for it:

- Unrelated tool installs ran concurrently, so 8B's free%/swap columns are
  contaminated. They cannot explain the outcome — installs cost hundreds of MB,
  not the 7.6 GB that is the actual cause — but do not quote them as clean.
- The run logged 3 cloud requests. Those were the driver's 3 closing stability
  parses: `/api/v1/query/parse` resolves the **default** provider (Gemini), not
  `LOCAL_AI_MODEL`. This was **not** a local→cloud fallback — provider selection
  logged 14 × `provider=local`, zero otherwise, and no task produced any answer
  at all. It does mean 8B's "stability parse" timings are Gemini's, not 8B's,
  and that the earlier 2B/4B "0 cloud requests" figures were unreliable: they
  were counted against a `backend.log` that had since been replaced.

**30B was abandoned.** Three attempts; the last two were killed by the memory
watchdog mid-download at 15 GB of 19 GB. The partial blob is preserved on disk
and nothing was deleted. Note for anyone resuming it: the volume is exFAT, which
has no sparse-file support, so ollama preallocates the blob at full size — the
file measuring ~19.6 GB never indicated how much had actually been fetched.

Do not infer 30B behaviour from the 2B→4B trend. That trend already broke at 8B:
quality rose from 2B to 4B while cost rose too, and at 8B the cost simply
exceeded the machine and quality went to zero.

The repository default is unchanged: `local_ai_model = "qwen3-vl:4b-instruct"`.

### Tests

`tests/test_local_provider.py` (request shape, contract authority, every
failure message, discovery, selection without cloud keys, grounding withholds a
number the local model invents); isolation both ways in
`test_provider_isolation.py` (a local run with Ollama down and every cloud
provider credentialed AND armed fails honestly and touches none);
`tests/test_local_credential_isolation.py` (with every cloud key set to a
sentinel, no local role sends a credential or an Authorization header, and a
local failure discloses none in its response or logs); the table test covers
keyless providers; `conftest.py` stubs the catalog probe so the
suite never calls a real Ollama (verified: zero `/api/tags` calls during a
run). Frontend: local attribution and badge readiness. Live, not CI:
`backend/scripts/local_model_smoke.py`.

### Baseline — VERIFIED

| Check | Result |
| --- | --- |
| `pytest -q` | **2060 passed** |
| `ruff check .` / `git diff --check` | clean / clean |
| `npm run test` | **274 passed** |
| `npm run lint` / `typecheck` / `build` | clean / clean / builds |

---

## 22. Production readiness pass — IMPLEMENTED (2026-09-17)

An audit-and-remediate pass over correctness, resource safety, provenance,
observability and deployment. Sixteen audit findings were re-verified against
this tree before anything was changed; two did not hold as described and are
recorded as such. Nothing was committed or pushed.

### Correctness

- **A parsed intent was silently narrowed by the manual form.**
  `QueryPanel.currentIntent()` rebuilt the intent from the form's own controls,
  so anything the form had no control for was dropped between parsing and
  executing: a TIME SERIES collapsed to `windows[0]` ("monthly, January through
  March" executed as January alone) and an NDWI THRESHOLD never reached the
  wire. Neither failed - both answered a different question. The panel now holds
  the parsed series and threshold, offers a "Time series (N windows)" mode, and
  emits `ndwi_threshold` when the request stated one. Contract tests assert the
  executed body equals the parsed intent.
- **Plan completion honoured substring presence, not intent.** "Show imagery
  only. Do not calculate NDVI." added NDVI. Completion now classifies each
  mention - refusal, quotation, supposition or definition - and adds nothing
  unless it is a request. Deliberately biased one way: an unclear mention adds
  nothing, because a missed completion leaves the planner's own judgement while
  an invented one executes work the user refused. `_DEFINITION_LEAD` separates
  "What is NDWI?" from "What is the NDWI of Chennai?" by the determiner.
- **The MapLibre worker was never emitted.** MapLibre resolves its worker
  relative to its own module URL; rolled into the app chunk, that pointed at
  `/assets/maplibre-gl-worker.mjs`, which Vite never built. A single-page host
  answers that with `index.html`, so the browser loaded HTML as JavaScript - the
  basemap rendered (it needs no worker) and footprint geometry did not.
  `maplibreWorker.ts` imports the worker with `?worker&url` (which bundles it
  WITH its dependencies) and states the URL through MapLibre's own
  `setWorkerUrl`. The build now emits `maplibre-gl-worker-<hash>.js` and the app
  chunk references it.

### The suite was not independent of the machine

2112 passed with a real `GEMINI_API_KEY` present; 2108 passed and 4 failed
without one - same commit, same command. `tests/conftest.py` now clears every
provider variable AND stops `Settings` reading the developer's `.env`. No
credential is invented: a test needing one sets an obviously fake value itself.

### Resource and policy controls

- `app/core/limits.py`: per-client rate limit (429 + `Retry-After`), workflow
  slots (503 after a brief wait), a raster gate, and a body-size refusal (413)
  installed INSIDE the CORS middleware so a refusal still carries CORS headers.
- The geocoder's budget is **application-wide**, in
  `geospatial/nominatim.py`: one request at a time, spaced, with a TTL cache and
  one bounded retry. "One per second per user" would let ten users send ten.
- **Every limit is per process.** Documented in `DEPLOYMENT.md`, and the
  production image runs one worker for that reason.

### Honest results

- **Partial execution.** One window's catalog failure aborted the whole run,
  discarding windows that had already succeeded. Failures are now recorded
  against their window (`ExecutedWindow.error`), `QueryExecutionResult.status`
  is derived (`completed`/`partial`/`failed`), and a run where NOTHING succeeded
  still raises rather than returning an empty result as success.
- **Completeness.** `status: "ok"` is derived from the TASK, so an NDWI request
  over an execution with no optical window returned "ok" with no measurements.
  `analysis_outcomes` and a derived `completeness` now answer what was asked for
  and what came of it; `status` keeps its exact previous values.
- **Mixed-provider provenance.** The top-level `catalog` was assigned inside the
  execution loop, so whichever window ran last spoke for all of them. Each
  window carries its own `catalog`; `catalogs` lists every service that answered.
- **Selection scope.** The catalog's `numberMatched` was read and discarded, so
  "best of the 10 returned" and "best of 900 matching" looked identical.
  `scenes_matched` is now carried through to the window.
- **Execution integrity** (`services/query/integrity.py`). `/query/analyze`
  accepts a client-supplied `QueryExecutionResult`; Pydantic proves its shape,
  not that its parts agree. The relations are now checked - selected scene among
  the scenes returned, counts coherent, imagery belonging to the selected scene,
  windows belonging to the intent. **Periods bind, labels describe**: a window's
  period must be one the intent requested; a label the intent assigns must mean
  what the intent says.

### Readiness, observability, provenance

- `/health` stays liveness. `/ready` reports each capability and answers 503 when
  one is missing - and never makes a paid provider call. The header no longer
  says "Operational" when the selected AI path cannot run.
- `core/observability.py`: a run id in a `ContextVar` stamped onto every log
  line (including third-party loggers) plus stage timings. A failing stage logs
  the exception CLASS, never its message.
- The manual evidence export now carries the submitted intent, per-window
  catalogs, discovery failures (kept apart from imagery failures) and the
  analysis's own warnings and completeness.

### Security

- Asset hrefs come from an external catalog. A non-public address is refused -
  measured on this Python: `100.64.0.1` (CGNAT) is **not** `is_private`, and
  `224.0.0.1` (multicast) **is** `is_global`, so the rule is "not global or
  multicast". `SATQUERY_TRUSTED_ASSET_HOSTS` narrows reads further when set.
  KNOWN LIMITATION: the check is on the name, not on the address it resolves to,
  so DNS rebinding is not defeated.
- The frontend client no longer throws a raw `SyntaxError` on a non-JSON body
  (an HTML error page, an empty body); every failure is an `ApiError` carrying
  the status.

### Two findings that did not hold as described

- **"Application-factory settings do not propagate"** - true, and worse than
  described: `api/router.py` called `get_settings()` at IMPORT time, so
  `create_app(settings=...)` could not change a route prefix at all. Fixed by
  building the router per application.
- **"2,108 passed / 4 failed"** - reproduced exactly, but the cause was the
  developer's `.env`, not only the ambient environment.

### Deliberately NOT done

- `ruff format` - 82 of 124 files would be reformatted; doing it here would bury
  a behavioural diff in whitespace. Separate work.
- **Docker images were never built or run**: the daemon is unavailable on this
  machine. `docker compose -f docker-compose.prod.yml config` parses and
  resolves; the images themselves are UNVERIFIED.
- Durable jobs. Browser cancellation still does not cancel backend work, bounded
  by `SATQUERY_WORKFLOW_BUDGET_SECONDS` and documented.

### Baseline

| Check | Result |
| --- | --- |
| `pytest -q` | **2277 passed** (2112 before, with a credential present) |
| `ruff check .` / `git diff --check` | clean / clean |
| `npm test` | **307 passed** (276 before) |
| `npm run lint` / `typecheck` / `build` | clean / clean / builds |

---

## 23. Scientific core — M1–M5 validation stages — IMPLEMENTED (2026-09-23)

Before any index or backscatter number exists, four stages run in order. Each
refuses BEFORE the cost it protects. Stage 1 failures are HTTP 422; stages 2-4
degrade the affected operation to a warning plus an `unavailable` outcome
(the existing per-operation architecture), never to a silent number.

| Stage | Module | Decides | Refusal |
| --- | --- | --- | --- |
| 1 Request (M1) | `analysis/validation.py` | operation allowlist, dataset/sensor, AOI (WGS84 rectangle; native-resolution size from the reader's OWN limits), Date 1 strictly before Date 2, parameters | `AnalysisRequestRejectedError`, 21 codes. The agent applies the area rule after geocoding and BEFORE any STAC search (`QueryExecutionService.execute(before_discovery=...)`, `AnalysisService.precheck_plan`) |
| 2 Scene + asset (M2) | `satellite/scene_validation.py` | the CATALOG item is the authority (a client's `Scene` date, cloud, footprint are never used); sensor and collection; AOI coverage by footprint geometry; each asset exists, is a COG with role `data` and the expected dtype (`scl` is `uint8`); processing metadata recorded | `SceneValidationError` (`scene_not_found`, `scene_does_not_cover_aoi`, `required_asset_missing`, `unsupported_asset_type`, `unsupported_asset_encoding`, `unknown_processing_baseline`, `incompatible_sensor`, `incompatible_collection`, `temporal_scene_incompatible`, ...) |
| 3 Pixel quality (M3) | `analysis/pixel_quality.py` | the Sentinel-2 SCL is placed on the FINAL 10 m analysis grid by whole-cell assignment (never blended - averaging classes 4 and 8 would invent 6, water); usable classes 4/5/6 only; every pixel counted once (nodata > saturated/defective > cloud > cloud shadow > snow > unknown class > other); the mask goes into `BandWindow.valid` BEFORE every statistic, overlay, threshold count and temporal change | `PixelQuality` per index grid and per temporal observation; `{index}_quality_*` measurements. SCL missing or unreadable: the index is not computed |
| 4 Radiometric (M4) | `satellite/radiometry.py` | are the values on the representation the formula assumes, AS-IS? Optical: no additive offset and one shared scale/unit per band pair. SAR: linear power (never dB, never scaled/offset). Nothing is ever corrected | `RadiometricState` (`verified` / `verified_with_unknown_metadata` / `incompatible` / `undetermined`); `RadiometricValidationError`. Incompatible and undetermined are refused before any read (for SAR, before any asset is signed) |

**SCL classes** come from ESA SentiWiki (S2 Processing); no catalog publishes
`classification:classes`. Baseline 05.11 renamed class 2 DARK_FEATURES ->
CAST_SHADOWS. Excluding 2 (cast shadow) and 7 (unclassified) is a SatQuery
policy choice, stated once in `SCL_CATEGORY`.

**The Sentinel-2 offset - measured, not assumed.** Baseline 04.00 (2022-01-25)
introduced an additive offset. Earth Search publishes `earthsearch:boa_offset_applied`
and a `raster:bands` offset of -0.1, and they disagree. Live, tile 44PMV:

| Baseline | Flag | Declared offset | Pixels (bounded water read, NIR DN median) | M4 |
| --- | --- | --- | --- | --- |
| 03.01 | false | 0 | 346 - no offset | usable (`not_introduced`) |
| 04.00 | false | -0.1 | 256-433 in 3 of 3 scenes - NO offset found | `undetermined`, refused |
| 04.00 / 05.00 / 05.09 / 05.11 | true | -0.1 | 338 (2023); NDVI 0.637 vs impossible 1.30 if applied (section 7) | usable (`removed_by_provider`), -0.1 recorded as a contradicted conflict and never applied |

Rule, in ONE place (`OFFSET_INTRODUCED_BASELINE`): offset-free iff the flag is
true on a baseline >= 04.00, or the baseline predates 04.00 with no non-zero
declared offset. Everything else is `undetermined`: metadata cannot establish
it, and measurement did not find the offset the metadata claims, so it is not
called `incompatible` either. Earth Search does not link ESA's product XML
(the authoritative `BOA_ADD_OFFSET`). The scene PAIR rule compares the
representation the pixels carry, not the flags: 03.01 (false) + 05.09 (true)
are both offset-free and ARE comparable; comparing flags had refused them.

**No thresholds anywhere in M1-M4.** `SATQUERY_SCENE_MIN_AOI_COVERAGE` defaults
to 0.0 (partial coverage is reported, not refused). No valid range is invented
from `bits_per_sample` (15, recorded). Saturation comes from SCL class 1 only.

**Contracts (additive):** `AnalysisResult.pixel_quality`, `.radiometry`;
`ObservationIndexResult.pixel_quality`, `.radiometry`. Optical reads: `scl` is
read first, so a temporal comparison is six reads, and NDVI+NDWI+NDBI is five.

### M5 - geometric validation (`analysis/geometry.py`)

One authoritative rule set; `_require_matching_band_grids`,
`coregister_to_finer_grid` and `_grids_are_comparable` all delegate to it. A
band pair, a paired temporal change and a VV-VH difference need IDENTICAL grids
(same CRS - equivalent spellings accepted - north-up, equal resolution per
axis, equal dimensions and origin). A coarser raster placed on the 10 m grid
(NDBI's SWIR, the SCL) must be EXACTLY NESTED: integer ratio on EACH axis, cell
edges on pixel edges (a 20 m grid shifted 5 m is refused - before M5 it was
silently misassigned), and overlap. Tolerance is 1e-6 px, for float noise only.
Checked twice: before any read from the catalog's `proj:*` source grids (M2
now records them per asset), and after the read on the actual windows. Unknown
CRS is refused (the NDWI statistics and threshold count used to report on a
CRS-less grid; two tests were updated to the refusal). The paired temporal
change and the VV-VH difference are withheld on a mismatch; each side's own
statistics never needed a common grid. `GridState` rides on
`AnalysisResult.grids`, `ObservationIndexResult.grid` and
`TemporalIndexComparison.pair_grid`, refusals included. Live: tiles 44PMV
(EPSG:32644) and 43QBA (EPSG:32643), baselines 03.01/05.09/05.11 - SWIR and SCL
nest 2:1 at offset 0; RTC VV/VH share the item grid.

### Known limitations

- About 9% of 2022-23 scenes over Chennai (13 of 145: baseline >= 04.00 with the
  flag false) are refused as `undetermined`, although measurement found them
  offset-free. Scene SELECTION does not yet prefer radiometrically usable scenes.
- `read_band` refetches the STAC item per band (correct, not optimal); the band
  pair is promoted to float64 twice per index (quality + engine).
- SCL misclassification is not assessed. The frontend is untouched: it does not
  type the new fields, and the new quality measurements appear in its raw list.

### Baseline - VERIFIED

| Check | Result |
| --- | --- |
| `pytest -q` | **2748 passed** (2316 before M1; +164 M1, +79 M2, +58 M3, +66 M4, +64 M5, +1 re-parametrized) |
| `ruff check .` / `git diff --check` | clean / clean |
| frontend | not re-run in M1-M4: no frontend file changed |

---

## 24. M5.5 - Natural-language intent layer, provider-independent - IMPLEMENTED (2026-09-24)

**The rule, stated once:** a request that names neither `provider` nor `model`
is interpreted by the STANDARD workflow - no model, no credential. Naming one
opts that run into AI interpretation. Applies to `POST /query/agent` and
`POST /query/parse` (which gained an optional `provider`). `AI_PROVIDER` now
only chooses the AI provider for a request that names a model without a
provider, the catalog default, and the optional readiness report.

**Where it lives.**
- `agent/interpretation.py` - pure: text -> `QueryInterpretation` or
  `ClarificationRequiredError`. Vocabularies map to NDVI, NDWI, NDBI, SAR
  backscatter, two-period NDWI comparison, or true-colour imagery; polarity is
  read by `plan_completion.requested_matches` (made public, not duplicated), so
  "show water, not vegetation" asks for water alone. Place = the phrase after
  around/in/at/of/near/over..., passed VERBATIM to the existing geocoder -
  never resolved here. Dates = explicit only: ISO day/month, "15 January
  2025", "January 15, 2025", "January 2025", "2025", "from X to Y",
  "between X and Y" (shared trailing year), "10 to 20 January 2025". No
  today, no default window, no season or "early 2024" boundaries, no year for a
  bare month. Sensor follows the analysis; a contradicting source ("vegetation
  using radar") is questioned.
- `agent/standard.py` - `StandardPlanner`, `StandardReport` (one fixed
  sentence per engine value - the grounding templates - plus "Scene X was
  selected." / "The scene was acquired on D.", or the abstention),
  `StandardIntentParser`.
- `AgentService`: a planner clarification -> `needs_clarification`; after
  execution, discovery code `not_found` -> `location_not_found` and M1's
  `aoi_too_large` -> `area_too_large` (M1's own message, M1 untouched).
- Contract: `AgentStatus` + `needs_clarification`; `AgentResult.clarification`
  (`reason`, `message`, `options`, `understood_*`); integrity: present exactly
  with that status, never beside an answer or a failure.
- `/ready`: new `interpretation` capability; `ai_provider` is `required: false`
  and no longer decides readiness. A missing REQUIRED capability is still 503.
- Frontend: selector default "Standard · no AI model"; a default run is
  attributed "the standard workflow · no AI model", never the default AI;
  clarification shown under the query box; pipeline "Needs clarification";
  examples are all standard-supported (NDWI, vegetation, built-up, radar,
  temporal); header ignores optional capabilities.

**Deliberately refused, never approximated:** visual questions ("visible",
"looks like") -> `requires_ai_model`; floods, counting, ships/vehicles,
classification, land cover -> `analysis_unsupported`, even beside a supported
analysis; any comparison other than water (temporal NDVI/NDBI/SAR is not
implemented) -> `analysis_unsupported`; a place given only by "this"/"here" ->
`location_missing`.

**Mutation-checked (8/8 caught, each restored byte-identical):** NDVI->NDWI (3
tests fail), SAR->optical (4), temporal->single (12), place discarded (13),
dates discarded (13), unsupported executed (7), AI made mandatory (5), fake
result substituted (11).

**Verified live (local backend, all three provider keys blanked, real
Nominatim / Earth Search / Planetary Computer, and the UI in Chrome):**
- "Show vegetation around Chennai" -> `date_missing`, 0 s, nothing run.
- "... around Chennai in January 2025" -> `area_too_large` (20.9 x 42.4 km),
  refused before any STAC search.
- "Show vegetation around Marina Beach, Chennai in January 2025" -> 5 scenes,
  S2B_44PMV_20250104, NDVI -0.06131 over 33,524 px (76 SCL-masked), grounded.
- "Analyze SAR backscatter around Marina Beach, Chennai in January 2025" ->
  Sentinel-1 RTC S1A ... 20250111, VV -5.444 dB, VH -17.85 dB, grounded.
- "Compare water at Marina Beach, Chennai between January 2024 and January
  2025" -> S2A 20240115 vs S2B 20250104, mean NDWI difference 0.12, grounded.
- Backend log: every run `provider=standard model=none`; zero AI endpoints.

**Known limitations.** Place extraction needs a lead word ("NDVI Chennai
January 2025" is asked to name the place); a map-drawn AOI is not an input
(no draw tool exists); "this reservoir" is not resolved from the map; the
deployed Render backend does not have M5.5 until it is pushed.

### Baseline - VERIFIED

| Check | Result |
| --- | --- |
| `pytest -q` | **2853 passed** (2748 at M5; +103 `test_standard_workflow.py`, +1 readiness non-vacuity, +1 split parse test) |
| `ruff check .` / `git diff --check` | clean / clean |
| `npm run test` | **367 passed** (358 before) |
| `npm run lint` / `typecheck` / `build` | clean / clean / builds |

---

## 25. M5.6 - Local SatQuery intent model - IMPLEMENTED (2026-09-24)

**Scope.** Natural language -> ONE operation label, and nothing else. The
model computes no index, picks no raster, sees no coordinate and no date:
places and dates are replaced by `PLACE` / `DATE` (`classifier_text`) before it
reads the question, and the deterministic reader stays authoritative for both.
The scientific pipeline (M0-M5) is unchanged.

**Labels (exactly the backend's capabilities):** NDVI, NDWI, NDBI,
SAR_BACKSCATTER, TEMPORAL_NDWI, TRUE_COLOR, CLARIFICATION, UNSUPPORTED.

**Model.** scikit-learn Pipeline: TF-IDF word 1-2-grams + char_wb 2-5-grams
(min_df 2), sublinear TF, L2 per block -> multinomial LogisticRegression, C=10
(grouped 5-fold CV). Exported to ONE gzip JSON artifact (185,182 bytes; 2,135
word + 3,500 char features) - data, never a pickle - and run in numpy
(`intent_model.py`), so production needs no scikit-learn (dev dependency only;
`uv.lock` production set unchanged). numpy vs scikit-learn: max |dp| 2.0e-9.
Predict ~0.05 ms, full operation routing ~0.2 ms, ~3.7 MB peak to load.
Reproducible: same dataset + `scripts/train_intent_model.py` -> same bytes.

**Data** (`backend/data/intent/`, see its README): 1,018 hand-written
examples, 127-129 per label, many places worldwide, typos, vocabulary-free
paraphrases. Split BY FAMILY (near-duplicates and cross-label skeleton twins
never straddle a split): 714 train / 152 validation / 152 test. A 120-item
challenge set was written after every choice was frozen and evaluated once.

**Decision rule** (`intent_router.decide_operation`): no model, invalid output
or confidence < threshold -> the M5.5 rules decide alone. Confident analysis
label -> executed only if the rules read no analysis (the model adds recall) or
the same one (agreement; the rules' full set is kept). Contradiction ->
`analysis_ambiguous` clarification (new reason). UNSUPPORTED -> refused.
CLARIFICATION -> the rules ask their specific question. Guards (unsupported,
visual, sensor contradictions) and slots are always the rules'. Threshold
0.90 = lowest grid value with >= 99% accepted precision on out-of-fold train +
validation predictions; shipped inside the artifact.

**Results.** Test (n=152): accuracy 0.9145, macro F1 0.9161, weighted F1
0.9161; at 0.90, 74 acted on, 0 wrong. Challenge (n=120): accuracy 0.9417,
macro F1 0.9401; at 0.90, 75 acted on, 0 wrong. Full pipeline, WRONG operation
executed: model+rules 0 (test) / 0 (challenge); rules only 2 / 0. Correct:
model+rules 127 / 101 vs rules only 120 / 92. The test split informed
interpreter fixes, so the challenge set is the clean number; 15 of its 120
items are >= 0.75 similar to a training sentence after masking (short
skeletons recur) - stated in the report.

**Interpreter fixes found by the evaluation (M5.5 code):** a place phrase
swallowed a change verb ("Ukai dam differed") and lowercase request words
("Srinagar for water", "the rice crop near X"); "normalized difference ...
index" read as a comparison; "N D V I" / "S.A.R." not read; heat, temperature,
disease, pests, volume/depth and species were executed as indices instead of
refused. All fixed; M5.5 tests unchanged and green.

**Security.** Artifact validated field by field on load (format, exact label
set, shapes, finiteness, sizes); refused -> rules only. Predictions are
`IntentPrediction(label: Literal[8], confidence: finite 0..1, extra=forbid)`;
malformed output -> rules. Intent modules import no provider, transport,
pickle or scikit-learn (AST test); inference passes with sockets blocked.

**Mutation-checked (9/9, restored byte-identical):** NDVI->NDWI (9 fail),
NDWI->NDBI (5), SAR->optical (3), temporal->single (3), unsupported->analysis
(2), low confidence executed (1), schema bypass (6), external provider call
(35), engine replaced by model output (10).

**Verified live** (local backend, every AI key blank, Ollama unreachable, real
Nominatim / Earth Search / Planetary Computer, real UI in Chrome): NDVI
-0.06131 (76 px SCL-masked), NDWI 0.1466, NDBI 0.01244, SAR VV -5.444 / VH
-17.85 dB, temporal NDWI difference 0.12 (S2A 20240115 vs S2B 20250104),
true colour imagery only, "Analyze Chennai" -> analysis_missing, "Count ships"
-> analysis_unsupported. "How much concrete is around Marina Beach, Chennai in
January 2025" - unanswerable by the rules - routed by the model (NDBI 0.93) and
measured. Log: every decision `intent_model`, every run `provider=standard`,
0 AI endpoints.

**Known limitations.** Coverage at 0.90 is ~49-63%: many vocabulary-free
phrasings ("How leafy is ...") are asked back rather than guessed. Single label
per question (multi-analysis questions rely on the rules' explicit terms).
English only. Place extraction still needs a lead word. A retrain must rerun
the script and keep `tests/test_intent_model.py` green (it pins the report).

### Baseline - VERIFIED

| Check | Result |
| --- | --- |
| `pytest -q` | **2938 passed** (2853 at M5.5; +85 `test_intent_model.py`) |
| `ruff check .` / `git diff --check` | clean / clean |
| `npm run test` | **368 passed** (367 before) |
| `npm run lint` / `typecheck` / `build` | clean / clean / builds |

---

## 26. Demo readiness / UX hardening, intent model v2 - IMPLEMENTED (2026-09-24)

No scientific change: M0-M5, formulas, SCL, radiometric and geometric rules and
the analysis area limit are untouched. No new model, provider, service or
dependency. Everything here is reading, wording, data and presentation.

**Example chips.** "Vegetation around Bengaluru" and "built-up area around
Hyderabad" were refused by the M1 area gate (`area_too_large`) - whole cities
are larger than one native 10 m read. Option A: replaced with real places
inside those cities, each verified live to return a measurement:
"Show vegetation around Cubbon Park, Bengaluru in December 2024." (NDVI 0.5204,
16,988 px, S2B_43PGQ_20241208) and "Show built-up area around Ameerpet,
Hyderabad in January 2025." (NDBI -0.02482, 196,620 px, S2B_43QHV_20250107).
January 2025 at Cubbon Park was NOT used: its least-cloudy scene is S2C on
baseline 05.11 with `boa_offset_applied=false`, which M4 correctly refuses as
`radiometric_undetermined` (section 23's known limitation - selection does not
yet prefer radiometrically usable scenes). Nothing was loosened to make a chip
pass. A frontend test pins that every chip names a comma-qualified place.

**Places without lead words** (`interpretation._residual_place`). When no
"around / in / at / near / of" phrase names a place, the ONE contiguous run of
name-like words the question leaves is the place ("NDVI Chennai January 2025",
"water Dal Lake January 2025", "Isle of Man NDVI 2024"). Analysis, comparison,
season, visual, detection and unsupported words, commands, question words and
generic nouns are removed first; in mixed case only Capitalised words join; two
separate runs are asked back as candidates; a Capitalised analysis word beside
the run ("water Forest Hill") is asked back for confirmation. The geocoder
stays the authority. Fixed on the way: `_DEICTIC` matched a PREFIX, so
"Mysuru", "Itanagar" and "Hereford" read as "my" / "it" / "here" (M5.5 bug);
a lowercase visual phrase ended nothing, so "the image of Pune looks like" made
"Pune looks like" the place; opening auxiliaries ("Did", "Has") became a second
candidate.

**Clarification UX.** Every clarification asks for everything missing at once,
in plain words, with no placeholder and no internal name ("Analyze Chennai" ->
"What would you like to analyse at Chennai?" + five options; "Show vegetation"
-> place AND date; "Compare water" -> place AND both periods). New
`AgentClarification.option_questions`: one complete question per option, built
only from what the user said (their place and their periods, never a guessed
date); a validator requires one per option or none. The UI shows options
capitalised and, when the questions line up, as links that FILL the query box
- nothing runs until the user presses Run. `area_too_large` and
`location_not_found` no longer print `'<landmark>, <city>'`; the area message
names the user's own place ("a neighbourhood, park or landmark in Chennai
together with the city name").

**Rule gaps the evaluation exposed, fixed** (tests in
`tests/test_demo_readiness.py`, section C):
- change verbs are comparisons: declined, risen/rising, rose, fell/fallen,
  drop(ped), shrinkage, receded, dried up/out ("fall" left out - it is also a
  season). Before, "Has the vegetation of Kodagu declined from 2020 to 2024?"
  EXECUTED a single NDVI over 2020-2024; now it is refused (vegetation change is
  not implemented) and a water change runs temporal NDWI;
- yield and forecasting are refused ("Predict the crop yield near Bathinda"
  used to execute NDVI);
- vocabulary: leafy, foliage, canopy, lush, verdant, greening (NDVI); developed
  area/land (NDBI); sigma/gamma naught, sigma0, gamma0 (SAR);
- "How leafy is X" was read as a DEFINITION ("<term> is ..."); a lowercase word
  straight after "how" is now a degree question (`plan_completion.
  _DEGREE_QUESTION`); "What is vegetation?" and "How NDVI is computed" stay
  definitions.
The model tests that used "How lush are the tea gardens" as a phrase the
vocabulary misses now use "How green are the tea gardens" ("green" is left out
of the vocabulary on purpose - place names).

**Intent model v2** (`satquery-intent-v2`, `data/intent/README.md`). Dataset
v2 = v1 + ~30 diverse examples per label targeting the recurring v1 challenge
misses (leafy/lush/green/crop vigour; extent/surface water/spread;
concrete/developed/impervious/urbanisation; radar/backscatter/microwave;
changed/rose/fell/before vs after/compared with/from-to), with hard negatives
(vegetation or built-up change -> UNSUPPORTED; bare "compare" ->
CLARIFICATION). 1,258 rows. A fresh 120-item challenge set was written BEFORE
any v2 training. Same features and protocol; the threshold rule gained a floor
(`THRESHOLD_FLOOR = 0.90`, never lowered for coverage) and calibration itself
chose **0.95**; C = 100. Artifact 190,567 bytes, 2,217 word + 3,639 char
features, numpy vs scikit-learn max |dp| 2.4e-9, predict ~0.05 ms.

| Held-out challenge v2 (n=120), full pipeline | acted on (wrong) | correct | clarified | WRONG executed |
| --- | --- | --- | --- | --- |
| DEPLOYED M5.6 (v1 model + M5.6 rules, from `git archive HEAD`) | 80 (0) | 90 | 28 | **2** |
| v1 model + today's rules | 81 (0) | 114 | 6 | 0 |
| **v2 model + today's rules (shipped)** | **106 (0)** | **118** | **2** | **0** |
| rules only, today | - | 102 | 18 | 0 |

The FIRST, blind v2 evaluation (threshold then 0.90, rules before the fixes
above): v2 114 / 5 / 1 wrong, v1 109 / 9 / 2 wrong, rules only 93 / 24 / 3
wrong. The rule fixes were informed by that evaluation (and by the v2 test
split), so the table above is not fully blind for the items that exposed them.
Model-only on challenge v2: accuracy 0.9833, macro F1 0.9830; 39 of 120 items
are >= 0.75 similar to a training sentence after masking (short skeletons).
Test split (n=184): accuracy 0.9402, macro F1 0.9406; at 0.95, 125 acted on, 0
wrong; full pipeline 175 correct / 9 clarified / 0 wrong (rules only 162 / 21 /
1). Challenge v1 (development for v2): 114 / 6 / 0 (deployed M5.6: 101 / 19 /
0). Every model-acted prediction on every set was correct.

**Mutation-checked (7/7, restored byte-identical):** deictic prefix (3 fail),
change verbs (3), forecast refusal (3), degree question (3), visual cut (1),
area placeholder (1), frontend option/question alignment (1).

**Verified locally against live services** (standard workflow, no AI key
used): the 9-query matrix - NDVI -0.06131, NDWI 0.1466, NDBI 0.01244 (Marina
Beach, S2B_44PMV_20250104), SAR VV -5.444 / VH -17.85 dB, temporal NDWI
0.02658 -> 0.1466 (difference 0.12), "How leafy is Marina Beach" -> NDVI,
"NDVI Chennai January 2025" -> `area_too_large` naming Chennai's 20.9 x 42.4 km,
"Analyze Chennai" -> five clickable options, "Count ships in Chennai harbor" ->
`analysis_unsupported`.

**Verified in production** (`12d24c3` on Render - `/ready` names
`satquery-intent-v2` at 0.95 - and `4c3a5a1` on Vercel; standard workflow, no AI
key; real UI in a Playwright browser). Every example chip ran from the UI to a
grounded answer with evidence: NDWI 0.1466 (Marina Beach, S2B_44PMV_20250104),
NDVI 0.5204 (Cubbon Park, S2B_43PGQ_20241208), NDBI -0.02482 (Ameerpet,
S2B_43QHV_20250107), SAR VV -5.444 / VH -17.85 dB (S1A ... 20250111_rtc),
temporal NDWI 0.02658 -> 0.1466 (difference 0.12, overlay on the basemap). The
matrix: vegetation / built-up / "how leafy" around Marina Beach -> NDVI
-0.06131 / NDBI 0.01244 / NDVI -0.06131; "NDVI Chennai January 2025" ->
`area_too_large` naming Chennai (no placeholder); "Analyze Chennai" -> five
clickable options, a click fills the box and runs nothing; "Count ships in
Chennai harbor" -> `analysis_unsupported`; "Show vegetation", "Compare water",
"Vegetation around Chennai" ask for exactly what is missing. Console: 0 errors,
0 warnings (the `favicon.ico` 404 is fixed by `4c3a5a1`); every API call 200; no
CORS error.

**External constraint found in production: the public Nominatim geocoder
throttles Render's shared outbound IP.** From 19:45 to 20:26 IST most geocodes
from Render returned HTTP 429 while the same User-Agent from a developer
machine got 200. SatQuery reported it honestly ("Scene discovery did not
complete: The geocoding service responded with status 429.", pipeline
Failed, no number invented), and each place worked once a geocode got through
(the 15-minute geocode cache then serves repeats). Not fixed here - a different
geocoder or a self-hosted Nominatim would be a new service, out of scope. For a
live demo, run each example once beforehand to warm the cache.

**Known limitations.** "NDVI Chennai January 2025" is correctly refused as too
large: sub-areas are not suggested (option B was not chosen). Scene selection
still ignores radiometric usability, so some place/month pairs return a scene
with the index withheld (honestly, with the reason). "fall" is not read as a
change verb. Vague water-level phrasings ("how did the water level move") stay
the model's call. English only.

### Baseline - VERIFIED

| Check | Result |
| --- | --- |
| `pytest -q` | **3000 passed** (2938 at M5.6) |
| `ruff check .` / `git diff --check` | clean / clean |
| `npm run test` | **372 passed** (368 at M5.6) |
| `npm run lint` / `typecheck` / `build` | clean / clean / builds (with `VITE_API_BASE_URL`) |

---

## 27. Geocoder reliability on a shared outbound IP - IMPLEMENTED (2026-09-24)

No scientific change (M0-M5, scene selection, masking, radiometric and
geometric validation, formulas, limits, intent model untouched) and no new
provider, dependency or service. Nominatim stays the only geocoder and the
authority for every coordinate.

**Why production hit 429 - established, not guessed.** Render's outbound IPs
are "shared across all services in the same region" (Render docs) and Nominatim
limits per IP. During the 429 window this application sent a few requests a
MINUTE while the same User-Agent from another network got 200, so the shared
budget was spent by other tenants. The client then made it worse: it retried 1
s after a 429 (doubling its ask), had no cooldown (every new question probed
the limit again), cached for only 15 minutes, and a free instance loses that
cache whenever it sleeps or redeploys. One geocode per agent run
(`QueryService.build_plan`); the frontend geocodes only on the manual panel's
explicit Resolve.

**What changed** (`app/services/geospatial/nominatim.py`, public API unchanged):
- **Cache**: successes only; LRU-bounded (`geocoder_cache_entries`, 256);
  TTL 24 h (was 15 min); keyed by normalised query + service; each entry
  records query, source, monotonic and UTC resolution time.
- **Coalescing**: concurrent callers for one place await ONE in-flight future
  and share its answer OR its failure (8 callers during an outage = 1 request,
  not 8 sequences). A leader's cancellation does not cancel its waiters.
- **Pacing**: unchanged (1 request/s, one in flight); the lock is now held for
  one request, never across a backoff.
- **429**: `Retry-After` honoured (seconds or HTTP-date, capped at 1 h), else
  backoff 2 s doubling to a 300 s cap; a PROCESS-WIDE cooldown during which no
  request is sent; a caller whose wait would exceed
  `geocoder_retry_budget_seconds` (15) gets `GeocodingUnavailableError` at once
  - HTTP 503, code `geocoding_unavailable`, `Retry-After` header, message
  "Location lookup is temporarily unavailable ... Try again in about N
  seconds." A subclass of `UpstreamServiceError`, so every existing handler
  still applies.
- **Bounded retry**: at most `geocoder_max_attempts` (3) requests per geocode
  for 429/5xx/timeout/connection errors, with backoff, charged to the wait
  budget counted as PLANNED waiting (bounded however the clock behaves). No
  retry for a 4xx or a malformed answer. Upstream 503/5xx keep their existing
  502 `upstream_error` after retries.
- **Observability**: `/ready`'s geocoder detail reports upstream requests,
  429s, cache hits, coalesced callers, refusals during a cooldown, cache size
  and any active cooldown - from memory, contacting nothing.
- **Warm-up (optional)**: `SATQUERY_GEOCODER_WARM_PLACES` (";"-separated) is
  geocoded in the background after startup through the same path; a refused
  place is retried after the cooldown for at most 4 rounds; only real answers
  are cached. `render.yaml` warms the three example-chip places.
- The agent path is unchanged: an outage is still NOT a clarification (M5.5);
  the failed step and `execution.discovery_failure` now carry the plain
  "temporarily unavailable ... try again in N seconds" message; nothing is
  searched and no number is produced.

**Not added: a secondary geocoder.** `GeospatialService` calls Nominatim
directly and `ResolveResponse.source` is `Literal["nominatim", "input"]` in
both contracts; the keyless alternatives (e.g. Photon) would be a provider
abstraction plus uncertain behaviour from the same shared IP, and different
bounding boxes would silently change every AOI. The durable fix is a dedicated
outbound IP or a keyed/self-hosted geocoder.

**Tests** - `tests/test_geocoder_reliability.py` (50, fake clock that advances
instead of sleeping): success, cache hit, repeats, provenance, expiry, LRU
bound, concurrent same query (1 request, 7 coalesced), shared failure,
Retry-After seconds / HTTP-date / beyond budget / absurd / unparseable, backoff
2-4, growth 2-4-8-16 capped, no request during a cooldown, cooldown expiry,
attempt cap 1/2/3/5, budget stop, retry exhaustion, timeout, unreachable,
5 malformed answers, 6 failure kinds never yield a location, route 503 +
Retry-After, agent run searches nothing, readiness counters, warm-up (parse,
cache, throttle, bound, not-found, lifespan). Two existing policy tests moved
with the design (3 attempts, not 2; three failures to exhaust); none weakened.

**Mutation-checked (5/5, restored byte-identical):** retry storm - no cooldown
(13 fail), cache bypass (8), fabricated fallback coordinates (39), ignoring
Retry-After (12), unbounded retries (8).

**Verified locally against real Nominatim**: warm-up geocoded Marina Beach,
Cubbon Park and Ameerpet (3 requests, paced ~1 s, all 200); Cubbon Park NDVI
0.5204 from the warm cache; Dal Lake NDWI 0.03169 geocoded once, then a cache
hit on repeat; `/ready`: 4 upstream requests, 2 cache hits.

**Verified in production** (`7a2c978` on Render, real UI on Vercel in a
Playwright browser, `/ready` counters read after every run). At startup the
warm-up made 3 real requests (0 refused). Then, once each: Cubbon Park NDVI
0.5204 (S2B_43PGQ_20241208), Ameerpet NDBI -0.02482 (S2B_43QHV_20250107),
Marina Beach NDWI 0.1466 (S2B_44PMV_20250104) - all three from the warm cache;
Dal Lake NDWI 0.03169 (S2B_43SDT_20250109) and Chennai (`area_too_large`,
20.9 x 42.4 km, M1 unchanged) each geocoded once. Repeats of Dal Lake (twice,
phrased differently) and Chennai were cache hits: 8 questions + warm-up = 5
upstream requests, 0 HTTP 429, 6 cache hits. Console 0 errors / 0 warnings;
every API call 200; no CORS error. No 429 occurred during this verification, so
the throttle path is proven by the tests and mutations above, not observed live.

### Baseline - VERIFIED

| Check | Result |
| --- | --- |
| `pytest -q` | **3050 passed** (3000 before) |
| `ruff check .` / `git diff --check` | clean / clean |
| `npm run test` | **372 passed** (no frontend change) |
| `npm run lint` / `typecheck` / `build` | clean / clean / builds |

---

## 28. Pre-M6 UX hardening: a location outage is not "Insufficient evidence" - IMPLEMENTED (2026-09-24)

**Root cause.** `AgentService.answer()` special-cased only the discovery codes
`not_found` (-> `location_not_found` clarification) and `aoi_too_large`
(-> `area_too_large`). Every other discovery failure went on to synthesis,
where the evidence held no measurement, so `StandardReport` (or an AI
synthesizer) abstained with `ABSTENTION` - "Insufficient evidence to answer the
question." - and the run returned `status: "ok"` with that as its answer; the
frontend rendered it as the headline. No scene had been searched. Underneath,
geocoder 5xx / timeout / unreachable (after retries) were coded
`upstream_error`, indistinguishable from a catalog outage.

**Fix (existing error architecture, no new component):**
- `nominatim.py`: exhausted retries of 5xx / timeout / connection now raise
  `GeocodingUnavailableError` (503, `geocoding_unavailable`), like a 429 or a
  cooldown refusal; the retry hint is optional and never invented. A malformed
  answer or a 4xx stays `upstream_error` (502).
- Contract: `AgentStatus` + `location_unavailable`; `AgentFailure.stage` +
  `location`; `AgentFailure.dependency` (`"geocoder"`). Integrity: the status
  requires a `location` failure and carries no answer and no clarification.
- `AgentExecutor` keeps the cooldown / Retry-After wait
  (`discovery_failure_retry_after_seconds`, read by `isinstance` - the
  executor may not call `getattr`, a test forbids it).
- `AgentService`: `geocoding_unavailable` -> `location_unavailable` with a
  fixed, system-written message ("Location service temporarily unavailable.
  The place could not be looked up, so no scene was searched and nothing was
  measured."). Synthesis is not called. The failed step and the
  `execution.discovery_failure` evidence keep the diagnostic detail.
- Frontend: the status notice "Location service temporarily unavailable." +
  "Try again in about N seconds." (or "Try again shortly.") + a muted
  `geocoder · geocoding_unavailable`; pipeline label "Location unavailable"
  (amber); `validate.ts` accepts the status only with its failure.

**Unchanged:** A clarification (incl. `location_not_found`, `area_too_large`)
is still a question; a catalog/raster/analysis failure still surfaces through
the trace and evidence with the existing abstention; a genuine evidence verdict
still reads "Insufficient evidence to answer the question."; the successful
workflow is byte-identical. No scientific, routing, intent-model, STAC, raster
or evidence-calculation change.

**Tests:** `tests/test_geocoder_ux.py` (12, the REAL geocoder behind a scripted
transport): 429, cooldown refusal, 5xx, not "Insufficient evidence", never a
clarification, no upstream text in the user sentence, genuine abstention kept,
catalog outage not a location outage, not-found still a clarification, success
unchanged, contract integrity, the HTTP route. Updated deliberately: the
status-vocabulary pin (+`location_unavailable`), the two resolve-route tests
(503 `geocoding_unavailable` instead of 502 for timeout/5xx), three
reliability tests (message now wraps the last failure). Frontend: 5 rendering
tests + 2 validation tests. Mutations: outage falls through to synthesis (6
fail), outage turned into a clarification (7 fail); both restored.

**Verified in production** (`bfc2241` on Render and Vercel; real UI in a
Playwright browser): the live bundle carries the outage notice; Dal Lake NDWI
0.03169 (fresh geocode), Cubbon Park NDVI 0.5204 (warm cache), Dal Lake again
(cache hit); 0 HTTP 429, 0 console errors or warnings, every API call 200, no
CORS error. The outage path itself was not triggered against production on
purpose; the tests above cover it. During the first Dal Lake run the upstream
counter rose by 2 with 2 new places cached - the same question locally makes
exactly 1 request, so the other was most likely a concurrent visitor; not
attributable without Render logs.

| Check | Result |
| --- | --- |
| `pytest -q` | **3062 passed** (3050 before) |
| `ruff check .` / `git diff --check` | clean / clean |
| `npm run test` | **379 passed** (372 before) |
| `npm run lint` / `typecheck` / `build` | clean / clean / builds |

---

## 29. M6 - Product completion (workspace UX) - IMPLEMENTED (2026-09-24)

Frontend only. No backend file changed: the scientific core (M0-M5), the
intent model v2 and its thresholds, routing, STAC selection, the geocoder and
every formula are untouched (backend suite unchanged at 3062).

**Audit (before):** the answer headline was the prose sentence ("The mean NDVI
was 0.5204 index. Scene S2B_... was selected...") with no operation, place,
period or quality context; a comparison was not laid out as two periods and a
change, and the evidence panel set the EARLIER observation's mean alone under
"NDWI mean"; the pipeline showed internal tool names; the evidence was a flat
grid plus a long raw citation list, and the pixel-quality (M3), radiometric
(M4) and grid (M5) validation the API returns was never shown; area-too-large,
not-found and unsupported all read "the question needs one more detail", and a
validated refusal (scene found, index withheld by M4) read like a success;
every new question wiped the map and panels and re-showed the intro; examples
and clarification choices only filled the box; Enter did not run.

**What changed:**
- `features/agent/resultModel.ts` (new, pure): `outcomeOf` (success /
  clarification / location_not_found / location_unavailable / area_too_large /
  unsupported / insufficient_evidence / analysis_refused / provider_failure),
  `resultSummary` (index / sar / temporal / imagery / none, read by the
  backend's own evidence ids), `resultContext`, `formatPeriod` (a whole month
  as "December 2024"; nothing moved), `runStages` (the stages the RESPONSE
  shows happened, stopping at the first that did not).
- Answer panel: an outcome chip, then an operation-specific result card - index
  (title, signed mean, valid pixels, range), SAR (VV/VH/VV-VH dB), temporal
  (Earlier -> Later with period, acquisition and mean, then mean difference and
  paired-pixel change, "no cause is inferred") - then Location / Period /
  Scene / Scenes matched / Pixel quality, then the grounded sentence as the
  explanation. Distinct notices per refusal kind; a validated refusal keeps the
  server's reason under "Why".
- Pipeline: while running, one honest sentence and a live elapsed counter (one
  request, no progress events - no stage is claimed early); afterwards the
  plain-language stages (Understand question, Resolve location, Find satellite
  scenes, Validate imagery, Run analysis, Check answer) with the technical
  trace kept as a secondary row.
- Evidence: source catalog host; scenes matched with the server's selection
  rule per sensor (S2 lowest cloud cover, S1 earliest); "Quality &
  validation" (pixel quality per index/observation, radiometric state and
  baseline, grid verification); "Observations compared" with each
  observation's own mean; the citation list folded under "Technical evidence ·
  N items" (still in the DOM, still exported).
- Query: a capability line under the input; Enter runs (Shift+Enter newline;
  IME-safe); examples and clarification choices RUN the real question through
  the same request (`useAgentRun.ask`); refused while a run is in flight.
- Continuity: a new question no longer wipes the map or panels. The previous
  result stays, marked "Previous result" (dimmed; map chip "Previous result ·
  updating"), and is REPLACED, never merged, on completion; cleared on error.
  Latest-wins tickets and aborts unchanged.
- Copy fixes: the intro no longer says results arrive "as each stage returns";
  the visual panel no longer says it is "waiting for the model" in the
  standard workflow; the idle map note no longer sits under the intro card; a
  scrolling pipeline strip was squeezed to a sliver by the centre grid at
  1280x800 (now sizes to content).

**Tests:** `resultModel.test.ts` (29), `AgentPanel.m6.test.tsx` (29: NDVI,
NDWI, NDBI, SAR, temporal cards; seven distinct states plus a no-two-alike
check; refusal reason; evidence source/rule/validation/observations/technical;
Enter; running state; duplicate-submission refusal; AOI handoff; stale then
replaced; latest-wins via the hook; failure clears the stale result),
`App.test.tsx` (+1: the map keeps its place, marked previous). Fixtures in
`m6Fixtures.ts` carry the real production values. Deliberately updated (the
brief changed the behaviour): examples and clarification choices now run;
"clears the previous scene" -> "keeps the map until replaced"; temporal
headline labels; the year-boundary range now appears twice (asserted twice);
the SAR "never implies a measurement" scan excludes only the static capability
line.

| Check | Result |
| --- | --- |
| `npm run test` | **438 passed** (379 before) |
| `npm run typecheck` / `lint` / `build` | clean / clean / builds |
| `pytest -q` / `ruff` / `git diff --check` | 3062 passed / clean / clean |
