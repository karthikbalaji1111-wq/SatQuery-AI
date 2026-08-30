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
8. Sentinel-1 VV imagery retrieval for visualization: the existing
   ImageryService / raster path now accepts a single-band Float32 VV asset,
   applies a 2nd-98th percentile display clip -> min-max to 8-bit grayscale ->
   3 identical bands -> PNG, and returns the existing ImageryResponse. Display
   only, NOT calibrated. SAR scientific processing (speckle filtering,
   calibration, terrain correction, fusion, analysis) remains out of scope.

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

Current HEAD represents the completed Temporal Observation Model phase.

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
- HEAD: `6036768` — `feat(analysis): add single-scene Sentinel-2 NDWI`
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
- `MapPanel` is a placeholder; there is no MapLibre dependency.

## 4. Test / regression baseline — VERIFIED

Any Phase 11 work must keep these green and must not reduce them:

| Check | Baseline at `43f06ee` |
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
| `scl` | SCL | 20 m | `uint8` | `0` | **deferred** for the first 10 m NDWI path |
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
   real coverage is larger. Measured: ~2 px / **~21 m** systematic offset, and
   worse/asymmetric when a window clamps at a scene edge. Georeferencing a
   detection through it is silently wrong.
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
idempotent, and an NDWI needs only two reads.

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
did not perform.