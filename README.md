# SatQuery AI

Ask a question about satellite imagery in plain language and get an answer that
is traceable to the pixels it came from.



---

## What it does

You type: *"Is there visible water in the Sentinel-2 image of Marina Beach,
Chennai?"*

The system geocodes the place, searches the Sentinel-2 catalog, deterministically
selects a scene, reads a bounded window of the real raster, computes a spectral
index, asks a vision-language model to describe the picture, and writes an answer
— then **mechanically checks that every number in that answer traces back to
something it actually measured** before showing it to you.

If the check fails, the answer is withheld and the evidence is shown instead.

## The idea that shapes everything

> **The model chooses what to run. It never computes the result.**

A language model selects which of the deterministic analyses to execute, and
describes the findings afterwards. It does not calculate an index, position a
raster, or decide a measurement. Those come from the raster and the catalog.

This distinction is enforced in code, not by convention:

- **Blue** in the interface means a value computed from pixels or STAC metadata.
- **Violet** means a model said it — qualitative interpretation, never a measurement.
- A model-sourced statement **cannot authorise a number**, even one the model
  writes inside its own sentence (`grounding._numeric_authorities`).
- Every visual observation is attributed to the exact provider and model that
  produced it. An observation whose author cannot be named is not publishable.

---

## Architecture

```
Natural-language question
        │
        ▼
  Agent orchestration ──────────► AI provider layer
        │                              │
        │                    ┌─────────┴─────────┐
        │                 Gemini              NVIDIA
        │                    │                   │
        │              planner · visual · synthesizer
        │
        ▼
  Deterministic pipeline  (identical under either provider)
        │
        ├─ Geocoding ................ Nominatim
        ├─ Scene discovery .......... Earth Search STAC
        ├─ Scene selection .......... deterministic, not model-chosen
        ├─ Imagery retrieval ........ windowed COG reads → PNG
        ├─ Quantitative bands ....... raw uint16, never the display path
        ├─ NDWI / temporal NDWI ..... pure functions
        └─ Georeferencing ........... source affine → WGS 84 corners
        │
        ▼
  Grounding ── numbers traced · citations resolved · terminology checked
        │
        ▼
  Answer + evidence + attributed observation
```

**Provider selection never reaches the deterministic half.** For the same query
and scene, Gemini and NVIDIA produce byte-identical evidence; only the prose and
the visual observation differ. There is a regression test that asserts exactly
this.

---

## AI providers

Two interchangeable backends behind one abstraction. All three AI roles —
planning, visual analysis, answer synthesis — come from the **same** provider.

```bash
AI_PROVIDER=gemini     # GEMINI_API_KEY, GEMINI_MODEL
AI_PROVIDER=nvidia     # NVIDIA_API_KEY, NVIDIA_BASE_URL, NVIDIA_MODEL
```

A single run can override the default without a restart:

```json
{ "question": "...", "provider": "nvidia",
  "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning" }
```

### Model catalog

Models carry capability metadata, because "which provider" does not determine
whether a run can happen — the visual step sends an image, and most hosted
models cannot accept one.

| Model | Image input | Role |
|---|---|---|
| `gemini-3.6-flash` | yes | all |
| `nvidia/nemotron-nano-12b-v2-vl` | yes | all |
| `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | yes | all |
| `meta/llama-3.2-11b-vision-instruct` | yes | all |
| `nvidia/nemotron-3-super-120b-a12b` | **no** | text only |
| `nvidia/nemotron-3.5-lightning-30b-a3b` | **no** | text only |

A text-only model is refused for visual analysis **before** any request is made —
no image is ever sent to a model that cannot see it. `GET /api/v1/ai/models`
reports what a deployment can actually offer.

**No silent fallback.** A missing credential fails clearly rather than answering
as the other provider; an unattributable result would make any evaluation
meaningless. Capability metadata is curated from vendor documentation, because
the OpenAI-compatible `/v1/models` endpoint lists ids but not modalities. Hosted
availability and free-tier limits change over time.

---

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness and build metadata |
| `GET` | `/api/v1/ai/models` | Model catalog with capabilities and status |
| `POST` | `/api/v1/query/agent` | **Ask a question** — plan, execute, ground, answer |
| `POST` | `/api/v1/query/parse` | Text → structured intent |
| `POST` | `/api/v1/query/build-plan` | Intent → resolved plan with AOI |
| `POST` | `/api/v1/query/execute` | Run discovery, selection, optional imagery |
| `POST` | `/api/v1/query/analyze` | NDWI / temporal NDWI over a result |
| `POST` | `/api/v1/geospatial/resolve` | Place name → bounding box |
| `POST` | `/api/v1/satellite/search` | STAC scene discovery |
| `POST` | `/api/v1/satellite/imagery` | Bounded windowed raster → PNG |

### Failure is a first-class outcome

Every agent outcome is **HTTP 200**, including the failures:

| Status | Meaning | What survives |
|---|---|---|
| `ok` | Grounded answer produced | everything |
| `planner_unavailable` | No plan; nothing ran | nothing is claimed |
| `synthesis_unavailable` | Tools ran, prose failed | the evidence |
| `answer_withheld` | Answer failed validation | the evidence and the checks |

Converting these to a 5xx would discard the useful half of the response. The
measurements are the product; the sentence is a presentation of them.

---

## Interface

A three-column geospatial workstation, not a chat window.

```
┌──────────────┬────────────────────────────┬──────────────────┐
│ CONFIGURATION│ NATURAL-LANGUAGE QUERY     │ FOOTPRINT        │
│              ├────────────────────────────┤                  │
│ Location     │ PIPELINE                   │ ANALYSIS RESULT  │
│ Date window  ├────────────────────────────┤                  │
│ Sensor       │                            │ VISUAL           │
│ Analysis     │   SATELLITE IMAGERY        │ OBSERVATION      │
│ Scene        │   (the visual hero)        │ (violet)         │
│ candidates   ├────────────────────────────┤                  │
│              │ DETERMINISTIC EVIDENCE     │                  │
└──────────────┴────────────────────────────┴──────────────────┘
```

Real Sentinel imagery is positioned by its **four source-derived corners**, not
an axis-aligned box — a reprojected UTM window is a quadrilateral in WGS 84, and
treating it as a rectangle misplaces it by ~144 m over a city-sized AOI.

Empty states are honest: the interface says *"No scene loaded"* or *"Imagery not
retrieved"* rather than showing a plausible placeholder. Nothing on screen is
fabricated to make a screenshot look better.

---

## Running it

```bash
# Backend
cd backend
cp .env.example .env          # add your keys; .env is git-ignored
uv sync
uv run uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev                   # http://localhost:5173
```

The dev server must run on port **5173** — the backend CORS allowlist permits
that origin. If the backend binds IPv4-only, point the frontend at
`VITE_API_BASE_URL=http://127.0.0.1:8000`, since browsers resolve `localhost`
to `::1` first.

### Checks

```bash
cd backend  && uv run pytest -q && uv run ruff check .
cd frontend && npm test -- --run && npx tsc -b --noEmit && npm run lint && npm run build
```

**1321 backend tests · 209 frontend tests.** Every phase was written test-first:
the tests were added, observed failing for the expected reason, and only then
satisfied.

---

## Stack

**Backend** — Python 3.12, FastAPI, Pydantic v2, rasterio, httpx, google-genai, `uv`
**Frontend** — React 19, TypeScript, Vite, MapLibre GL, Vitest
**Data** — Earth Search STAC (Sentinel-2 L2A, Sentinel-1 GRD), OpenStreetMap Nominatim

---

## Honest limitations

Stated plainly, because a system that reports what it cannot do is more useful
than one that implies it can do everything.

- **NDWI is a spectral index, not a water classifier.** A threshold is reported
  as *"% of valid pixels with NDWI > 0.3"*, never as detected water. No
  validated water or flood classification exists here.
- **Temporal NDWI compares two aggregate statistics**, each over a different set
  of pixels. It is not per-pixel change detection, and the difference is
  suppressed entirely when that framing would mislead.
- **No co-registration or resampling.** Change detection across unaligned grids
  is refused rather than approximated.
- **Absolute surface reflectance is not established.** NDWI is computed on raw
  DN, where the scale cancels identically in a normalised difference.
- **Grounding is containment, not proof.** It establishes that numbers trace to
  evidence and citations resolve. It cannot establish that a qualitative claim
  is correct — an unquantified sentence passes.
- **Sentinel-1 VV is display-only** — percentile-stretched for visualisation, not
  calibrated. No speckle filtering or terrain correction.
- **`/query/parse` is Gemini-only.** It predates the provider abstraction and is
  not routed through it; the agent path is fully provider-agnostic.
- Cloud masking (`scl`) is deferred; cloud cover is reported as context.

---

## Repository

```
backend/
  app/
    api/routes/        health · geospatial · satellite · query · ai
    services/
      geospatial/      place → bounding box
      satellite/       STAC discovery · windowed raster reads
      analysis/        NDWI engines, temporal statistics
      query/           orchestration, observations, compatibility
      agent/           planner · executor · grounding · synthesizer
        providers/     gemini · nvidia · catalog · factory
  tests/
frontend/
  src/
    api/               typed client mirroring backend contracts
    features/
      agent/           query, pipeline, evidence, answer, observation
      map/             MapLibre viewport and footprint locator
      query/           configuration rail
```

`CLAUDE.md` holds the authoritative phase-by-phase record, including the
boundaries each phase deliberately did not cross and the architectural traps
that were verified against live data.
