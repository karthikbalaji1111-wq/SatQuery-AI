# SatQuery AI

Ask a question about satellite imagery in plain language and get an answer that
is traceable to the pixels it came from.

Built for **Smart India Hackathon 2026, Problem Statement 26167**.

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
        ├─ NDVI · NDWI · NDBI ....... pure functions, raw DN
        ├─ Temporal NDWI ............ two dates, exact-grid change
        └─ Georeferencing ........... source affine → WGS 84 corners
        │
        ▼
  Grounding ── numbers traced · citations resolved · terminology checked
        │
        ▼
  Answer + evidence + attributed observation
```

**Provider selection never reaches the deterministic half.** The measurements
are computed by the same code from the same pixels whichever provider planned
the run: **given equivalent validated plans** — the same tools over the same
intent and parameters — Gemini and NVIDIA produce identical evidence, and only
the prose and the visual observation differ. The condition matters, and it is
the plan rather than the question: two providers handed the same question may
still validate to different plans (a different index, a different window), and
then the evidence legitimately differs because a different analysis was asked
for. A regression test asserts the provider cannot reach the deterministic
half; it does not assert that two models always plan alike.

---

## AI providers

Four interchangeable backends behind one abstraction — three cloud APIs and
one open-weight model running on this machine. All three AI roles — planning,
visual analysis, answer synthesis — come from the **same** provider.

```bash
AI_PROVIDER=gemini     # GEMINI_API_KEY, GEMINI_MODEL
AI_PROVIDER=nvidia     # NVIDIA_API_KEY, NVIDIA_BASE_URL, NVIDIA_MODEL
AI_PROVIDER=anthropic  # ANTHROPIC_API_KEY, ANTHROPIC_MODEL
AI_PROVIDER=local      # LOCAL_AI_BASE_URL, LOCAL_AI_MODEL - no key, no quota
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
| `nvidia/nemotron-nano-12b-v2-vl` | yes | **retired by NVIDIA on 2026-08-26** (the endpoint answers 410 Gone) |
| `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | yes | all |
| `meta/llama-3.2-11b-vision-instruct` | yes | all |
| `nvidia/nemotron-3-super-120b-a12b` | **no** | text only |
| `nvidia/nemotron-3.5-lightning-30b-a3b` | **no** | text only |
| `qwen3-vl:4b-instruct` (local, default) | yes | all |
| `qwen3-vl:2b-instruct` / `8b-instruct` / `30b-a3b-instruct` (local) | yes | all |

A text-only model is refused for visual analysis **before** any request is made —
no image is ever sent to a model that cannot see it. `GET /api/v1/ai/models`
reports what a deployment can actually offer.

**No silent fallback.** A missing credential fails clearly rather than answering
as the other provider; an unattributable result would make any evaluation
meaningless. Capability metadata is curated from vendor documentation, because
the OpenAI-compatible `/v1/models` endpoint lists ids but not modalities. Hosted
availability and free-tier limits change over time.

### Hosted-model reliability: what the server guarantees

Hosted models are the least reliable part of the system, so the server does not
depend on them for anything it can establish itself. Each rule below was added
after a failure observed against the live endpoints:

- **NVIDIA plans through a forced tool call.** Asking for "some JSON object" let
  the configured model return tool *names* as bare strings, with no parameters.
  Planning now forces one tool call whose parameter schema is `AgentPlan`'s own
  JSON Schema. `AgentPlan` still validates what comes back, and a model that
  ignores `tools` falls back to the plain-JSON path.
- **An explicitly requested step always runs.** Planners sometimes return a
  valid plan that omits the analysis the question asked for. The server then
  adds exactly that step: the named index when a single-window Sentinel-2
  question says NDVI, NDWI or NDBI; the visual observation when it asks what is
  *visible*; temporal NDWI when a two-window Sentinel-2 comparison asks about
  water. It never adds a measurement, never changes where or when, and never
  drops a step the planner chose. `trace.plan` keeps the planner's plan;
  `trace.steps` shows what ran.
- **The synthesiser sees provenance and readable numbers.** Evidence is shown to
  it with the queried location, the requested window, the selected scene and its
  acquisition date, and with values rounded to four significant figures for
  reading. Stored evidence, the API response and the export keep full
  precision, and every displayed value is one grounding accepts.
- **Transient failures are retried, boundedly.** NVIDIA synthesis and visual
  calls make up to three attempts on a 429 or 5xx; a STAC search makes a second
  attempt after a transport error, 429 or 5xx. A rejected request or a malformed
  response is never retried.
- **Free tiers are the practical limit.** During verification Gemini's free tier
  answered 429 on first attempts for long stretches, and NVIDIA's hosted
  endpoint answered roughly half of sampled requests with 503 (capacity) or
  timed out. Both surface as `planner_unavailable` / `synthesis_unavailable`
  with the evidence preserved; neither is hidden by switching provider.

### Local provider: Ollama + Qwen3-VL (no key, no quota)

`AI_PROVIDER=local` routes planning, visual analysis and synthesis to an
open-weight Qwen3-VL model served by [Ollama](https://ollama.com) on the same
machine. Requests go to `LOCAL_AI_BASE_URL` and nowhere else. There is **no
fallback**: if Ollama is down or the model is missing, the run fails and says
so — it is never quietly answered by Gemini or NVIDIA.

**1. Install and start Ollama** (macOS):

```bash
brew install ollama        # or install the app from https://ollama.com/download
ollama serve               # or simply open the Ollama app
curl http://127.0.0.1:11434/api/version    # must answer before SatQuery can use it
```

**2. Install a model** — an `-instruct` tag:

```bash
ollama pull qwen3-vl:4b-instruct
ollama run qwen3-vl:4b-instruct "Reply with one word: ready"
```

Use an `-instruct` tag. Every plain `qwen3-vl` tag on the Ollama registry
(`qwen3-vl:4b`, `:8b`, …) is byte-identical to its `-thinking` variant, which
reasons in hidden tokens whatever you ask: verified live, a one-word reply cost
220 generated tokens and a real planning call did not finish in 300 s.

| Unified memory | Model | Download |
|---|---|---|
| 8 GB | `qwen3-vl:4b-instruct` (default). `qwen3-vl:2b-instruct` fits in 3 GB but failed most SatQuery tasks in the bake-off below. **Do not use `8b-instruct` here** — measured unusable, see below | 3.3 / 1.9 GB |
| 16 GB | `qwen3-vl:8b-instruct` — **untested at this tier** | 6.1 GB |
| 32 GB+ | `qwen3-vl:30b-a3b-instruct` — never installed, download abandoned | 19.6 GB |

**What has actually been measured:** `2b-instruct` and `4b-instruct` carry real
numbers, each over two full passes of the same seven tasks (below).

`8b-instruct` was benchmarked on an **8 GB** machine and **failed every task**:
14 of 14 E2E attempts returned `planner_unavailable` after exceeding the 300 s
timeout, with zero successful local calls. It needs 7.63 GB resident on an 8 GB
machine, driving free memory to 2% and swap to a 14.6 GB peak. That result
disproves it *at 8 GB only* — the 16 GB tier above is untested, not refuted.

`30b-a3b-instruct` is **not installed**; the download was abandoned after three
attempts, the last two killed at 15 GB of 19 GB. No figure for it exists
anywhere in this repository.

**3. Run SatQuery on it** — either for every run:

```bash
cd backend && AI_PROVIDER=local uv run uvicorn app.main:app --reload --port 8000
```

(or set `AI_PROVIDER=local` in `backend/.env`), or keep any default and choose
**Local · Qwen3-VL 4B Instruct** in the workspace's model selector for a single
run. Switching back is the same selector, or `AI_PROVIDER=gemini` / `nvidia`.
The selector reports each local model as *Ready*, *Not installed*, *Ollama not
running*, or *Ollama cannot read models* (Ollama answers but cannot list its
models — for example, the drive holding them is disconnected), from one
read-only `GET /api/tags`.

| Variable | Default | Purpose |
|---|---|---|
| `AI_PROVIDER` | `gemini` | `local` selects this provider |
| `LOCAL_AI_BASE_URL` | `http://127.0.0.1:11434` | where Ollama listens |
| `LOCAL_AI_MODEL` | `qwen3-vl:4b-instruct` | the installed tag to use |
| `SATQUERY_LOCAL_AI_TIMEOUT_SECONDS` | `300` | the first request loads the model |
| `SATQUERY_LOCAL_AI_NUM_CTX` | `4096` | context window. The largest prompt measured is about 1.7–2.2k tokens; 3072 would save only 0.16 GB of 4.64 GB resident, and 2048 cannot hold it |

**Measured on the reference machine** — Apple M2 MacBook Air, 8 GB unified
memory, `qwen3-vl:4b-instruct`, Ollama 0.20.2, models on a USB hard disk, with
the dev servers and a browser open (bake-off, 2026-09-16; two passes of seven
real Sentinel-2 / Sentinel-1 tasks):

| | |
|---|---|
| Model load (cold, from the USB disk) | about 41 s; first request about 63 s |
| Resident memory | 4.64 GB at `num_ctx` 4096 |
| Planning call (≈1.3k prompt tokens) | 10–22 s (median 12 s) |
| Observation of a real 112×300 scene | 4–6 s |
| Synthesis call | 5–14 s (median 6 s) |
| End-to-end NDVI, NDWI, NDBI, visual, temporal, refusal | 17–48 s (median 22–25 s) |
| Free memory while it ran | 8–13%; swap grew about 1 GB over the session |

On an 8 GB machine the latency is set by memory pressure, not by the model: in
an earlier session with 10 GB of swap in use the same UI queries took 94–131 s.
Close other applications before a demo. In this session NVIDIA answered a
grounded NDVI in 31 s (after one transient 503) and Gemini's free tier refused
with 429 — the dependence on outside quotas that local inference removes.

**Limitations of the local model.** A 4B model is weaker than the cloud ones:
it may choose a neighbouring analysis (NDWI statistics for a "visible water"
question — plan completion then adds the observation the question asked for),
it abstains more readily, its image descriptions are general, and it was not
trained on remote sensing. The model selector's status is the last catalog
read; it refreshes when the window regains focus.

**What the local model does — and does not do.** It plans, describes the
retrieved image and writes the answer. It is never the authority for a number:
NDVI/NDWI/NDBI values, pixel counts, temporal differences, coordinates, scene
dates and SAR statistics all come from the deterministic pipeline. Its plan is
decoded under a grammar derived from `AgentPlan`'s own JSON Schema — a required
discovery step, then up to two analysis steps — and then validated by
`AgentPlan`, so it can only choose among the existing tools: no shell, URL, file
or query of its own. Its answer passes the same grounding as a cloud model's: an
invented number is withheld. The image it sees is the exact PNG the pipeline
already retrieved; nothing is re-fetched for it.

**Smoke test** (live, not part of CI — needs Ollama and network for one scene):

```bash
cd backend && uv run python scripts/local_model_smoke.py [--model qwen3-vl:2b-instruct]
```

**Troubleshooting**

| You see | Fix |
|---|---|
| *Local AI provider is unavailable. Start Ollama…* / selector says *Ollama not running* | start Ollama; check `curl $LOCAL_AI_BASE_URL/api/version` |
| *The local model '…' is not installed* / selector says *Not installed* | `ollama pull <that tag>` |
| *…does not have enough free memory…* | close other apps; `qwen3-vl:2b-instruct` needs less memory but is much weaker at SatQuery's tasks |
| *…did not answer within 300 seconds* | the first load is slow; retry, raise `SATQUERY_LOCAL_AI_TIMEOUT_SECONDS`, or use a smaller model — and make sure the tag is `-instruct` |
| Selector says *Ollama cannot read models*, or a query says *Ollama is running but cannot read its installed models…* | the drive holding the models (below) is disconnected or has dropped off USB: reconnect it — Ollama picks it up again without a restart (verified live). Ollama does not start at all while that drive is missing, so if it was opened then, reconnect and reopen Ollama |
| Disk full | each model is 1.9–19.6 GB; `ollama list`, then `ollama rm <tag>` for ones you no longer use, or keep the models on an external disk (below) |

**Keeping models on an external disk.** Ollama stores models in `~/.ollama/models`.
To keep them on an external drive instead, quit Ollama, then:

```bash
mkdir /Volumes/<Drive>/ollama
cp -RX ~/.ollama/models /Volumes/<Drive>/ollama/models   # -X: no ._ files on exFAT
mv ~/.ollama/models ~/.ollama/models.internal-backup
ln -s /Volumes/<Drive>/ollama/models ~/.ollama/models
```

Reopen Ollama, check `ollama list`, run one query, then delete the backup to free
the space. Only loading gets slower: on a USB hard disk a cold load of
`qwen3-vl:4b-instruct` took 55.5 s against 19.9 s from the internal SSD, and once the
model is resident answers are unchanged. Eject the drive before unplugging it.

What an external hard disk changes, measured on the reference machine (a USB
hard disk, exFAT):

- **Cold loads.** A cold load of `qwen3-vl:4b-instruct` took 38–55 s from the
  drive against about 16–20 s from the internal SSD. Ollama unloads a model
  after 5 idle minutes by default, so the first query after a pause pays that
  again. For a demo session, keep it loaded longer with
  `launchctl setenv OLLAMA_KEEP_ALIVE 30m` and reopen Ollama — that holds
  4.6 GB of an 8 GB machine for as long as it lasts;
  `launchctl unsetenv OLLAMA_KEEP_ALIVE` undoes it.
- **Drive sleep.** macOS spins the disk down after 10 idle minutes (`pmset -g`
  shows `disksleep 10`). Waking it takes about 4 s, which the model catalog
  allows for. `sudo pmset -a disksleep 0` keeps it spinning — a system setting
  to change only if you want that trade.
- **Dropouts.** During testing the drive dropped off USB once (it came back as a
  different disk device). Every local request then failed within milliseconds
  and SatQuery said so (*Ollama cannot read models*) — no cloud fallback, and
  the deterministic evidence was kept. Once the drive was back, Ollama recovered
  without a restart. For a demo, connect the drive directly rather than through
  a hub, or keep the models on the internal disk.

---

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | **Liveness** — is this process alive? Answers from configuration alone |
| `GET` | `/ready` | **Readiness** — can this deployment actually work? 503 when not, with the reason |
| `GET` | `/api/v1/ai/models` | Model catalog with capabilities and status |
| `POST` | `/api/v1/query/agent` | **Ask a question** — plan, execute, ground, answer |
| `POST` | `/api/v1/query/parse` | Text → structured intent |
| `POST` | `/api/v1/query/build-plan` | Intent → resolved plan with AOI |
| `POST` | `/api/v1/query/execute` | Run discovery, selection, optional imagery |
| `POST` | `/api/v1/query/analyze` | Spectral indices / temporal NDWI over a result |
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

A failure inside a tool is explained too: when scene discovery, an analysis or
the visual observation fails (a catalog timeout, say), the evidence carries one
item stating why, so the answer can report what happened instead of
"insufficient evidence".

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
cd frontend && npm test -- --run && npx tsc -b --noEmit && npm run lint && VITE_API_BASE_URL=https://api.satquery.example.com npm run build
```

**2277 backend tests · 307 frontend tests.** Every phase was written test-first:
the tests were added, observed failing for the expected reason, and only then
satisfied.

The backend suite is **independent of the machine it runs on** — it clears every
provider variable and does not read a developer's `.env`, so a credentialed
laptop and a fresh checkout run the same suite. (Before that: 2112 passed with a
key present, 2108 passed and 4 failed without one.)

### Production

```bash
export GEMINI_API_KEY=...        # whatever this deployment holds
export SATQUERY_CORS_ORIGINS=https://satquery.example.com
export SATQUERY_PUBLIC_API_URL=https://api.satquery.example.com
docker compose -f docker-compose.prod.yml up --build -d
```

A separate stack from the development one: no `--reload`, no source mount,
locked dependencies, non-root, a built frontend served statically, and host
ports bound only to loopback. Configure a host TLS reverse proxy for
the two HTTPS domains above before external access. Production settings and
frontend builds reject missing or unsafe public origins. This is not yet an
authenticated public service; keep access restricted pending later milestones.
See
[`DEPLOYMENT.md`](DEPLOYMENT.md) for the resource limits, the liveness/readiness
split, and the one limitation that matters — every limit is **per process**, so
workers and replicas multiply them.

What the system will and will not measure is stated in
[`SUPPORTED_DATA.md`](SUPPORTED_DATA.md): supported, conditionally supported
(with the condition), and unsupported (with the reason).

---

## Stack

**Backend** — Python 3.12, FastAPI, Pydantic v2, rasterio, httpx, google-genai, anthropic, `uv`
**Frontend** — React 19, TypeScript, Vite, MapLibre GL, Vitest
**Data** — Earth Search STAC (Sentinel-2 L2A), Microsoft Planetary Computer STAC (Sentinel-1 RTC), OpenStreetMap Nominatim

---

## Honest limitations

Stated plainly, because a system that reports what it cannot do is more useful
than one that implies it can do everything.

- **These are spectral indices, not classifiers.** NDVI, NDWI and NDBI are
  normalised differences over Sentinel-2 bands. A high NDBI is a built-up-like
  *reflectance signature*, not a detected building; a high NDVI is not verified
  healthy vegetation. Nothing here performs land-cover classification.
- **NDBI resolves no finer than 20 m.** SWIR (B11) is a 20 m band against NIR's
  10 m. The two are placed on the 10 m grid by explicit whole-cell assignment -
  each 20 m value is used by the four 10 m pixels it contains, which invents no
  values because the grids are exactly 2:1 nested with a shared origin (verified
  from the COG headers). The arithmetic runs at 10 m; the detail is still 20 m,
  and that is reported with the result.
- **NDWI is a spectral index, not a water classifier.** A threshold is reported
  as *"% of valid pixels with NDWI > 0.3"*, never as detected water. No
  validated water or flood classification exists here.
- **Temporal NDWI compares two aggregate statistics**, each over a different set
  of pixels. It is not per-pixel change detection, and the difference is
  suppressed entirely when that framing would mislead.
- **No temporal co-registration or resampling.** A paired-pixel comparison of
  two dates runs only when both reads land on an identical grid; across
  unaligned grids it is refused rather than approximated. (NDBI's 20 m → 10 m
  whole-cell assignment, above, is the only regridding anywhere, and it invents
  no values.)
- **Absolute surface reflectance is not established.** The indices are computed
  on raw DN. A common multiplicative scale cancels identically in a normalised
  difference; an additive offset does not, and the advertised offset is not
  applied because it does not describe these pixels (measured, not assumed).
  The result is proportional to reflectance, which a ratio needs and an absolute
  measurement would not.
- **Temporal NDWI difference is implemented; general change detection is not.**
  Two Sentinel-2 acquisitions are indexed independently, paired chronologically,
  and reported side by side with a mean difference. Where the two reads land on
  an identical grid (same CRS, dimensions and affine — checked exactly, with no
  tolerance) a paired-pixel difference and a **georeferenced difference overlay**
  are produced. That is a difference between two dated observations of one
  index. It is **not** general-purpose change detection: nothing is classified,
  no change type is inferred, and no threshold declares a pixel "changed".
- **Grounding is containment, not proof.** It establishes that every number
  traces to evidence — at the stated precision, with a matching unit stated on
  either side of the value — and that citations resolve. It cannot establish
  that a *qualitative* claim is true: it checks support, not correctness. In
  practice it is conservative, and a qualitative sentence must repeat a
  citation to survive, so an arbitrary unsupported sentence does **not** simply
  pass. What remains outside its reach is a supported-but-wrong interpretation.
- **Sentinel-1: RTC imagery AND quantitative backscatter.**
  Discovery, deterministic selection and bounded **VV and VH** retrieval all run
  against Microsoft Planetary Computer's Sentinel-1 RTC collection, which
  publishes analysis-ready, **provider terrain-corrected** gamma-naught COGs.
  Assets are not anonymously readable (`409 PublicAccessNotPermitted`); the
  public `/api/sas/v1/sign` endpoint signs them with no credentials, so this
  deployment stores no secret for it. The rendered PNG is
  `10·log10` → 2nd–98th percentile clip → 8-bit grayscale: **display only**, a
  monotonic transform that never touches a quantitative path.
  **Quantitative backscatter is implemented.** The `vv`/`vh` assets are linear
  gamma-naught **power** (`float32`, `nodata -32768`, no scale/offset — proven
  from the product, not assumed), so statistics are computed as
  `dB = 10·log10(power)`. Means are averaged **in linear power and then
  converted** — `10·log10(mean(power))`, never `mean(10·log10(power))`, which
  would give a geometric mean, biased low and not mean backscatter. Reported per
  polarization: mean/min/max dB, valid pixel count, plus `VV−VH` on common valid
  pixels. Validated against an independent NumPy computation on the same window
  to **0.000e+00**.

  **What is still NOT implemented:** radiometric calibration from raw GRD,
  speckle filtering, custom terrain correction, layover/shadow masking,
  polarimetric decomposition, SAR classification and optical-SAR fusion. The
  terrain correction is the **provider's**, not ours — never describe it as
  SatQuery calibration. The product exposes no quality mask beyond `nodata`, so
  no quality score is reported, and `VV−VH` is a difference of measurements, not
  a land-cover classification.
- **Sentinel-1 GRD via Earth Search remains unreachable**, and is refused with a
  clear error naming the cause: the measurement asset is an `s3://` URI on a
  requester-pays bucket, the GRD product is in radar geometry (`crs=None`, 210
  GCPs) rather than a map projection, and the pixels are uncalibrated `uint16`
  DN whose calibration LUTs live in separate XML assets. The RTC route above is
  the one that works.
  The GRD display code path is retained and tested; it is correct for a
  projected single-band asset and simply has no such GRD asset to read. Nothing
  quantitative reads GRD - backscatter statistics come only from the RTC route
  above.
- **`/query/parse` follows `AI_PROVIDER`.** It resolves its parser through the
  same provider factory as the agent path, so the configured provider governs
  it too.
- **No cloud masking.** Scene selection is deterministic: Sentinel-2 takes the
  lowest reported scene cloud cover, then the earliest acquisition, then the
  scene id; Sentinel-1 takes the earliest acquisition. Scene cloud cover is
  catalog metadata for the whole tile, not a mask over the area analysed, and no
  per-pixel cloud or shadow mask (`scl`) is applied, so a cloudy pixel inside
  the AOI is measured like any other. Cloud cover is reported as context.

---

## Repository

```
backend/
  app/
    api/routes/        health · geospatial · satellite · query · ai
    services/
      geospatial/      place → bounding box
      satellite/       STAC discovery · windowed raster reads
      analysis/        spectral index engines, temporal statistics
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
