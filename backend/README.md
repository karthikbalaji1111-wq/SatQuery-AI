# SatQuery Backend

Python 3.12 + FastAPI. Managed with [uv](https://docs.astral.sh/uv/).

## Setup

```bash
uv sync
cp .env.example .env     # add credentials; .env is git-ignored
```

## Run

```bash
uv run uvicorn app.main:app --reload --port 8000
```

| | |
|---|---|
| Liveness | <http://localhost:8000/health> |
| Readiness | <http://localhost:8000/ready> |
| API docs | <http://localhost:8000/docs> |

For the production image (no reload, locked dependencies, non-root, static
frontend) see [`../DEPLOYMENT.md`](../DEPLOYMENT.md).

## Checks

```bash
uv run pytest -q
uv run ruff check .
```

`ruff format` is **not** part of the gate. The tree predates it and 82 of 124
files would be reformatted; doing that in the same change as behavioural work
would bury the diff. It is a deliberate, separate piece of work.

## Layout

| Path | Responsibility |
| --- | --- |
| `app/main.py` | App factory: middleware order, exception handlers, router |
| `app/api/router.py` | Router assembled **per application** from its own settings |
| `app/api/routes/` | `health` (+`/ready`) · `geospatial` · `satellite` · `query` · `ai` |
| `app/core/config.py` | Environment configuration (`pydantic-settings`) |
| `app/core/errors.py` | Error types, one JSON envelope, `Retry-After` handling |
| `app/core/limits.py` | Admission control: rate limit, workflow slots, raster gate, body size |
| `app/core/observability.py` | Run ids (`ContextVar`) and stage timing |
| `app/core/logging.py` | Logging setup; stamps every line with its run id |
| `app/services/geospatial/` | Place → bounding box; the application-wide Nominatim budget |
| `app/services/satellite/` | STAC discovery, windowed raster reads, asset-host policy |
| `app/services/analysis/` | Spectral index engines, temporal statistics, SAR backscatter |
| `app/services/query/` | Orchestration, observations, compatibility, execution integrity |
| `app/services/agent/` | Planner · executor · grounding · synthesizer |
| `app/services/agent/providers/` | `gemini` · `nvidia` · `anthropic` · `local` · catalog · factory |

Every SDK is confined to its own provider module; AST tests assert the importer
list. `services/query` must never import `services/analysis`.

## What the boundaries guarantee

- **A client-supplied execution result is checked for internal consistency**
  before it is analysed (`services/query/integrity.py`). Pydantic proves the
  shape; that module proves the parts agree with each other and with the intent
  they claim to answer — a selected scene really is among the scenes returned, a
  window's period really was requested. Structural validity is not provenance
  integrity.
- **Expensive routes are bounded** (`core/limits.py`): a per-client rate limit
  (429 with `Retry-After`), a cap on simultaneous workflows (503), a body-size
  refusal (413) and a raster-read gate. All per process — see `DEPLOYMENT.md`
  for exactly what that does and does not guarantee.
- **The geocoder has one application-wide budget** (`geospatial/nominatim.py`):
  one request at a time, spaced by `SATQUERY_GEOCODER_MIN_INTERVAL_SECONDS`, a
  TTL cache for repeats, and one bounded retry. Not per client — "one per second
  per user" would let ten users send ten per second under this application's
  single User-Agent.
- **Asset hrefs come from an external catalog**, so a non-public address is
  never opened on its behalf, and `SATQUERY_TRUSTED_ASSET_HOSTS` can narrow
  reads to named hosts.

## AI providers

The agent's three roles — planning, visual analysis, answer synthesis — all come
from one provider, selected by configuration:

```bash
AI_PROVIDER=gemini    # Google Gemini      (GEMINI_API_KEY, GEMINI_MODEL)
AI_PROVIDER=nvidia    # NVIDIA-hosted NIM  (NVIDIA_API_KEY, NVIDIA_BASE_URL, NVIDIA_MODEL)
AI_PROVIDER=anthropic # Anthropic Claude   (ANTHROPIC_API_KEY, ANTHROPIC_MODEL)
AI_PROVIDER=local     # Ollama + Qwen3-VL  (LOCAL_AI_BASE_URL, LOCAL_AI_MODEL; no key)
```

A single run may override the default:

```bash
curl -X POST localhost:8000/api/v1/query/agent \
  -H 'Content-Type: application/json' \
  -d '{"question": "...", "provider": "nvidia"}'
```

**Only the inference backend changes.** Geospatial grounding, STAC discovery,
retrieval, the raster path, the indices, temporal analysis, the grounding checks
and the evidence contract are identical whichever provider is selected, so the
deterministic results for a given scene and query do not depend on it. Only the
model-generated interpretation differs, and it is attributed to the provider and
model that produced it.

Notes:

- **A visual model must accept image input.** A text-only model is refused
  *before* a request is made rather than being asked to describe a picture it
  never received. `GET /api/v1/ai/models?role=visual` reports what this
  deployment can actually offer, and whether each model is catalogued,
  configured and compatible.
- **There is no fallback between providers.** A missing or invalid credential
  fails clearly; answering as a different provider would make the result
  unattributable.
- An invalid `AI_PROVIDER` is rejected when settings load, not at request time.
- The local provider never silently becomes a cloud one. If Ollama is down, the
  model is missing, or the drive holding the models is disconnected, the run
  says so and the deterministic evidence is preserved.

## Tests

No test contacts a real provider, Nominatim, STAC or real imagery. Fakes are
hand-written recording doubles; `app.dependency_overrides` injects them at the
route boundary.

**The suite is independent of the machine it runs on.** `tests/conftest.py`
clears every provider variable and stops `Settings` reading a developer's
`.env`, so a credentialed laptop and a fresh CI checkout run the same suite. A
test that needs a provider configured sets an obviously fake value itself and
clears the settings cache. Measured before that fixture existed: 2112 passed
with a real key present, 2108 passed and 4 failed without one — same commit,
same command.
