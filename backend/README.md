# SatQuery Backend

Python 3.12 + FastAPI service foundation. Managed with [uv](https://docs.astral.sh/uv/).

## Setup

```bash
uv sync
```

## Run

```bash
uv run uvicorn app.main:app --reload --port 8000
```

Health check: <http://localhost:8000/health>
API docs: <http://localhost:8000/docs>

## Checks

```bash
uv run ruff check .
uv run pytest
```

## Layout

| Path | Responsibility |
| --- | --- |
| `app/main.py` | App factory, middleware, exception handlers |
| `app/core/config.py` | Environment configuration (`pydantic-settings`) |
| `app/core/logging.py` | Logging setup |
| `app/core/errors.py` | Error types + handlers |
| `app/api/` | Routers (currently `/health`) |
| `app/services/` | Domain module boundaries (query, satellite, multimodal, temporal, geospatial, ai, map) |

Each `app/services/<domain>/` package defines an interface and a stub that raises
`NotImplementedError`. No AI or satellite logic is implemented yet.

## AI providers

The agent's visual-analysis step - the one that looks at a retrieved satellite
PNG - runs through a provider-neutral `VisualAnalyst` abstraction. Two backends
implement it and are selected by configuration:

```bash
AI_PROVIDER=gemini    # Google Gemini      (GEMINI_API_KEY, GEMINI_MODEL)
AI_PROVIDER=nvidia    # NVIDIA-hosted NIM  (NVIDIA_API_KEY, NVIDIA_BASE_URL, NVIDIA_MODEL)
```

A single run may override the configured default:

```bash
curl -X POST localhost:8000/api/v1/query/agent \
  -H 'Content-Type: application/json' \
  -d '{"question": "...", "provider": "nvidia"}'
```

**Only the inference backend changes.** Geospatial grounding, STAC discovery,
Sentinel-1/Sentinel-2 retrieval, the raster path, NDWI, temporal analysis, the
grounding checks and the evidence contract are identical under either provider,
so deterministic results for the same scene and query are unaffected by the
choice. Only the model-generated interpretation differs, and it is attributed to
the provider and model that produced it.

Notes:

- **The NVIDIA model must accept image input.** The visual path sends a PNG, so
  a text-only model - including most Nemotron variants - cannot serve it. The
  default is `nvidia/nemotron-nano-12b-v2-vl`, a Nemotron *VL* model documented
  by NVIDIA as accepting PNG via the OpenAI-compatible `image_url` content part.
- **Models are catalogued with capabilities**, not hardcoded. `GET
  /api/v1/ai/models?role=visual` lists what this deployment can offer and
  whether each model is catalogued, configured and compatible. A model that
  cannot accept an image is refused *before* a request is made, rather than
  being asked to describe a picture it never received. Capability metadata is
  curated from NVIDIA's published documentation because the OpenAI-compatible
  `/v1/models` endpoint reports ids but not modalities; free-endpoint
  availability and limits change, so treat the catalog as current-at-writing.
- A run may name both: `{"question": "...", "provider": "nvidia", "model":
  "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"}`.
- **There is no fallback between providers.** A missing or invalid credential
  fails clearly; it never answers as the other provider, because that would make
  a result unattributable.
- An invalid `AI_PROVIDER` is rejected when settings load, not at request time.
