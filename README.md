# SatQuery

Natural-language satellite query platform for Smart India Hackathon 2026,
Problem Statement **26167**.

## Status

Implemented and tested: natural-language intent parsing (Gemini), geocoding and
AOI grounding (Nominatim), Sentinel-2 optical **and** Sentinel-1 SAR scene
discovery (Earth Search STAC), deterministic per-modality scene selection,
bounded windowed imagery retrieval (COG range reads → PNG), quantitative
raw-value band access, single-scene Sentinel-2 NDWI statistics, metadata-only
observation compatibility reporting, Temporal NDWI Statistics across two
Sentinel-2 dates, and **agentic orchestration** over those deterministic tools.

**Agentic orchestration** (`POST /api/v1/query/agent`): a language model
proposes a plan over a closed three-tool allowlist; the server validates it,
executes it through the same services the manual endpoints use, and
mechanically checks the generated answer against the collected evidence before
returning it. The model chooses what to run — it never computes a result, and
it never receives an image.

Design philosophy: every phase is written **test-first**, and failure is
deterministic. When planning, synthesis or answer validation fails, the API
still returns HTTP 200 with whatever deterministic evidence was established and
**no fabricated answer** — the measurements are the product, the prose is a
presentation of them.

Not implemented: raster co-registration or resampling, per-pixel change
detection, optical/SAR fusion, vision-language reasoning, object detection, and
map rendering (`MapPanel` is still a placeholder; there is no MapLibre
dependency).

**Not yet verified:** the agent path is covered by deterministic tests against
fake provider clients; a live end-to-end Gemini run has not been exercised.

See `CLAUDE.md` for the authoritative phase-by-phase state and the boundaries
each phase deliberately did not cross.

## Stack

| Layer | Tech |
| --- | --- |
| Frontend | Vite + React + TypeScript, ESLint, Vitest (structured for MapLibre) |
| Backend | Python 3.12 + FastAPI, managed with uv, Ruff, pytest |
| Dev infra | Docker Compose, GitHub Actions CI |

## Layout

```
SatQuery/
├── backend/                 FastAPI service
│   ├── app/
│   │   ├── main.py          app factory (CORS, routers, error handlers)
│   │   ├── core/            config, logging, error types
│   │   ├── api/             routers — health, geospatial, satellite, query
│   │   └── services/        domain boundaries:
│   │       ├── query/       natural-language query orchestration
│   │       ├── satellite/   Sentinel-1 SAR + Sentinel-2 optical retrieval
│   │       ├── multimodal/  SAR + optical + text fusion (STUB - not implemented)
│   │       ├── temporal/    multitemporal change detection (STUB - not implemented)
│   │       ├── geospatial/  geocoding, AOI, spatial grounding
│   │       ├── ai/          Gemini-backed intent parsing (no vision model)
│   │       └── map/         map layer / tile preparation (STUB - not implemented)
│   └── tests/
├── frontend/                React app
│   └── src/
│       ├── api/             typed backend client (client.ts, health.ts, types.ts)
│       ├── config/          env.ts — reads VITE_API_BASE_URL
│       ├── components/      BackendStatus (live /health check)
│       └── features/
│           ├── query/       QueryPanel (parse → plan → execute → analyse)
│           └── map/         MapPanel (placeholder; no MapLibre dependency)
├── scripts/                 setup.sh · dev.sh · check.sh
├── docker-compose.yml       local backend + frontend stack
└── .github/workflows/ci.yml lint · typecheck · test · build
```

## Prerequisites

- Node.js 22+ and npm
- [uv](https://docs.astral.sh/uv/) (installs Python 3.12 automatically)
- Docker (optional, for the Compose stack)

## Setup

```bash
./scripts/setup.sh
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env.local
```

## Run

**Both servers:**

```bash
./scripts/dev.sh
```

**Individually:**

```bash
# backend  → http://localhost:8000  (docs at /docs, health at /health)
cd backend && uv run uvicorn app.main:app --reload --port 8000

# frontend → http://localhost:5173
cd frontend && npm run dev
```

**With Docker:**

```bash
docker compose up --build
```

## Checks

```bash
./scripts/check.sh          # everything, mirrors CI
```

Or individually:

| Command | Runs in |
| --- | --- |
| `npm run lint` | frontend |
| `npm run typecheck` | frontend |
| `npm run test` | frontend |
| `npm run build` | frontend |
| `uv run ruff check .` | backend |
| `uv run pytest` | backend |

## Configuration

| Variable | Side | Default | Purpose |
| --- | --- | --- | --- |
| `SATQUERY_ENVIRONMENT` | backend | `development` | environment name |
| `SATQUERY_LOG_LEVEL` | backend | `INFO` | log verbosity |
| `SATQUERY_CORS_ORIGINS` | backend | `localhost:5173,...` | allowed frontend origins |
| `VITE_API_BASE_URL` | frontend | `http://localhost:8000` | backend base URL |

## Roadmap

1. **Geospatial grounding** — geocoding + AOI parsing in `services/geospatial`.
2. **Satellite retrieval** — Sentinel-1/2 access in `services/satellite`.
3. **Query + AI** — NL parsing and model inference (`services/query`, `services/ai`).
4. **Agentic orchestration** — ✅ implemented in `services/agent`, exposed at
   `POST /api/v1/query/agent` and in the `AgentPanel` UI.
5. **Multimodal + temporal** — fusion and change detection.
6. **Map** — MapLibre integration in `frontend/src/features/map` fed by `services/map`.

