# SatQuery

Natural-language satellite query platform — **foundation build** for
Smart India Hackathon 2026, Problem Statement **26167**.

This repository is the architectural skeleton only. AI models, Sentinel-1 SAR /
Sentinel-2 optical retrieval, multimodal analysis, change detection, and map
rendering are **not implemented yet** — the code establishes clean module
boundaries and a working frontend ⇄ backend connection for later phases.

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
│   │   ├── api/             routers — currently /health
│   │   └── services/        domain boundaries (stubs, no logic):
│   │       ├── query/       natural-language query orchestration
│   │       ├── satellite/   Sentinel-1 SAR + Sentinel-2 optical retrieval
│   │       ├── multimodal/  SAR + optical + text fusion
│   │       ├── temporal/    multitemporal change detection
│   │       ├── geospatial/  geocoding, AOI, spatial grounding
│   │       ├── ai/          model inference and reasoning
│   │       └── map/         map layer / tile preparation
│   └── tests/
├── frontend/                React app
│   └── src/
│       ├── api/             typed backend client (client.ts, health.ts, types.ts)
│       ├── config/          env.ts — reads VITE_API_BASE_URL
│       ├── components/      BackendStatus (live /health check)
│       └── features/
│           ├── query/       QueryPanel (disabled until query service lands)
│           └── map/         MapPanel (MapLibre mount point)
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
4. **Multimodal + temporal** — fusion and change detection.
5. **Map** — MapLibre integration in `frontend/src/features/map` fed by `services/map`.
