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
