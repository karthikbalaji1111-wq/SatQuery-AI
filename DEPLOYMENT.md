# Deploying SatQuery

Two stacks, deliberately kept apart.

| | Development | Production |
|---|---|---|
| Compose file | `docker-compose.yml` | `docker-compose.prod.yml` |
| Backend | source bind-mounted, `--reload`, root | image-copied source, no reload, non-root, locked deps |
| Frontend | `vite dev` (a compiler with a web server attached) | built once, served statically by nginx |
| Dependencies | resolved at build time | `uv sync --frozen` / `npm ci` from lockfiles |

The production stack is a separate file rather than an override. An override has
to *un-say* the development settings, and one forgotten line ships a reloading,
root-owned service with the source tree mounted into it.

```bash
# Development (unchanged)
docker compose up --build

# Production
export GEMINI_API_KEY=...            # whatever this deployment holds
export SATQUERY_AI_PROVIDER=gemini
export SATQUERY_CORS_ORIGINS=https://satquery.example.com
export SATQUERY_PUBLIC_API_URL=https://api.satquery.example.com
docker compose -f docker-compose.prod.yml up --build -d
```

Secrets are read from the host environment. Nothing is baked into an image and
nothing is written into a committed file.

Replace the example domains with your own. A host TLS reverse proxy must route
the frontend domain to `127.0.0.1:8080` and the API domain to `127.0.0.1:8000`.
Both published ports are loopback-only. TLS certificates and external ingress
are operator prerequisites; this stack does not provision them. Do not expose
the API publicly before authentication and quotas are implemented.

Compose refuses missing/empty `SATQUERY_CORS_ORIGINS` and
`SATQUERY_PUBLIC_API_URL`. Backend production startup additionally rejects HTTP,
localhost, local/internal names, IP literals, wildcard origins, credentials,
paths, queries and fragments in CORS origins, and requires an asset allowlist.
Public upstream URLs require HTTPS. Private Ollama and health-check transport
may use HTTP. These checks validate syntax and policy, not DNS ownership or TLS
certificate validity.

Every frontend build requires an explicit public HTTPS API URL, including local
release checks: `VITE_API_BASE_URL=https://api.satquery.example.com npm run build`.
Development `npm run dev` retains its localhost behavior. The frontend URL is
baked into the image; changing it requires rebuilding. CI runs type checking,
tests, the exact production build and both production Dockerfiles. The backend
image defaults to production mode even outside Compose. CI also imports the
backend inside its image to catch missing native libraries (the first real
container run exposed a missing `libexpat1` dependency used by Rasterio).

## Verification status

Milestone 1 verification on 2026-09-19:

* Frontend: 352 unit tests, TypeScript, ESLint and the exact `npm run build`
  passed. Empty and HTTP/localhost API URLs were refused by that build command.
* Backend: 2,316 tests and Ruff passed, including 31 production configuration
  cases. Those tests use local/fake providers, not live satellite or AI services.
* The real frontend production Docker image built on Linux/ARM64. Its nginx
  configuration, both HTML entries, emitted JavaScript (including the MapLibre
  worker), and missing-asset 404 behavior were smoke-tested in an isolated
  container.
* The real backend production image built on Linux/ARM64 after adding the
  missing `libexpat1` runtime dependency. Its actual startup command exits with
  a configuration error when required production settings are absent. With
  explicit HTTPS origins and an asset allowlist, `/health` returned HTTP 200
  and `environment=production`. The smoke test used no API keys or external
  network; temporary containers were removed afterward.
* Compose validation passes with explicit HTTPS configuration and refuses
  missing required URLs.

GitHub-hosted CI, Linux/AMD64, external HTTPS ingress, real browser E2E, live
provider workflows and load tests remain unverified by this milestone.
The source tree still includes pre-existing uncommitted/untracked work; no
release commit or tag was created. Base-image tags and OS package repositories
remain mutable, so this is not a claim of bit-for-bit reproducibility.

## Why the frontend is served the way it is

The nginx configuration refuses to serve `index.html` for a missing file under
`/assets/`:

```nginx
location /assets/ { try_files $uri =404; }
```

This is not a preference. The usual single-page fallback — "try the file,
otherwise index.html" — turns a missing build artefact into a 200 response
containing HTML. The browser then loads that HTML *as JavaScript* and reports a
syntax error from somewhere inside the bundle. That is precisely how the
MapLibre worker failed before this release: the worker asset was never emitted,
the host answered with the application's own page, and the map rendered a
basemap with no footprint geometry on it. A 404 is a fact an operator can act
on; a 200 of the wrong content type is a mystery.

## Resource limits

All limits are read from the environment. Defaults are in
`backend/app/core/config.py`; the production compose file sets them explicitly so
that the running configuration is visible in one place.

| Variable | Default | What it bounds |
|---|---|---|
| `SATQUERY_MAX_REQUEST_BYTES` | `1000000` | Request body size, refused before the body is read |
| `SATQUERY_RATE_LIMIT_REQUESTS` | `120` | Requests per client per window, on the expensive routes |
| `SATQUERY_RATE_LIMIT_WINDOW_SECONDS` | `60` | The window those requests are counted in |
| `SATQUERY_MAX_CONCURRENT_WORKFLOWS` | `4` | Simultaneous discovery / analysis / agent runs |
| `SATQUERY_MAX_CONCURRENT_RASTER_READS` | `2` | Simultaneous windowed COG reads |
| `SATQUERY_ADMISSION_WAIT_SECONDS` | `2` | How long a request waits for a slot before 503 |
| `SATQUERY_WORKFLOW_BUDGET_SECONDS` | `900` | Total budget for one workflow |
| `SATQUERY_GEOCODER_MIN_INTERVAL_SECONDS` | `1` | Minimum spacing between geocoder requests |
| `SATQUERY_GEOCODER_CACHE_TTL_SECONDS` | `900` | How long a resolved place is reused |
| `SATQUERY_TRUSTED_ASSET_HOSTS` | *(empty)* | Hosts whose rasters may be opened |
| `SATQUERY_IMAGERY_MAX_DIMENSION` | `1024` | Pixels per side of a returned window |
| `SATQUERY_IMAGERY_MAX_WINDOW_PIXELS` | `50000000` | Total pixels a quantitative read may cover |

Three refusals, deliberately distinguishable:

* **429** — this client is asking too often. Carries `Retry-After`.
* **503** — this process is at capacity. The caller did nothing wrong.
* **413** — this body is too large.

### The limitation that matters: they are PER PROCESS

The rate limiter's counters, the workflow slots, the raster gate and the
geocoder's throttle all live in one process's memory. That has three
consequences, none of which is hidden:

1. **`--workers N` multiplies every limit by N.** The backend image runs a
   single uvicorn worker for exactly this reason. Raising it means dividing the
   limits by the worker count.
2. **N replicas admit N times as much**, and make N requests per second to
   Nominatim rather than one. A deployment running more than one replica **must
   not claim** compliance with OpenStreetMap's usage policy on the strength of
   this code alone.
3. **A restart forgets every counter.**

Making these limits global needs shared storage (Redis or equivalent). That is a
real dependency and nothing else here needs it, so it has not been added. The
boundary is drawn so the substitution is small: the limiter and the gate are
constructed once, in `install_request_limits`, and used only through the
dependencies at the bottom of `app/core/limits.py`.

## Liveness and readiness

| Endpoint | Question | Wire it to |
|---|---|---|
| `GET /health` | Is this process alive? | the container restart policy |
| `GET /ready` | Can this deployment do its work? | the load balancer / readiness gate |

`/health` answers from configuration alone and keeps answering while the service
is busy, misconfigured, or cut off from every upstream — none of which is a
reason to kill it.

`/ready` reports each capability (application, satellite catalogs, geocoder,
selected AI provider) and answers **503 when any is not ready**, so status-code
semantics work; the body still names which one and why.

**Readiness never makes a paid provider call.** A probe that spent quota to
answer would attach a bill to every poll a monitoring system makes. A configured
cloud provider is reported as *configured* and nothing more is claimed. The one
exception is the local provider when it is the selected one: Ollama runs on the
same machine, and one read-only `GET /api/tags` is what actually determines
whether a local run can work.

Do **not** wire `/ready` to the restart policy. "The AI provider is
unconfigured" is not a reason to kill a healthy process, and restarting it will
not configure anything.

## Observability

Every workflow opens a correlated scope. A short run id is attached to every log
line emitted anywhere during that run — including third-party loggers — and each
stage reports its own duration:

```
2026-09-17T08:14:22+0530 INFO  satquery.run [3f9a1c2b7d01] | stage=planning outcome=ok ms=812
2026-09-17T08:14:47+0530 INFO  satquery.run [3f9a1c2b7d01] | stage=execution outcome=ok ms=24193 steps=2
2026-09-17T08:14:51+0530 INFO  satquery.run [3f9a1c2b7d01] | workflow=agent outcome=ok ms=29418 provider=local
```

Fields are names, ids, durations and outcomes. A question, a place name and a
credential are never logged, and a failing stage records the exception **class**
rather than its message, because an upstream message can carry content this
system did not write.

## Scaling notes

A modular monolith with a bounded synchronous workflow. A long agent run is
tied to its HTTP request, and **browser cancellation does not cancel the backend
work** — the client stops waiting; the server finishes what it started, bounded
by `SATQUERY_WORKFLOW_BUDGET_SECONDS`. Durable jobs would change that and are
not implemented; if they are ever needed, the shape is a controlled worker tier
behind the same API, not a fleet of microservices.
