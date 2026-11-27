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
are operator prerequisites; this stack does not provision them.

**What a public deployment does and does not have.** There are no user
accounts. What makes an anonymous public deployment safe to run is narrower:
in production a caller cannot choose an AI provider or model (403
`ai_selection_disabled`, see below), so nobody can spend a metered key; every
expensive route is rate-limited per caller, bounded in concurrency and in time;
and the standard workflow needs no key at all. Per-user quotas, attribution and
revocation need accounts, which do not exist.

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
release commit or tag was created. Base images are now pinned by digest (see Supply chain); the OS package
repository used by `apt-get` remains mutable, so this is still not a claim of
bit-for-bit reproducibility.

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
| `SATQUERY_TRUSTED_PROXY_HOPS` | `0` | Proxies in front that append to `X-Forwarded-For`; see below |
| `SATQUERY_MAX_CONCURRENT_WORKFLOWS` | `4` | Simultaneous discovery / analysis / agent runs (render.yaml: `2`) |
| `SATQUERY_MAX_CONCURRENT_RASTER_READS` | `6` | Simultaneous windowed COG reads, per process. Bounds open remote connections; memory is bounded by workflows x tile size |
| `SATQUERY_ADMISSION_WAIT_SECONDS` | `2` | How long a request waits for a workflow slot before 503 |
| `SATQUERY_RASTER_ADMISSION_WAIT_SECONDS` | `60` | How long one raster read waits for a raster slot before 503 |
| `SATQUERY_WORKFLOW_BUDGET_SECONDS` | `900` | Total time for one request on an expensive route; past it, 504 `workflow_timeout` |
| `SATQUERY_IMAGERY_HARD_MAX_DIMENSION` | `2048` | Largest SINGLE native-resolution read per side, and the tile size of a tiled analysis (render.yaml: `1024`) |
| `SATQUERY_IMAGERY_TILED_MAX_DIMENSION` | `5000` | Largest analysis area per side, read in tiles; larger areas are refused as too large (render.yaml: `4500`, a whole city such as Chennai) |
| `SATQUERY_IMAGERY_TILED_MAX_WINDOW_PIXELS` | `25000000` | Total pixels a tiled analysis may cover (render.yaml: `16000000`) |
| `SATQUERY_CLIENT_AI_SELECTION` | *(unset)* | Whether a caller may name an AI provider/model. Unset: allowed in development, refused in production |
| `SATQUERY_LOG_FORMAT` | `text` | `json` writes one JSON object per log line |
| `SATQUERY_METRICS_TOKEN` | *(unset)* | Bearer token for `GET /metrics`; unset, the route answers 404 |
| `SATQUERY_GEOCODER_MIN_INTERVAL_SECONDS` | `1` | Minimum spacing between geocoder requests |
| `SATQUERY_GEOCODER_CACHE_TTL_SECONDS` | `86400` | How long a resolved place is reused (successes only) |
| `SATQUERY_GEOCODER_CACHE_ENTRIES` | `256` | Places kept in the geocoder cache (least recently used evicted) |
| `SATQUERY_GEOCODER_MAX_ATTEMPTS` | `3` | Upstream requests one geocode may make |
| `SATQUERY_GEOCODER_RETRY_BUDGET_SECONDS` | `15` | Most one geocode waits (backoff, Retry-After, cooldown) before `geocoding_unavailable` |
| `SATQUERY_GEOCODER_BACKOFF_BASE_SECONDS` | `2` | First backoff; doubles per consecutive failure |
| `SATQUERY_GEOCODER_MAX_COOLDOWN_SECONDS` | `300` | Cap on the backoff-derived cooldown after HTTP 429 |
| `SATQUERY_GEOCODER_WARM_PLACES` | *(empty)* | `;`-separated places geocoded in the background at startup |
| `SATQUERY_GEOCODER_NEW_PLACES_PER_CALLER` | `30` | New (uncached) places one client may look up per window; `0` disables. Cached places are never refused |
| `SATQUERY_GEOCODER_NEW_PLACES_WINDOW_SECONDS` | `600` | The window those lookups are counted in |
| `SATQUERY_TRUSTED_ASSET_HOSTS` | *(empty)* | Hosts whose rasters may be opened |
| `SATQUERY_IMAGERY_MAX_DIMENSION` | `1024` | Pixels per side of a returned window |
| `SATQUERY_IMAGERY_MAX_WINDOW_PIXELS` | `50000000` | Total pixels a quantitative read may cover |

### Large areas: tiled analysis

An analysis area larger than one read (`SATQUERY_IMAGERY_HARD_MAX_DIMENSION`)
is measured in tiles of at most that size, up to the tiled limits above. The
tile plan comes from the catalog's own `proj:*` grid, tiles share exact pixel
edges, and every pixel is read once at native resolution - nothing is
decimated. Tile results are combined exactly: counts add, means are
pixel-weighted, and radar means are merged in linear power before the
conversion to dB. Tests compare a tiled run with one read of the same area
(identical counts and quality notes, means within 1e-12). A two-period
comparison's change map is drawn at most 1024 px on its long side; its
statistics use every pixel.

Memory does not grow with the area: one tile per workflow is held at a time
(measured about 132 MB per workflow at 1024 px tiles, so `2` workflows fit
Render's 512 MB). Time does grow: measured locally, all of Chennai (2171 x 4245
px, 15 tiles) took 194 s for NDVI and 241 s for radar. That is inside the
900 s workflow budget, but a free instance with a slow network may not be.
Past the budget the request ends with 504 `workflow_timeout` and its slot is
freed.

### The geocoder on a shared outbound IP

Nominatim limits requests per IP, and a cloud host's outbound IPs are shared
(Render: "shared across all services in the same region"), so the budget this
application spends is shared with strangers. A 429 therefore says the SHARED
budget is spent. `app/services/geospatial/nominatim.py` asks as little as it
can and backs off when refused: a day-long cache of real answers, one upstream
sequence per place however many callers ask at once, one request per second,
and after a 429 no request at all until `Retry-After` (or an exponential
backoff, when absent) has passed. A caller who would wait longer than the retry
budget gets **503 `geocoding_unavailable`** with `Retry-After` at once - never a
guessed location. `/ready` reports the geocoder's counters (upstream requests,
429s, cache hits, coalesced callers, refusals during a cooldown) without
contacting it. A free instance loses its cache when it sleeps;
`SATQUERY_GEOCODER_WARM_PLACES` refills it with real answers in the background
after startup. One client may look up at most `SATQUERY_GEOCODER_NEW_PLACES_PER_CALLER` new places per window (cached places never count), so a single caller cannot spend everyone's budget; past it the client gets 429 `rate_limited` with `Retry-After` (an agent run reports it as "Location service temporarily unavailable" with the time to wait). None of this can create budget the shared IP does not have: a
dedicated outbound IP or a keyed/self-hosted geocoder is the durable fix.

Four refusals, deliberately distinguishable:

* **429** — this client is asking too often. Carries `Retry-After`.
* **503** — this process is at capacity. The caller did nothing wrong.
* **504** — this one request used its whole `SATQUERY_WORKFLOW_BUDGET_SECONDS`.
  The slot is freed at once; a raster read already running in a worker thread
  finishes (bounded by its HTTP timeouts) and its result is discarded.
* **413** — this body is too large.

### Who a request is attributed to

Behind a load balancer the TCP peer is the balancer, so a peer-keyed limit is
one allowance shared by everyone. Each proxy APPENDS the address it received
the connection from to `X-Forwarded-For`, so the entry
`SATQUERY_TRUSTED_PROXY_HOPS` places from the right was written by the outermost
trusted proxy and cannot be forged by the caller. That entry is used; a shorter
chain or a non-address falls back to the peer. The container runs uvicorn with
`--no-proxy-headers`, because uvicorn told to trust every proxy
(`FORWARDED_ALLOW_IPS=*`) takes the LEFTMOST entry, which the caller chose.

Render does not document its forwarding chain. `render.yaml` sets `1` (its
load balancer). To measure it: after a deploy, load the site once in a browser
and read the log line `first request carried N X-Forwarded-For entries`
(logged once per process; the count only, never an address). Set the variable
to `N`.

### Memory: why render.yaml lowers two limits

Measured with `backend/scripts/benchmark_memory.py` (the heaviest path: NDVI,
NDWI and NDBI with SCL masking, both dates of a comparison held):

| Window ceiling | One date | Two dates |
|---|---|---|
| 1024 px | 118 MB | 132 MB |
| 2048 px | 385 MB | 454 MB |

An idle server holds about 104 MB. The defaults (2048 px, 4 workflows) need
about 1.9 GB worst case; on Render's 512 MB free plan a single full-size
comparison could exhaust memory. `render.yaml` therefore sets a 1024 px window
(about 10 km at 10 m) and 2 workflows, about 370 MB worst case, and
`tests/test_deployment_memory_budget.py` fails if those values stop fitting.
On a 2 GB instance the defaults fit.

### Caller-chosen AI is off in production

`POST /query/agent` and `POST /query/parse` accept a `provider` and `model`.
With no accounts, letting any caller name them lets anyone spend the
operator's metered keys, so production refuses them with **403
`ai_selection_disabled`** before any provider is built, and
`GET /api/v1/ai/models` reports `client_selection_enabled: false` so the web
app shows "Standard · no external AI" instead of a picker. The standard
workflow (deterministic rules plus the local intent model, no key) answers
every question. Set `SATQUERY_CLIENT_AI_SELECTION=true` only where every
caller is trusted. A configured `AI_PROVIDER` key is still used for one step:
describing a retrieved image when a question asks what it shows.

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

Every response carries **`X-Request-ID`** (readable by the browser app, which
appends it to error messages as "reference …"), and every log line written
while that request runs carries the same id. One access line is written per
request: method, route, status and milliseconds, never a query string, body,
place name or address. Probes (`/health`, `/ready`, `/metrics`) log at DEBUG.

`SATQUERY_LOG_FORMAT=json` writes one JSON object per line (`ts`, `level`,
`logger`, `run_id`, `message`, plus `exc_type`/`exc` for exceptions) for a log
platform to index.

With `SATQUERY_METRICS_TOKEN` set, `GET /metrics` serves Prometheus text
(`satquery_http_requests_total{route,method,status}`, a latency histogram per
route, and requests in flight) to a caller sending
`Authorization: Bearer <token>`. Metrics are per process. Paths that matched no
route share the label `unmatched`, so probing random URLs cannot grow the
series set. Nothing scrapes it by default: on a host without Prometheus, use
the access lines.

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

## Supply chain

Base images are pinned by digest and uv by version and digest; CI actions are
pinned to commit SHAs and the workflow token is read-only. Dependabot proposes
updates weekly; each still has to pass CI. CI fails on:

* `scripts/check_secrets.py`: provider keys, private keys, a committed `.env`,
  `.zip` or `.mcp.json` in any tracked file (test sentinels marked `SENTINEL`
  are allowed);
* `pip-audit` on the locked production Python dependencies;
* `npm audit --omit=dev --audit-level=high` on the shipped npm dependencies.

## Scaling notes

A modular monolith with a bounded synchronous workflow. A long agent run is
tied to its HTTP request, and **browser cancellation does not cancel the backend
work** — the client stops waiting; the server finishes what it started, bounded
by `SATQUERY_WORKFLOW_BUDGET_SECONDS`. Durable jobs would change that and are
not implemented; if they are ever needed, the shape is a controlled worker tier
behind the same API, not a fleet of microservices.
