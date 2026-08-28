#!/usr/bin/env bash
# Run backend and frontend dev servers together.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

pids=()
cleanup() { for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

(cd "$ROOT/backend" && uv run uvicorn app.main:app --reload --port 8000) &
pids+=($!)

(cd "$ROOT/frontend" && npm run dev) &
pids+=($!)

echo "Backend:  http://localhost:8000  (docs: /docs)"
echo "Frontend: http://localhost:5173"
wait
