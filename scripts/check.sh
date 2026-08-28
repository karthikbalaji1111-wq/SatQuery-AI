#!/usr/bin/env bash
# Run every lint / type / test / build check. Mirrors CI.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Frontend: lint"
(cd "$ROOT/frontend" && npm run lint)
echo "==> Frontend: typecheck"
(cd "$ROOT/frontend" && npm run typecheck)
echo "==> Frontend: test"
(cd "$ROOT/frontend" && npm run test)
echo "==> Frontend: build"
(cd "$ROOT/frontend" && npm run build)

echo "==> Backend: ruff"
(cd "$ROOT/backend" && uv run ruff check .)
echo "==> Backend: pytest"
(cd "$ROOT/backend" && uv run pytest)

echo "==> All checks passed."
