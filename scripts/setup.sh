#!/usr/bin/env bash
# Install all dependencies for the monorepo.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Backend (uv sync)"
(cd "$ROOT/backend" && uv sync)

echo "==> Frontend (npm install)"
(cd "$ROOT/frontend" && npm install)

echo "==> Done. Copy .env.example files as needed."
