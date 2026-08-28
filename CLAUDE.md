# SatQuery AI — Claude Code Instructions

## Project

SatQuery AI is a Smart India Hackathon 2026 system for natural-language interaction with multimodal remote-sensing imagery.

Core intended capabilities:
- Natural-language satellite queries
- Geospatial grounding
- Sentinel-2 optical imagery
- Sentinel-1 SAR imagery
- Multitemporal analysis
- Change detection
- Multimodal reasoning
- Vision-language reasoning
- Geospatial localization
- Explainable results
- Visualization

## Current Implemented Pipeline

1. Geospatial resolution
2. Sentinel-2 STAC scene discovery
3. Bounded Sentinel-2 imagery retrieval
4. Structured query intent
5. Deterministic query-plan resolution

Current HEAD represents the completed Query/Intent Grounding phase.

## Architecture Rules

Keep these concepts separate:

SatQueryIntent
→ Geospatial Service
→ BoundingBox
→ ResolvedQueryPlan
→ STAC discovery
→ Imagery retrieval
→ future analysis/reasoning layers

Reuse existing services, schemas and contracts whenever possible.

Do not duplicate:
- Geospatial resolution
- BoundingBox
- STAC discovery
- Imagery retrieval

Inspect existing implementations before creating new abstractions.

## Phase Discipline

Implement only the phase explicitly requested.

Do not prematurely implement future capabilities such as:
- Sentinel-1 processing
- SAR fusion
- Change detection
- Temporal analysis
- VLM
- LLM/NLP parsing
- AI inference
- MapLibre
- spectral indices

unless the current task explicitly authorizes them.

Do not modify working functionality unnecessarily.

## Development Rules

- Inspect before editing.
- Prefer small, focused changes.
- Reuse existing architecture.
- Avoid unnecessary dependencies.
- Never add credentials, secrets or API keys.
- Never download large datasets or imagery unless explicitly requested.
- Maintain backward compatibility with existing functionality.
- Keep backend and frontend contracts synchronized.
- Use typed Pydantic models on the backend.
- Use TypeScript types on the frontend.

## Testing

Before and after significant changes run the project's existing checks.

Backend:
- pytest
- ruff check .

Frontend:
- npm run lint
- npm run typecheck
- npm run test
- npm run build

Also run:

git diff --check

Do not claim completion if checks fail.

## Git

Each implementation phase should have its own focused commit.

Before committing:
- inspect git diff
- run tests/checks
- run git diff --check
- verify no unrelated files changed

Never force-push or rewrite history unless explicitly instructed.

## Important

The repository's existing implementation is authoritative.

Do not assume a file, service, schema, dependency or API exists.

Inspect the repository first and adapt to the actual codebase.