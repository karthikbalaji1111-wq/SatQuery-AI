/**
 * Runtime shape checks for the responses this application stores.
 *
 * The frontend keeps its own TypeScript mirror of the backend contracts, and
 * TypeScript is erased at runtime: `apiRequest<AgentResult>(...)` is a *claim*
 * about what will arrive, not a check. When the claim is wrong - an older or
 * newer server, a proxy that rewrote the body, a partially deployed stack - the
 * mistake surfaces far from its cause, as `undefined is not an object` inside a
 * component rendering evidence that was never there.
 *
 * So the critical responses are checked at the boundary and refused as an
 * `ApiError` if they cannot be used. One error type, one place, one moment.
 *
 * **Deliberately shallow.** These verify the SHAPE the UI depends on - the
 * fields it indexes into without asking first - and nothing more. A full mirror
 * of the backend schema would be a second contract to keep in step with the
 * first, and the two would drift; the backend's own Pydantic models remain the
 * authority on what is valid. What this catches is a response that is
 * structurally unusable, not one that is semantically wrong.
 */

import { ApiError } from "./client";
import type {
  AgentResult,
  AnalysisResult,
  GeoResolveResponse,
  ModelCatalogResponse,
  QueryExecutionResult,
  SceneSearchResponse,
} from "./types";

function fail(what: string, why: string): never {
  throw new ApiError(
    `The ${what} response could not be used: ${why}.`,
    200,
    "invalid_response",
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function record(value: unknown, what: string): Record<string, unknown> {
  if (!isRecord(value)) fail(what, "it is not an object");
  return value;
}

function requireArray(
  body: Record<string, unknown>,
  key: string,
  what: string,
): void {
  if (!Array.isArray(body[key])) fail(what, `'${key}' is missing or not a list`);
}

function requireString(
  body: Record<string, unknown>,
  key: string,
  what: string,
): void {
  if (typeof body[key] !== "string") fail(what, `'${key}' is missing`);
}

function requireNumber(
  body: Record<string, unknown>,
  key: string,
  what: string,
): void {
  if (typeof body[key] !== "number") fail(what, `'${key}' is missing`);
}

/** The six statuses are exhaustive; anything else means a contract mismatch. */
const AGENT_STATUSES = new Set([
  "ok",
  "planner_unavailable",
  "synthesis_unavailable",
  "answer_withheld",
  "needs_clarification",
  "location_unavailable",
]);

export function asAgentResult(value: unknown): AgentResult {
  const body = record(value, "agent");
  if (typeof body.status !== "string" || !AGENT_STATUSES.has(body.status)) {
    fail("agent", "'status' is not one this build understands");
  }
  // The panels read `trace.steps` and `evidence.items` directly.
  const trace = record(body.trace, "agent");
  requireArray(trace, "steps", "agent");
  const evidence = record(body.evidence, "agent");
  requireArray(evidence, "items", "agent");
  // The question panel renders the clarification's message and options, so a
  // result claiming to need one must actually carry them.
  if (body.status === "needs_clarification") {
    const clarification = record(body.clarification, "agent");
    requireString(clarification, "message", "agent");
    requireArray(clarification, "options", "agent");
  }
  // A location outage is shown from its failure (what, and when to retry), so
  // a result claiming one must carry it.
  if (body.status === "location_unavailable") {
    const failure = record(body.failure, "agent");
    requireString(failure, "message", "agent");
    requireString(failure, "code", "agent");
  }
  return value as AgentResult;
}

export function asExecutionResult(value: unknown): QueryExecutionResult {
  const body = record(value, "query execution");
  requireArray(body, "windows", "query execution");
  requireArray(body, "executed_modalities", "query execution");
  requireString(body, "catalog", "query execution");
  // Everything downstream - the map, the analysis request, the export - reads
  // the plan's own bbox and intent.
  const plan = record(body.plan, "query execution");
  record(plan.intent, "query execution");
  record(plan.bbox, "query execution");
  return value as QueryExecutionResult;
}

export function asAnalysisResult(value: unknown): AnalysisResult {
  const body = record(value, "analysis");
  requireString(body, "status", "analysis");
  requireString(body, "task", "analysis");
  requireString(body, "answer", "analysis");
  requireArray(body, "measurements", "analysis");
  requireArray(body, "warnings", "analysis");
  requireArray(body, "windows_considered", "analysis");
  return value as AnalysisResult;
}

export function asModelCatalog(value: unknown): ModelCatalogResponse {
  const body = record(value, "model catalog");
  requireArray(body, "models", "model catalog");
  requireString(body, "default_provider", "model catalog");
  requireString(body, "default_model", "model catalog");
  return value as ModelCatalogResponse;
}

export function asResolveResponse(value: unknown): GeoResolveResponse {
  const body = record(value, "location");
  const centre = record(body.center, "location");
  if (typeof centre.lat !== "number" || typeof centre.lon !== "number") {
    fail("location", "'center' has no usable coordinates");
  }
  const bbox = record(body.bbox, "location");
  for (const edge of ["west", "south", "east", "north"]) {
    if (typeof bbox[edge] !== "number") {
      fail("location", `'bbox.${edge}' is missing`);
    }
  }
  return value as GeoResolveResponse;
}

export function asSceneSearchResponse(value: unknown): SceneSearchResponse {
  const body = record(value, "scene search");
  requireArray(body, "scenes", "scene search");
  requireNumber(body, "scene_count", "scene search");
  requireString(body, "catalog", "scene search");
  return value as SceneSearchResponse;
}
