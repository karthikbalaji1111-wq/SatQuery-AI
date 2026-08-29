import { apiRequest } from "./client";
import type {
  AnalysisRequest,
  AnalysisResult,
  QueryExecutionRequest,
  QueryExecutionResult,
  ResolvedQueryPlan,
  SatQueryIntent,
} from "./types";

/**
 * Translate a natural-language request into a structured {@link SatQueryIntent}.
 * Parsing only - the backend does not build a plan, geocode, or run discovery.
 */
export function parsePrompt(
  prompt: string,
  signal?: AbortSignal,
): Promise<SatQueryIntent> {
  return apiRequest<SatQueryIntent>("/api/v1/query/parse", {
    method: "POST",
    body: { prompt },
    signal,
  });
}

/**
 * Validate a structured query intent and ground its location to a bounding box
 * via the backend Geospatial Service. No STAC discovery or imagery retrieval.
 */
export function buildQueryPlan(
  intent: SatQueryIntent,
  signal?: AbortSignal,
): Promise<ResolvedQueryPlan> {
  return apiRequest<ResolvedQueryPlan>("/api/v1/query/build-plan", {
    method: "POST",
    body: intent,
    signal,
  });
}

/**
 * Execute a validated {@link SatQueryIntent} end to end: ground the location,
 * run Sentinel-2 discovery once per temporal window, deterministically select a
 * scene, and - when {@link QueryExecutionRequest.include_imagery} is set -
 * retrieve one bounded RGB window per selected scene. Sentinel-1 SAR is
 * reported as skipped, not executed.
 */
export function executeQuery(
  request: QueryExecutionRequest,
  signal?: AbortSignal,
): Promise<QueryExecutionResult> {
  return apiRequest<QueryExecutionResult>("/api/v1/query/execute", {
    method: "POST",
    body: request,
    signal,
  });
}

/**
 * Interpret an already-computed {@link QueryExecutionResult}. The analysis task
 * is derived from `execution.plan.intent.task`. The backend performs no
 * discovery, no imagery retrieval, and no model inference here; a task with no
 * engine yet comes back as `status: "not_implemented"` in a 200 response.
 */
export function analyzeQuery(
  request: AnalysisRequest,
  signal?: AbortSignal,
): Promise<AnalysisResult> {
  return apiRequest<AnalysisResult>("/api/v1/query/analyze", {
    method: "POST",
    body: request,
    signal,
  });
}
