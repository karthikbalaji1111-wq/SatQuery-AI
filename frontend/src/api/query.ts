import { apiRequest } from "./client";
import type { ResolvedQueryPlan, SatQueryIntent } from "./types";

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
