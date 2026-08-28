import { apiRequest } from "./client";
import type { ResolvedQueryPlan, SatQueryIntent } from "./types";

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
