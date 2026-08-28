import { apiRequest } from "./client";
import type { GeoResolveRequest, GeoResolveResponse } from "./types";

/**
 * Resolve a place name or bounding box into a validated geographic
 * representation (center point + bounding box) via the backend.
 */
export function resolveLocation(
  request: GeoResolveRequest,
  signal?: AbortSignal,
): Promise<GeoResolveResponse> {
  return apiRequest<GeoResolveResponse>("/api/v1/geospatial/resolve", {
    method: "POST",
    body: request,
    signal,
  });
}
