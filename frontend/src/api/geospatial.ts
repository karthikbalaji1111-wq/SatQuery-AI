import { apiRequest } from "./client";
import { asResolveResponse } from "./validate";
import type { GeoResolveRequest, GeoResolveResponse } from "./types";

/**
 * Resolve a place name or bounding box into a validated geographic
 * representation (center point + bounding box) via the backend.
 */
export async function resolveLocation(
  request: GeoResolveRequest,
  signal?: AbortSignal,
): Promise<GeoResolveResponse> {
  // Coordinates go straight to the map camera; a missing one would place the
  // viewport at NaN rather than fail.
  return asResolveResponse(
    await apiRequest<unknown>("/api/v1/geospatial/resolve", {
      method: "POST",
      body: request,
      signal,
    }),
  );
}
