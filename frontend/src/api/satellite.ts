import { apiRequest } from "./client";
import type { SceneSearchRequest, SceneSearchResponse } from "./types";

/**
 * Discover Sentinel-2 L2A scenes (STAC metadata only) for a bounding box and
 * date range via the backend. No imagery is fetched.
 */
export function searchScenes(
  request: SceneSearchRequest,
  signal?: AbortSignal,
): Promise<SceneSearchResponse> {
  return apiRequest<SceneSearchResponse>("/api/v1/satellite/search", {
    method: "POST",
    body: request,
    signal,
  });
}
