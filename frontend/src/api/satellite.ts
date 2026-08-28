import { apiRequest } from "./client";
import type {
  ImageryRequest,
  ImageryResponse,
  SceneSearchRequest,
  SceneSearchResponse,
} from "./types";

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

/**
 * Retrieve a bounded RGB window for an already-discovered scene. The backend
 * reads only the requested spatial window - no full-scene download.
 */
export function fetchSceneImagery(
  request: ImageryRequest,
  signal?: AbortSignal,
): Promise<ImageryResponse> {
  return apiRequest<ImageryResponse>("/api/v1/satellite/imagery", {
    method: "POST",
    body: request,
    signal,
  });
}
