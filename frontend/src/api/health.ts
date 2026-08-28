import { apiRequest } from "./client";
import type { HealthResponse } from "./types";

/** Fetch backend liveness status. */
export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return apiRequest<HealthResponse>("/health", { signal });
}
