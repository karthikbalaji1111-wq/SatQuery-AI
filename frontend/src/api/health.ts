import { ApiError, apiRequest } from "./client";
import { config } from "../config/env";
import type { HealthResponse, ReadinessResponse } from "./types";

/** Fetch backend liveness status. */
export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return apiRequest<HealthResponse>("/health", { signal });
}

/**
 * Fetch backend readiness - whether the deployment can actually do its work.
 *
 * Deliberately NOT `apiRequest`. Readiness answers **503 when not ready**, so
 * that orchestration reads the status code; but that 503 is the ANSWER, not a
 * transport failure, and its body carries the explanation. `apiRequest` throws
 * on any non-2xx and would discard exactly the information this call exists to
 * retrieve.
 */
export async function getReadiness(
  signal?: AbortSignal,
): Promise<ReadinessResponse> {
  const url = `${config.apiBaseUrl}/ready`;

  let response: Response;
  try {
    response = await fetch(url, {
      headers: { Accept: "application/json" },
      signal,
    });
  } catch {
    throw new ApiError(`Network request to ${url} failed`, 0, "network_error");
  }

  // 200 = ready, 503 = not ready. Anything else is a genuine failure of the
  // probe itself rather than a verdict from it.
  if (response.status !== 200 && response.status !== 503) {
    throw new ApiError(
      `Request failed with ${response.status}`,
      response.status,
      "http_error",
    );
  }

  const text = await response.text();
  try {
    return JSON.parse(text) as ReadinessResponse;
  } catch {
    throw new ApiError(
      "The readiness probe returned a non-JSON response.",
      response.status,
      "invalid_response",
    );
  }
}
