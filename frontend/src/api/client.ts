/** Minimal typed HTTP client for the SatQuery backend. */

import { config } from "../config/env";
import type { ApiErrorBody } from "./types";

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;

  constructor(message: string, status: number, code = "http_error") {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

export interface RequestOptions extends Omit<RequestInit, "body"> {
  body?: unknown;
}

/**
 * Perform a JSON request against the backend and return the parsed body.
 * Throws {@link ApiError} on network failure or non-2xx responses.
 */
export async function apiRequest<T>(
  path: string,
  options: RequestOptions = {},
): Promise<T> {
  const { body, headers, ...rest } = options;
  const url = `${config.apiBaseUrl}${path.startsWith("/") ? path : `/${path}`}`;

  let response: Response;
  try {
    response = await fetch(url, {
      ...rest,
      headers: {
        Accept: "application/json",
        ...(body !== undefined ? { "Content-Type": "application/json" } : {}),
        ...headers,
      },
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
  } catch {
    throw new ApiError(`Network request to ${url} failed`, 0, "network_error");
  }

  const text = await response.text();
  const parsed: unknown = text ? JSON.parse(text) : null;

  if (!response.ok) {
    const errBody = parsed as ApiErrorBody | null;
    throw new ApiError(
      errBody?.error?.message ?? `Request failed with ${response.status}`,
      response.status,
      errBody?.error?.code ?? "http_error",
    );
  }

  return parsed as T;
}
