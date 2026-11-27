/** Minimal typed HTTP client for the SatQuery backend. */

import { config } from "../config/env";
import type { ApiErrorBody } from "./types";

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  /** The server's `X-Request-ID`, when the response carried one. */
  readonly requestId: string | null;

  constructor(
    message: string,
    status: number,
    code = "http_error",
    requestId: string | null = null,
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.requestId = requestId;
  }
}

/**
 * An API failure as one sentence for the UI, with the server's request id
 * when there is one - the reference an operator needs to find the run's log
 * lines.
 */
export function describeApiError(error: ApiError): string {
  return error.requestId
    ? `${error.message} (reference ${error.requestId})`
    : error.message;
}

/** Only an id the server could have issued is shown; anything else is dropped. */
function requestIdOf(response: Response): string | null {
  const id = response.headers?.get?.("x-request-id") ?? null;
  return id !== null && /^[0-9a-f]{12}$/.test(id) ? id : null;
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
  let parsed: unknown = null;
  if (text) {
    try {
      parsed = JSON.parse(text);
    } catch {
      // A body that is not JSON at all: an HTML error page from a proxy or a
      // single-page host, a gateway's own plain-text message, or a truncated
      // response. `JSON.parse` threw a raw SyntaxError straight out of this
      // client, past every caller's `instanceof ApiError` handling, so the UI
      // reported "Unexpected error" with no status - indistinguishable from a
      // bug in the application. The status code is the useful fact here, and
      // it survives.
      throw new ApiError(
        response.ok
          ? `The server returned a non-JSON response (HTTP ${response.status}).`
          : `Request failed with ${response.status}`,
        response.status,
        response.ok ? "invalid_response" : "http_error",
        requestIdOf(response),
      );
    }
  }

  if (!response.ok) {
    const errBody = parsed as ApiErrorBody | null;
    throw new ApiError(
      errBody?.error?.message ?? `Request failed with ${response.status}`,
      response.status,
      errBody?.error?.code ?? "http_error",
      requestIdOf(response),
    );
  }

  return parsed as T;
}
