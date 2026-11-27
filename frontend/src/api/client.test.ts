import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, apiRequest, describeApiError } from "./client";

afterEach(() => {
  vi.restoreAllMocks();
});

function mockFetch(response: Partial<Response> & { text: () => Promise<string> }) {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue(response as Response),
  );
}

describe("apiRequest", () => {
  it("parses a successful JSON response", async () => {
    mockFetch({
      ok: true,
      status: 200,
      text: () => Promise.resolve(JSON.stringify({ status: "ok" })),
    });

    await expect(apiRequest("/health")).resolves.toEqual({ status: "ok" });
  });

  it("throws ApiError with the backend error code on non-2xx", async () => {
    mockFetch({
      ok: false,
      status: 501,
      text: () =>
        Promise.resolve(
          JSON.stringify({ error: { code: "not_implemented", message: "nope" } }),
        ),
    });

    await expect(apiRequest("/x")).rejects.toMatchObject({
      code: "not_implemented",
      status: 501,
    });
  });

  it("wraps network failures in an ApiError", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("boom")));

    const error = await apiRequest("/health").catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).code).toBe("network_error");
  });
});

describe("request ids", () => {
  it("keeps the server's request id on an error and shows it as a reference", async () => {
    mockFetch({
      ok: false,
      status: 504,
      headers: new Headers({ "x-request-id": "0123456789ab" }),
      text: () =>
        Promise.resolve(
          JSON.stringify({ error: { code: "workflow_timeout", message: "Too slow." } }),
        ),
    });

    const error = await apiRequest("/x").catch((cause: unknown) => cause);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).requestId).toBe("0123456789ab");
    expect(describeApiError(error as ApiError)).toBe("Too slow. (reference 0123456789ab)");
  });

  it("drops a header value the server could not have issued", async () => {
    mockFetch({
      ok: false,
      status: 500,
      headers: new Headers({ "x-request-id": "<script>alert(1)</script>" }),
      text: () => Promise.resolve(JSON.stringify({ error: { code: "x", message: "Boom." } })),
    });

    const error = (await apiRequest("/x").catch((cause: unknown) => cause)) as ApiError;
    expect(error.requestId).toBeNull();
    expect(describeApiError(error)).toBe("Boom.");
  });
});
