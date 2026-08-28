import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, apiRequest } from "./client";

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
