/**
 * The client must survive a body it did not expect.
 *
 * `apiRequest` parsed every non-empty body with `JSON.parse` and no guard. A
 * response that is not JSON - an HTML error page from a proxy or a single-page
 * host, a gateway's plain-text message, a truncated body - therefore threw a
 * raw `SyntaxError` out of the client. Every caller here tests
 * `instanceof ApiError`, so that error travelled past all of them and surfaced
 * as "Unexpected error": no status, no code, and indistinguishable from a bug
 * in the application itself.
 *
 * The HTML case is not hypothetical. A production deployment that serves the
 * frontend and proxies the API answers an unmatched path with `index.html`, so
 * a misrouted request returns 200 and a page of markup.
 *
 * What these pin: the failure is always an `ApiError`, the HTTP status is
 * preserved because it is the useful fact, and a well-formed error envelope is
 * still read exactly as before.
 */

import { describe, expect, it, vi, afterEach } from "vitest";

import { ApiError, apiRequest } from "./client";

afterEach(() => {
  vi.restoreAllMocks();
});

function respondWith(body: string, { ok = true, status = 200 } = {}) {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok,
      status,
      text: () => Promise.resolve(body),
    } as Response),
  );
}

const HTML = "<!DOCTYPE html><html><body>Bad Gateway</body></html>";

describe("apiRequest - unexpected bodies", () => {
  it("turns an HTML error page into an ApiError carrying the status", async () => {
    respondWith(HTML, { ok: false, status: 502 });

    const error = await apiRequest("/query/agent").catch((e: unknown) => e);

    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).status).toBe(502);
    expect((error as SyntaxError).name).not.toBe("SyntaxError");
  });

  it("turns an HTML body served with 200 into an ApiError", async () => {
    // The SPA-fallback case: the request was misrouted and the host answered
    // with the application's own index page.
    respondWith(HTML);

    const error = await apiRequest("/query/agent").catch((e: unknown) => e);

    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).code).toBe("invalid_response");
    expect((error as ApiError).message).toMatch(/non-JSON/);
  });

  it("turns a plain-text gateway message into an ApiError", async () => {
    respondWith("upstream connect error", { ok: false, status: 503 });

    const error = await apiRequest("/query/agent").catch((e: unknown) => e);

    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).status).toBe(503);
  });

  it("accepts an empty successful body as no content", async () => {
    // Unchanged behaviour, pinned so the guard above cannot alter it.
    respondWith("");

    await expect(apiRequest("/health")).resolves.toBeNull();
  });

  it("reports an empty error body by its status", async () => {
    respondWith("", { ok: false, status: 500 });

    const error = await apiRequest("/query/agent").catch((e: unknown) => e);

    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).status).toBe(500);
  });

  it("still reads a well-formed error envelope", async () => {
    // Non-vacuity: the guard must not swallow the server's own message.
    respondWith(
      JSON.stringify({
        error: { code: "upstream_error", message: "The catalog is unavailable." },
      }),
      { ok: false, status: 502 },
    );

    const error = await apiRequest("/query/execute").catch((e: unknown) => e);

    expect((error as ApiError).code).toBe("upstream_error");
    expect((error as ApiError).message).toBe("The catalog is unavailable.");
  });

  it("still returns a well-formed success body", async () => {
    respondWith(JSON.stringify({ status: "ok" }));

    await expect(apiRequest("/health")).resolves.toEqual({ status: "ok" });
  });
});
