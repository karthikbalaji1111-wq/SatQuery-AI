/**
 * "Operational" has to mean the deployment can actually work.
 *
 * This component read `/health` and printed "Operational". `/health` answers
 * only "is the process alive?" - so a backend with no credential for its
 * selected provider, or a local model that was never installed, was alive,
 * reachable, incapable of answering any query, and displayed as Operational.
 *
 * Pinned here: a ready backend still reads Operational; an unready one names
 * the missing capability; and readiness that cannot be fetched falls back to
 * the previous wording rather than inventing a verdict, because an unknown
 * capability is not a failed one.
 */

import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { BackendStatus } from "./BackendStatus";

afterEach(() => {
  vi.restoreAllMocks();
});

const HEALTH = {
  status: "ok",
  service: "SatQuery API",
  version: "0.1.0",
  environment: "development",
};

const READY = {
  ready: true,
  service: "SatQuery API",
  version: "0.1.0",
  environment: "development",
  capabilities: [
    { name: "application", ready: true, detail: "SatQuery API 0.1.0." },
    { name: "ai_provider", ready: true, detail: "gemini is configured." },
  ],
};

const NOT_READY = {
  ...READY,
  ready: false,
  capabilities: [
    { name: "application", ready: true, detail: "SatQuery API 0.1.0." },
    {
      name: "ai_provider",
      ready: false,
      detail:
        "local is selected but Ollama is not answering at http://127.0.0.1:11434.",
    },
  ],
};

/** Route by URL, with an explicit status so 503 can be exercised. */
function stub(routes: Record<string, { body?: unknown; status?: number }>) {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockImplementation((url: string) => {
      const key = Object.keys(routes).find((k) => String(url).includes(k));
      if (key === undefined) return Promise.reject(new Error(`no route: ${url}`));
      const route = routes[key];
      const status = route.status ?? 200;
      return Promise.resolve({
        ok: status >= 200 && status < 300,
        status,
        text: () => Promise.resolve(JSON.stringify(route.body ?? {})),
      } as Response);
    }),
  );
}

describe("BackendStatus", () => {
  it("reads Operational when the deployment is ready", async () => {
    stub({ "/health": { body: HEALTH }, "/ready": { body: READY } });
    render(<BackendStatus />);

    expect(await screen.findByText("Operational")).toBeInTheDocument();
  });

  it("names the missing capability instead of claiming Operational", async () => {
    // The readiness endpoint answers 503 when not ready; that is the verdict,
    // not a transport failure, and its body is the explanation.
    stub({
      "/health": { body: HEALTH },
      "/ready": { body: NOT_READY, status: 503 },
    });
    render(<BackendStatus />);

    const status = await screen.findByText(/Degraded/);
    expect(status).toHaveTextContent("ai provider");
    expect(screen.queryByText("Operational")).not.toBeInTheDocument();
    // The server's own reason travels with it.
    expect(status).toHaveAttribute(
      "title",
      expect.stringContaining("Ollama is not answering"),
    );
  });

  it("keeps the previous wording when readiness cannot be determined", async () => {
    // An older backend, or a probe that failed: unknown is not failed.
    stub({ "/health": { body: HEALTH } });
    render(<BackendStatus />);

    expect(await screen.findByText("Operational")).toBeInTheDocument();
  });

  it("still reports an unreachable backend", async () => {
    stub({});
    render(<BackendStatus />);

    await waitFor(() =>
      expect(screen.getByText(/Backend unreachable/)).toBeInTheDocument(),
    );
  });
});
