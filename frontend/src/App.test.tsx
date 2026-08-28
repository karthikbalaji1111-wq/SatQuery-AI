import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("App", () => {
  it("renders the title and panels", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      text: () => Promise.resolve("{}"),
    } as Response));

    render(<App />);

    expect(screen.getByRole("heading", { name: "SatQuery", level: 1 })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Ask" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Map" })).toBeInTheDocument();

    // Let the BackendStatus effect settle so state updates stay wrapped in act().
    await waitFor(() =>
      expect(screen.queryByText(/Checking backend/i)).not.toBeInTheDocument(),
    );
  });

  it("shows backend status once health resolves", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              status: "ok",
              service: "SatQuery API",
              version: "0.1.0",
              environment: "test",
            }),
          ),
      } as Response),
    );

    render(<App />);

    await waitFor(() =>
      expect(screen.getByText(/Backend online/i)).toBeInTheDocument(),
    );
  });
});
