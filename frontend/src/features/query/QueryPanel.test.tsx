import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { QueryPanel } from "./QueryPanel";

afterEach(() => {
  vi.restoreAllMocks();
});

const CHENNAI: unknown = {
  query_type: "place",
  display_name: "Chennai, Tamil Nadu, India",
  center: { lat: 13.0837, lon: 80.2702 },
  bbox: { west: 80.1, south: 12.9, east: 80.3, north: 13.2 },
  source: "nominatim",
};

function stubFetch(body: unknown, ok = true, status = 200) {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok,
      status,
      text: () => Promise.resolve(JSON.stringify(body)),
    } as Response),
  );
}

function submitPlace(value: string) {
  fireEvent.change(screen.getByLabelText("Place name"), { target: { value } });
  fireEvent.click(screen.getByRole("button", { name: /resolve location/i }));
}

describe("QueryPanel", () => {
  it("resolves a place name and shows coordinates + bounding box", async () => {
    stubFetch(CHENNAI);
    render(<QueryPanel />);

    submitPlace("Chennai");

    await waitFor(() =>
      expect(
        screen.getByText("Chennai, Tamil Nadu, India"),
      ).toBeInTheDocument(),
    );
    expect(screen.getByText("13.08370, 80.27020")).toBeInTheDocument();
    expect(
      screen.getByText(/W 80\.1000, S 12\.9000, E 80\.3000, N 13\.2000/),
    ).toBeInTheDocument();

    const fetchMock = fetch as unknown as ReturnType<typeof vi.fn>;
    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/api/v1/geospatial/resolve",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ place: "Chennai" }),
      }),
    );
  });

  it("shows the backend error message on failure", async () => {
    stubFetch(
      { error: { code: "not_found", message: "No matching location was found." } },
      false,
      404,
    );
    render(<QueryPanel />);

    submitPlace("nowhere-xyz");

    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent(
        "No matching location was found.",
      ),
    );
  });

  it("keeps submit disabled until a place is entered", () => {
    stubFetch(CHENNAI);
    render(<QueryPanel />);

    expect(
      screen.getByRole("button", { name: /resolve location/i }),
    ).toBeDisabled();
  });
});
