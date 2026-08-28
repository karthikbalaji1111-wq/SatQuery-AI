import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { QueryPanel } from "./QueryPanel";

afterEach(() => {
  vi.restoreAllMocks();
});

const CHENNAI = {
  query_type: "place",
  display_name: "Chennai, Tamil Nadu, India",
  center: { lat: 13.0837, lon: 80.2702 },
  bbox: { west: 80.1, south: 12.9, east: 80.3, north: 13.2 },
  source: "nominatim",
};

const SCENE = {
  id: "S2B_44PLA_20240715_0_L2A",
  datetime: "2024-07-15T05:12:34Z",
  bbox: { west: 80.1, south: 12.85, east: 80.42, north: 13.22 },
  geometry: null,
  cloud_cover: 12.3,
  collection: "sentinel-2-l2a",
  platform: "sentinel-2b",
  processing_level: "L2A",
  thumbnail_url: "https://example.test/thumb.jpg",
  assets: [],
};

function searchResponse(scenes: unknown[]) {
  return {
    query: {
      collections: ["sentinel-2-l2a"],
      bbox: [80.1, 12.9, 80.3, 13.2],
      datetime: "2024-06-01T00:00:00Z/2024-08-31T23:59:59Z",
      max_cloud_cover: null,
      limit: 10,
      filter: null,
    },
    scene_count: scenes.length,
    scenes,
    catalog: "https://earth-search.aws.element84.com/v1",
  };
}

type RouteResult =
  | { body: unknown; ok?: boolean; status?: number }
  | (() => Promise<Response>);

/** Route fetch calls to canned responses by URL substring. */
function stubRouter(routes: Record<string, RouteResult>) {
  const fn = vi.fn().mockImplementation((url: string) => {
    const key = Object.keys(routes).find((k) => url.includes(k));
    if (key === undefined) return Promise.reject(new Error(`no route: ${url}`));
    const route = routes[key];
    if (typeof route === "function") return route();
    return Promise.resolve({
      ok: route.ok ?? true,
      status: route.status ?? 200,
      text: () => Promise.resolve(JSON.stringify(route.body)),
    } as Response);
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

function resolvePlace(value = "Chennai") {
  fireEvent.change(screen.getByLabelText("Place name"), { target: { value } });
  fireEvent.click(screen.getByRole("button", { name: /resolve location/i }));
}

async function resolveAndAwaitBbox() {
  resolvePlace();
  await waitFor(() =>
    expect(screen.getByText("Chennai, Tamil Nadu, India")).toBeInTheDocument(),
  );
}

function fillDates(start = "2024-06-01", end = "2024-08-31") {
  fireEvent.change(screen.getByLabelText("Start date"), {
    target: { value: start },
  });
  fireEvent.change(screen.getByLabelText("End date"), { target: { value: end } });
}

describe("QueryPanel - geospatial resolve", () => {
  it("resolves a place name and shows coordinates + bounding box", async () => {
    stubRouter({ "/geospatial/resolve": { body: CHENNAI } });
    render(<QueryPanel />);

    resolvePlace();

    await waitFor(() =>
      expect(screen.getByText("Chennai, Tamil Nadu, India")).toBeInTheDocument(),
    );
    expect(screen.getByText("13.08370, 80.27020")).toBeInTheDocument();
    expect(
      screen.getByText("W 80.1000, S 12.9000, E 80.3000, N 13.2000"),
    ).toBeInTheDocument();
  });

  it("shows the backend error message when resolve fails", async () => {
    stubRouter({
      "/geospatial/resolve": {
        body: {
          error: { code: "not_found", message: "No matching location was found." },
        },
        ok: false,
        status: 404,
      },
    });
    render(<QueryPanel />);

    resolvePlace("nowhere-xyz");

    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent(
        "No matching location was found.",
      ),
    );
  });

  it("keeps the resolve button disabled until a place is entered", () => {
    stubRouter({ "/geospatial/resolve": { body: CHENNAI } });
    render(<QueryPanel />);

    expect(
      screen.getByRole("button", { name: /resolve location/i }),
    ).toBeDisabled();
  });
});

describe("QueryPanel - Sentinel-2 scene search", () => {
  it("runs the full resolve -> search flow and renders scenes", async () => {
    const fetchMock = stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/satellite/search": { body: searchResponse([SCENE]) },
    });
    render(<QueryPanel />);

    await resolveAndAwaitBbox();
    fillDates();
    fireEvent.click(
      screen.getByRole("button", { name: /search sentinel-2 scenes/i }),
    );

    await waitFor(() =>
      expect(screen.getByText("S2B_44PLA_20240715_0_L2A")).toBeInTheDocument(),
    );
    expect(screen.getByText(/1 scene ·/)).toBeInTheDocument();
    expect(screen.getByText("2024-07-15T05:12:34Z")).toBeInTheDocument();
    expect(screen.getByText("12.3%")).toBeInTheDocument();
    expect(screen.getByText("sentinel-2b")).toBeInTheDocument();
    expect(screen.getByText("L2A")).toBeInTheDocument();
    expect(
      screen.getByText("W 80.1000, S 12.8500, E 80.4200, N 13.2200"),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: "https://example.test/thumb.jpg" }),
    ).toBeInTheDocument();

    const searchCall = fetchMock.mock.calls.find((c) =>
      String(c[0]).includes("/satellite/search"),
    );
    expect(searchCall).toBeDefined();
    expect(JSON.parse((searchCall![1] as RequestInit).body as string)).toEqual({
      bbox: { west: 80.1, south: 12.9, east: 80.3, north: 13.2 },
      start_date: "2024-06-01",
      end_date: "2024-08-31",
    });
  });

  it("forwards max cloud cover when provided", async () => {
    const fetchMock = stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/satellite/search": { body: searchResponse([SCENE]) },
    });
    render(<QueryPanel />);

    await resolveAndAwaitBbox();
    fillDates();
    fireEvent.change(screen.getByLabelText("Max cloud %"), {
      target: { value: "20" },
    });
    fireEvent.click(
      screen.getByRole("button", { name: /search sentinel-2 scenes/i }),
    );

    await waitFor(() =>
      expect(screen.getByText("S2B_44PLA_20240715_0_L2A")).toBeInTheDocument(),
    );
    const searchCall = fetchMock.mock.calls.find((c) =>
      String(c[0]).includes("/satellite/search"),
    );
    expect(
      JSON.parse((searchCall![1] as RequestInit).body as string).max_cloud_cover,
    ).toBe(20);
  });

  it("shows a loading state while the search is in flight", async () => {
    let release!: (value: Response) => void;
    const pending = new Promise<Response>((resolve) => {
      release = resolve;
    });
    stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/satellite/search": () => pending,
    });
    render(<QueryPanel />);

    await resolveAndAwaitBbox();
    fillDates();
    fireEvent.click(
      screen.getByRole("button", { name: /search sentinel-2 scenes/i }),
    );

    await waitFor(() =>
      expect(screen.getByRole("button", { name: /searching…/i })).toBeDisabled(),
    );

    release({
      ok: true,
      status: 200,
      text: () => Promise.resolve(JSON.stringify(searchResponse([SCENE]))),
    } as Response);

    await waitFor(() =>
      expect(screen.getByText("S2B_44PLA_20240715_0_L2A")).toBeInTheDocument(),
    );
  });

  it("shows an error when the scene search fails", async () => {
    stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/satellite/search": {
        body: {
          error: { code: "upstream_error", message: "The satellite catalog timed out." },
        },
        ok: false,
        status: 502,
      },
    });
    render(<QueryPanel />);

    await resolveAndAwaitBbox();
    fillDates();
    fireEvent.click(
      screen.getByRole("button", { name: /search sentinel-2 scenes/i }),
    );

    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent(
        "The satellite catalog timed out.",
      ),
    );
  });

  it("renders an empty-results message when no scenes match", async () => {
    stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/satellite/search": { body: searchResponse([]) },
    });
    render(<QueryPanel />);

    await resolveAndAwaitBbox();
    fillDates();
    fireEvent.click(
      screen.getByRole("button", { name: /search sentinel-2 scenes/i }),
    );

    await waitFor(() =>
      expect(
        screen.getByText(
          "No Sentinel-2 scenes found for this area and date range.",
        ),
      ).toBeInTheDocument(),
    );
  });

  it("disables search until a bbox is resolved and valid dates are set", async () => {
    stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/satellite/search": { body: searchResponse([SCENE]) },
    });
    render(<QueryPanel />);

    // No search form before resolve.
    expect(
      screen.queryByRole("button", { name: /search sentinel-2 scenes/i }),
    ).not.toBeInTheDocument();

    await resolveAndAwaitBbox();
    const searchButton = screen.getByRole("button", {
      name: /search sentinel-2 scenes/i,
    });

    // Dates missing -> disabled.
    expect(searchButton).toBeDisabled();

    // Reversed dates -> disabled + hint.
    fillDates("2024-08-31", "2024-06-01");
    expect(searchButton).toBeDisabled();
    expect(
      screen.getByText("Start date must be on or before end date."),
    ).toBeInTheDocument();

    // Valid dates -> enabled.
    fillDates("2024-06-01", "2024-08-31");
    expect(searchButton).toBeEnabled();
  });
});
