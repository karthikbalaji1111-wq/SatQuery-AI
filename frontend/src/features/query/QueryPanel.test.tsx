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

// A 1x1 PNG (payload is opaque to the component - it only builds a data URI).
const PNG_B64 =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

function imageryResponse(sceneId: string) {
  return {
    scene_id: sceneId,
    bbox: { west: 80.1, south: 12.9, east: 80.3, north: 13.2 },
    asset: "visual",
    asset_href: "https://sentinel-cogs.s3.us-west-2.amazonaws.com/x/TCI.tif",
    width: 128,
    height: 96,
    format: "png",
    media_type: "image/png",
    bands: ["red", "green", "blue"],
    crs: "EPSG:32644",
    resolution: 10,
    normalization: "none (source is 8-bit RGB)",
    window: { col_off: 1328, row_off: 5830, width: 436, height: 445 },
    source_shape: [10980, 10980],
    image_base64: PNG_B64,
  };
}

async function resolveSearchAndAwaitScene() {
  await resolveAndAwaitBbox();
  fillDates();
  fireEvent.click(
    screen.getByRole("button", { name: /search sentinel-2 scenes/i }),
  );
  await waitFor(() =>
    expect(screen.getByText("S2B_44PLA_20240715_0_L2A")).toBeInTheDocument(),
  );
}

describe("QueryPanel - bounded imagery", () => {
  it("shows a Load image button for each scene", async () => {
    stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/satellite/search": { body: searchResponse([SCENE]) },
    });
    render(<QueryPanel />);

    await resolveSearchAndAwaitScene();

    expect(
      screen.getByRole("button", { name: /load image/i }),
    ).toBeInTheDocument();
  });

  it("requests bounded imagery with the selected scene id and resolved bbox", async () => {
    const fetchMock = stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/satellite/search": { body: searchResponse([SCENE]) },
      "/satellite/imagery": { body: imageryResponse(SCENE.id) },
    });
    render(<QueryPanel />);

    await resolveSearchAndAwaitScene();
    fireEvent.click(screen.getByRole("button", { name: /load image/i }));

    await waitFor(() =>
      expect(screen.getByRole("img")).toBeInTheDocument(),
    );

    const call = fetchMock.mock.calls.find((c) =>
      String(c[0]).includes("/satellite/imagery"),
    );
    expect(call).toBeDefined();
    expect(String(call![0])).toBe(
      "http://localhost:8000/api/v1/satellite/imagery",
    );
    expect(JSON.parse((call![1] as RequestInit).body as string)).toEqual({
      scene_id: "S2B_44PLA_20240715_0_L2A",
      bbox: { west: 80.1, south: 12.9, east: 80.3, north: 13.2 },
    });
  });

  it("shows a loading state while imagery is retrieved", async () => {
    let release!: (value: Response) => void;
    const pending = new Promise<Response>((resolve) => {
      release = resolve;
    });
    stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/satellite/search": { body: searchResponse([SCENE]) },
      "/satellite/imagery": () => pending,
    });
    render(<QueryPanel />);

    await resolveSearchAndAwaitScene();
    fireEvent.click(screen.getByRole("button", { name: /load image/i }));

    await waitFor(() =>
      expect(
        screen.getByText("Retrieving bounded RGB window…"),
      ).toBeInTheDocument(),
    );
    expect(
      screen.getByRole("button", { name: /loading image…/i }),
    ).toBeDisabled();

    release({
      ok: true,
      status: 200,
      text: () => Promise.resolve(JSON.stringify(imageryResponse(SCENE.id))),
    } as Response);

    await waitFor(() => expect(screen.getByRole("img")).toBeInTheDocument());
  });

  it("displays the returned RGB image and its metadata", async () => {
    stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/satellite/search": { body: searchResponse([SCENE]) },
      "/satellite/imagery": { body: imageryResponse(SCENE.id) },
    });
    render(<QueryPanel />);

    await resolveSearchAndAwaitScene();
    fireEvent.click(screen.getByRole("button", { name: /load image/i }));

    const img = await screen.findByRole("img");
    expect(img).toHaveAttribute("src", `data:image/png;base64,${PNG_B64}`);
    expect(img).toHaveAttribute("width", "128");
    expect(img).toHaveAttribute("height", "96");
    expect(
      screen.getByText(/visual · 128×96px · EPSG:32644/),
    ).toBeInTheDocument();
    expect(screen.getByText(/window 436×445 of 10980×10980/)).toBeInTheDocument();
  });

  it("shows a backend error when imagery retrieval fails", async () => {
    stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/satellite/search": { body: searchResponse([SCENE]) },
      "/satellite/imagery": {
        body: {
          error: {
            code: "invalid_input",
            message: "The requested bbox does not intersect the selected scene.",
          },
        },
        ok: false,
        status: 422,
      },
    });
    render(<QueryPanel />);

    await resolveSearchAndAwaitScene();
    fireEvent.click(screen.getByRole("button", { name: /load image/i }));

    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent(
        "The requested bbox does not intersect the selected scene.",
      ),
    );
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
  });
});

const PLAN_BBOX = { west: 80.14, south: 12.85, east: 80.33, north: 13.24 };

function planResponse(intent: Record<string, unknown>) {
  return { intent, bbox: PLAN_BBOX };
}

function setPlace(value: string) {
  fireEvent.change(screen.getByLabelText("Place name"), { target: { value } });
}

const SINGLE_INTENT = {
  location_query: "Chennai",
  temporal_mode: "single",
  time_windows: [{ start_date: "2024-07-01", end_date: "2024-07-01" }],
  modalities: ["sentinel-2-optical"],
  task: "visualize",
};

describe("QueryPanel - query plan", () => {
  it("renders the intent controls with sensible defaults", () => {
    stubRouter({ "/geospatial/resolve": { body: CHENNAI } });
    render(<QueryPanel />);

    expect(screen.getByLabelText("Place name")).toBeInTheDocument();
    expect(screen.getByLabelText("Single date")).toBeChecked();
    expect(screen.getByLabelText("Compare dates")).not.toBeChecked();
    expect(screen.getByLabelText("Sentinel-2 Optical")).toBeChecked();
    expect(screen.getByLabelText("Sentinel-1 SAR")).not.toBeChecked();

    const taskSelect = screen.getByLabelText("Task") as HTMLSelectElement;
    expect(taskSelect.value).toBe("visualize");
    // Phase 14.1: still selectable (the backend answers them with
    // status "not_implemented"), but labelled so neither is mistaken for a
    // capability the system has.
    expect(
      screen.getByRole("option", { name: "Change Detection (unavailable)" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("option", {
        name: "Object Identification (unavailable)",
      }),
    ).toBeInTheDocument();
  });

  it("reveals baseline/target fields only in compare mode", () => {
    stubRouter({ "/geospatial/resolve": { body: CHENNAI } });
    render(<QueryPanel />);

    expect(screen.queryByLabelText("Baseline start")).not.toBeInTheDocument();
    fireEvent.click(screen.getByLabelText("Compare dates"));
    expect(screen.getByLabelText("Baseline start")).toBeInTheDocument();
    expect(screen.getByLabelText("Baseline end")).toBeInTheDocument();
    expect(screen.getByLabelText("Target start")).toBeInTheDocument();
    expect(screen.getByLabelText("Target end")).toBeInTheDocument();
  });

  it("keeps Build Query Plan disabled until place + temporal + modality are set", () => {
    stubRouter({ "/geospatial/resolve": { body: CHENNAI } });
    render(<QueryPanel />);

    const button = screen.getByRole("button", { name: /build query plan/i });
    expect(button).toBeDisabled();

    setPlace("Chennai");
    expect(button).toBeDisabled(); // no observation date yet

    fireEvent.change(screen.getByLabelText("Observation date"), {
      target: { value: "2024-07-01" },
    });
    expect(button).toBeEnabled();

    fireEvent.click(screen.getByLabelText("Sentinel-2 Optical")); // uncheck the only modality
    expect(button).toBeDisabled();
    expect(
      screen.getByText("Select at least one modality."),
    ).toBeInTheDocument();
  });

  it("posts a single-mode intent and displays the resolved plan", async () => {
    const fetchMock = stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/query/build-plan": { body: planResponse(SINGLE_INTENT) },
    });
    render(<QueryPanel />);

    setPlace("Chennai");
    fireEvent.change(screen.getByLabelText("Observation date"), {
      target: { value: "2024-07-01" },
    });
    fireEvent.click(screen.getByRole("button", { name: /build query plan/i }));

    await waitFor(() =>
      expect(
        screen.getByText("W 80.1400, S 12.8500, E 80.3300, N 13.2400"),
      ).toBeInTheDocument(),
    );
    expect(screen.getByText("2024-07-01 → 2024-07-01")).toBeInTheDocument();
    expect(screen.getByText("sentinel-2-optical")).toBeInTheDocument();

    const call = fetchMock.mock.calls.find((c) =>
      String(c[0]).includes("/query/build-plan"),
    );
    expect(String(call![0])).toBe(
      "http://localhost:8000/api/v1/query/build-plan",
    );
    expect(JSON.parse((call![1] as RequestInit).body as string)).toEqual({
      location_query: "Chennai",
      temporal_mode: "single",
      time_windows: [{ start_date: "2024-07-01", end_date: "2024-07-01" }],
      modalities: ["sentinel-2-optical"],
      task: "visualize",
    });
  });

  it("posts a compare-mode intent with nested baseline/target windows", async () => {
    const fetchMock = stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/query/build-plan": {
        body: planResponse({
          ...SINGLE_INTENT,
          temporal_mode: "compare",
          time_windows: {
            baseline: { start_date: "2023-01-01", end_date: "2023-03-31" },
            target: { start_date: "2024-01-01", end_date: "2024-03-31" },
          },
        }),
      },
    });
    render(<QueryPanel />);

    setPlace("Chennai");
    fireEvent.click(screen.getByLabelText("Compare dates"));
    fireEvent.change(screen.getByLabelText("Baseline start"), {
      target: { value: "2023-01-01" },
    });
    fireEvent.change(screen.getByLabelText("Baseline end"), {
      target: { value: "2023-03-31" },
    });
    fireEvent.change(screen.getByLabelText("Target start"), {
      target: { value: "2024-01-01" },
    });
    fireEvent.change(screen.getByLabelText("Target end"), {
      target: { value: "2024-03-31" },
    });
    fireEvent.click(screen.getByLabelText("Sentinel-1 SAR"));
    fireEvent.change(screen.getByLabelText("Task"), {
      target: { value: "change_detection" },
    });
    fireEvent.click(screen.getByRole("button", { name: /build query plan/i }));

    await waitFor(() =>
      expect(screen.getByText(/baseline: 2023-01-01/)).toBeInTheDocument(),
    );

    const call = fetchMock.mock.calls.find((c) =>
      String(c[0]).includes("/query/build-plan"),
    );
    expect(JSON.parse((call![1] as RequestInit).body as string)).toEqual({
      location_query: "Chennai",
      temporal_mode: "compare",
      time_windows: {
        baseline: { start_date: "2023-01-01", end_date: "2023-03-31" },
        target: { start_date: "2024-01-01", end_date: "2024-03-31" },
      },
      modalities: ["sentinel-2-optical", "sentinel-1-sar"],
      task: "change_detection",
    });
  });

  it("shows a loading state while the plan is built", async () => {
    let release!: (value: Response) => void;
    const pending = new Promise<Response>((resolve) => {
      release = resolve;
    });
    stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/query/build-plan": () => pending,
    });
    render(<QueryPanel />);

    setPlace("Chennai");
    fireEvent.change(screen.getByLabelText("Observation date"), {
      target: { value: "2024-07-01" },
    });
    fireEvent.click(screen.getByRole("button", { name: /build query plan/i }));

    await waitFor(() =>
      expect(screen.getByRole("button", { name: /building…/i })).toBeDisabled(),
    );

    release({
      ok: true,
      status: 200,
      text: () => Promise.resolve(JSON.stringify(planResponse(SINGLE_INTENT))),
    } as Response);

    await waitFor(() =>
      expect(screen.getByText("Resolved bounding box")).toBeInTheDocument(),
    );
  });

  it("shows a backend error when plan building fails", async () => {
    stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/query/build-plan": {
        body: {
          error: {
            code: "not_found",
            message: "No matching location was found.",
          },
        },
        ok: false,
        status: 404,
      },
    });
    render(<QueryPanel />);

    setPlace("nowhere-xyz");
    fireEvent.change(screen.getByLabelText("Observation date"), {
      target: { value: "2024-07-01" },
    });
    fireEvent.click(screen.getByRole("button", { name: /build query plan/i }));

    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent(
        "No matching location was found.",
      ),
    );
  });

  it("leaves the STAC discovery + imagery flow working alongside the plan form", async () => {
    stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/satellite/search": { body: searchResponse([SCENE]) },
      "/satellite/imagery": { body: imageryResponse(SCENE.id) },
    });
    render(<QueryPanel />);

    await resolveSearchAndAwaitScene();
    fireEvent.click(screen.getByRole("button", { name: /load image/i }));
    expect(await screen.findByRole("img")).toBeInTheDocument();
  });
});

const COMPARE_INTENT = {
  location_query: "Rotterdam",
  temporal_mode: "compare",
  time_windows: {
    baseline: { start_date: "2023-06-01", end_date: "2023-06-30" },
    target: { start_date: "2024-06-01", end_date: "2024-06-30" },
  },
  modalities: ["sentinel-2-optical", "sentinel-1-sar"],
  task: "change_detection",
};

function typeNl(value: string) {
  fireEvent.change(screen.getByLabelText("Natural Language Request"), {
    target: { value },
  });
}

describe("QueryPanel - natural language parsing", () => {
  it("shows the NL textarea and disables Parse until text is entered", () => {
    stubRouter({ "/geospatial/resolve": { body: CHENNAI } });
    render(<QueryPanel />);

    expect(
      screen.getByLabelText("Natural Language Request"),
    ).toBeInTheDocument();
    const button = screen.getByRole("button", { name: /parse request/i });
    expect(button).toBeDisabled();

    typeNl("show me chennai");
    expect(button).toBeEnabled();
  });

  it("populates the Query Plan form from a parsed single-mode intent", async () => {
    stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/query/parse": { body: SINGLE_INTENT },
    });
    render(<QueryPanel />);

    typeNl("optical imagery of Chennai on 2024-07-01");
    fireEvent.click(screen.getByRole("button", { name: /parse request/i }));

    await waitFor(() =>
      expect(
        (screen.getByLabelText("Place name") as HTMLInputElement).value,
      ).toBe("Chennai"),
    );
    expect(screen.getByLabelText("Single date")).toBeChecked();
    expect(
      (screen.getByLabelText("Observation date") as HTMLInputElement).value,
    ).toBe("2024-07-01");
    expect(screen.getByLabelText("Sentinel-2 Optical")).toBeChecked();
    expect(screen.getByLabelText("Sentinel-1 SAR")).not.toBeChecked();
    expect((screen.getByLabelText("Task") as HTMLSelectElement).value).toBe(
      "visualize",
    );
    expect(screen.getByText(/Parsed intent: single/)).toBeInTheDocument();
  });

  it("populates compare fields, modalities and task from a compare-mode intent", async () => {
    stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/query/parse": { body: COMPARE_INTENT },
    });
    render(<QueryPanel />);

    typeNl("compare Rotterdam optical and radar, 2023 vs 2024");
    fireEvent.click(screen.getByRole("button", { name: /parse request/i }));

    await waitFor(() =>
      expect(screen.getByLabelText("Compare dates")).toBeChecked(),
    );
    expect(
      (screen.getByLabelText("Baseline start") as HTMLInputElement).value,
    ).toBe("2023-06-01");
    expect(
      (screen.getByLabelText("Target end") as HTMLInputElement).value,
    ).toBe("2024-06-30");
    expect(screen.getByLabelText("Sentinel-1 SAR")).toBeChecked();
    expect((screen.getByLabelText("Task") as HTMLSelectElement).value).toBe(
      "change_detection",
    );
  });

  it("does not auto-run Build Query Plan after parsing", async () => {
    const fetchMock = stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/query/parse": { body: SINGLE_INTENT },
      "/query/build-plan": { body: planResponse(SINGLE_INTENT) },
    });
    render(<QueryPanel />);

    typeNl("show chennai");
    fireEvent.click(screen.getByRole("button", { name: /parse request/i }));

    await waitFor(() =>
      expect(screen.getByText(/Parsed intent:/)).toBeInTheDocument(),
    );

    expect(
      fetchMock.mock.calls.some((c) =>
        String(c[0]).includes("/query/build-plan"),
      ),
    ).toBe(false);
    expect(screen.queryByText("Resolved bounding box")).not.toBeInTheDocument();
  });

  it("shows a loading state while parsing", async () => {
    let release!: (value: Response) => void;
    const pending = new Promise<Response>((resolve) => {
      release = resolve;
    });
    stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/query/parse": () => pending,
    });
    render(<QueryPanel />);

    typeNl("show chennai");
    fireEvent.click(screen.getByRole("button", { name: /parse request/i }));

    await waitFor(() =>
      expect(screen.getByRole("button", { name: /parsing…/i })).toBeDisabled(),
    );

    release({
      ok: true,
      status: 200,
      text: () => Promise.resolve(JSON.stringify(SINGLE_INTENT)),
    } as Response);

    await waitFor(() =>
      expect(screen.getByText(/Parsed intent:/)).toBeInTheDocument(),
    );
  });

  it("shows a backend error when parsing fails", async () => {
    stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/query/parse": {
        body: {
          error: { code: "invalid_input", message: "prompt must not be empty" },
        },
        ok: false,
        status: 422,
      },
    });
    render(<QueryPanel />);

    typeNl("   x");
    fireEvent.click(screen.getByRole("button", { name: /parse request/i }));

    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent(
        "prompt must not be empty",
      ),
    );
  });
});

function executionResult(overrides: Record<string, unknown> = {}) {
  return {
    plan: { intent: SINGLE_INTENT, bbox: PLAN_BBOX },
    executed_modalities: ["sentinel-2-optical"],
    skipped_modalities: [],
    windows: [
      {
        modality: "sentinel-2-optical",
        label: "single",
        time_range: { start_date: "2024-07-01", end_date: "2024-07-01" },
        scene_count: 1,
        scenes: [SCENE],
        selected_scene_id: "S2B_44PLA_20240715_0_L2A",
        imagery: null,
        imagery_error: null,
      },
    ],
    catalog: "https://earth-search.aws.element84.com/v1",
    ...overrides,
  };
}

const S1_WINDOW = {
  modality: "sentinel-1-sar",
  label: "single",
  time_range: { start_date: "2024-07-01", end_date: "2024-07-01" },
  scene_count: 1,
  scenes: [SCENE],
  selected_scene_id: "S1A_IW_GRDH_20240701",
  imagery: null,
  imagery_error: null,
};

describe("QueryPanel - run full query", () => {
  it("keeps Run full query disabled until place + temporal + modality are set", () => {
    stubRouter({ "/geospatial/resolve": { body: CHENNAI } });
    render(<QueryPanel />);

    const button = screen.getByRole("button", { name: /run full query/i });
    expect(button).toBeDisabled();

    setPlace("Chennai");
    fireEvent.change(screen.getByLabelText("Observation date"), {
      target: { value: "2024-07-01" },
    });
    expect(button).toBeEnabled();
  });

  it("sends the current intent to /query/execute and renders per-window results", async () => {
    const fetchMock = stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/query/execute": { body: executionResult() },
    });
    render(<QueryPanel />);

    setPlace("Chennai");
    fireEvent.change(screen.getByLabelText("Observation date"), {
      target: { value: "2024-07-01" },
    });
    fireEvent.click(screen.getByRole("button", { name: /run full query/i }));

    await waitFor(() =>
      expect(
        screen.getByRole("heading", { name: /sentinel-2-optical/ }),
      ).toBeInTheDocument(),
    );
    expect(screen.getByText("S2B_44PLA_20240715_0_L2A")).toBeInTheDocument();
    expect(screen.getByText("2024-07-01 → 2024-07-01")).toBeInTheDocument();

    const call = fetchMock.mock.calls.find((c) =>
      String(c[0]).includes("/query/execute"),
    );
    expect(String(call![0])).toBe("http://localhost:8000/api/v1/query/execute");
    expect(JSON.parse((call![1] as RequestInit).body as string)).toEqual({
      intent: {
        location_query: "Chennai",
        temporal_mode: "single",
        time_windows: [{ start_date: "2024-07-01", end_date: "2024-07-01" }],
        modalities: ["sentinel-2-optical"],
        task: "visualize",
      },
      include_imagery: false,
    });
  });

  it("requests bounded imagery when the checkbox is set and renders the preview", async () => {
    const withImagery = executionResult();
    (withImagery.windows[0] as Record<string, unknown>).imagery =
      imageryResponse("S2B_44PLA_20240715_0_L2A");
    const fetchMock = stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/query/execute": { body: withImagery },
    });
    render(<QueryPanel />);

    setPlace("Chennai");
    fireEvent.change(screen.getByLabelText("Observation date"), {
      target: { value: "2024-07-01" },
    });
    fireEvent.click(
      screen.getByLabelText("Include bounded imagery preview"),
    );
    fireEvent.click(screen.getByRole("button", { name: /run full query/i }));

    expect(await screen.findByRole("img")).toBeInTheDocument();
    const call = fetchMock.mock.calls.find((c) =>
      String(c[0]).includes("/query/execute"),
    );
    expect(
      JSON.parse((call![1] as RequestInit).body as string).include_imagery,
    ).toBe(true);
  });

  it("executes Sentinel-1 alongside Sentinel-2 and keeps the manual Build Query Plan flow", async () => {
    // Two windows share the temporal label "single"; the composite React key
    // must keep them distinct (no duplicate-key warning).
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/query/execute": {
        body: executionResult({
          executed_modalities: ["sentinel-2-optical", "sentinel-1-sar"],
          windows: [
            {
              modality: "sentinel-2-optical",
              label: "single",
              time_range: { start_date: "2024-07-01", end_date: "2024-07-01" },
              scene_count: 1,
              scenes: [SCENE],
              selected_scene_id: "S2B_44PLA_20240715_0_L2A",
              imagery: null,
              imagery_error: null,
            },
            S1_WINDOW,
          ],
        }),
      },
      "/query/build-plan": { body: planResponse(SINGLE_INTENT) },
    });
    render(<QueryPanel />);

    setPlace("Chennai");
    fireEvent.change(screen.getByLabelText("Observation date"), {
      target: { value: "2024-07-01" },
    });
    fireEvent.click(screen.getByLabelText("Sentinel-1 SAR"));
    fireEvent.click(screen.getByRole("button", { name: /run full query/i }));

    await waitFor(() =>
      expect(
        screen.getByRole("heading", { name: /sentinel-1-sar/ }),
      ).toBeInTheDocument(),
    );
    expect(
      screen.getByRole("heading", { name: /sentinel-2-optical/ }),
    ).toBeInTheDocument();
    expect(screen.getByText("S1A_IW_GRDH_20240701")).toBeInTheDocument();
    expect(errorSpy).not.toHaveBeenCalled();

    // Manual Build Query Plan still works.
    fireEvent.click(screen.getByRole("button", { name: /build query plan/i }));
    await waitFor(() =>
      expect(screen.getByText("Resolved bounding box")).toBeInTheDocument(),
    );
  });

  it("shows a backend error when execution fails", async () => {
    stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/query/execute": {
        body: {
          error: {
            code: "upstream_error",
            message: "The satellite catalog timed out.",
          },
        },
        ok: false,
        status: 502,
      },
    });
    render(<QueryPanel />);

    setPlace("Chennai");
    fireEvent.change(screen.getByLabelText("Observation date"), {
      target: { value: "2024-07-01" },
    });
    fireEvent.click(screen.getByRole("button", { name: /run full query/i }));

    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent(
        "The satellite catalog timed out.",
      ),
    );
  });
});

function analysisResult(overrides: Record<string, unknown> = {}) {
  return {
    status: "ok",
    task: "visualize",
    answer:
      "Retrieved 1 window(s) for 'Chennai' from " +
      "https://earth-search.aws.element84.com/v1: sentinel-2-optical single " +
      "(2024-07-01 to 2024-07-01) -> S2B_44PLA_20240715_0_L2A.",
    windows_considered: [
      {
        modality: "sentinel-2-optical",
        label: "single",
        time_range: { start_date: "2024-07-01", end_date: "2024-07-01" },
        selected_scene_id: "S2B_44PLA_20240715_0_L2A",
      },
    ],
    warnings: [],
    measurements: [],
    ...overrides,
  };
}

function runFullQuery() {
  setPlace("Chennai");
  fireEvent.change(screen.getByLabelText("Observation date"), {
    target: { value: "2024-07-01" },
  });
  fireEvent.click(screen.getByRole("button", { name: /run full query/i }));
}

describe("QueryPanel - analysis", () => {
  it("chains /query/analyze after a successful execute and sends the execution", async () => {
    const execution = executionResult();
    const fetchMock = stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/query/execute": { body: execution },
      "/query/analyze": { body: analysisResult() },
    });
    render(<QueryPanel />);

    runFullQuery();

    await waitFor(() =>
      expect(screen.getByRole("heading", { name: "Analysis" })).toBeInTheDocument(),
    );

    const call = fetchMock.mock.calls.find((c) =>
      String(c[0]).includes("/query/analyze"),
    );
    expect(call).toBeDefined();
    expect(String(call![0])).toBe("http://localhost:8000/api/v1/query/analyze");
    // The whole execution result, and nothing else - no separate task field.
    expect(JSON.parse((call![1] as RequestInit).body as string)).toEqual({
      execution,
    });
  });

  it("renders the analysis answer, status and window traceability", async () => {
    stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/query/execute": { body: executionResult() },
      "/query/analyze": { body: analysisResult() },
    });
    render(<QueryPanel />);

    runFullQuery();

    await waitFor(() =>
      expect(screen.getByRole("heading", { name: "Analysis" })).toBeInTheDocument(),
    );
    expect(screen.getByText(/visualize · ok/)).toBeInTheDocument();
    expect(screen.getByText(/Retrieved 1 window\(s\) for 'Chennai'/)).toBeInTheDocument();
    expect(
      screen.getByText(/sentinel-2-optical · single .*S2B_44PLA_20240715_0_L2A/),
    ).toBeInTheDocument();
  });

  it("renders a not_implemented analysis without claiming a result", async () => {
    stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/query/execute": { body: executionResult() },
      "/query/analyze": {
        body: analysisResult({
          status: "not_implemented",
          task: "change_detection",
          answer:
            "The 'change_detection' analysis is not implemented in this phase.",
        }),
      },
    });
    render(<QueryPanel />);

    runFullQuery();

    await waitFor(() =>
      expect(screen.getByText(/change_detection · not_implemented/)).toBeInTheDocument(),
    );
    expect(screen.getByText(/not implemented in this phase/)).toBeInTheDocument();
  });

  it("renders analysis warnings", async () => {
    stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/query/execute": { body: executionResult() },
      "/query/analyze": {
        body: analysisResult({
          warnings: ["No scene was selected for the sentinel-1-sar window 'single'."],
        }),
      },
    });
    render(<QueryPanel />);

    runFullQuery();

    await waitFor(() =>
      expect(
        screen.getByText(/Warning: No scene was selected/),
      ).toBeInTheDocument(),
    );
  });

  it("keeps the execution result visible when the analysis fails", async () => {
    stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/query/execute": { body: executionResult() },
      "/query/analyze": {
        body: {
          error: { code: "invalid_input", message: "analysis rejected the body" },
        },
        ok: false,
        status: 422,
      },
    });
    render(<QueryPanel />);

    runFullQuery();

    await waitFor(() =>
      expect(
        screen.getByText(/Analysis failed: analysis rejected the body/),
      ).toBeInTheDocument(),
    );

    // Failure isolation: the execution result must still be rendered.
    expect(
      screen.getByRole("heading", { name: /sentinel-2-optical/ }),
    ).toBeInTheDocument();
    expect(screen.getByText("S2B_44PLA_20240715_0_L2A")).toBeInTheDocument();
    expect(
      screen.queryByRole("heading", { name: "Analysis" }),
    ).not.toBeInTheDocument();
  });

  it("does not analyze when the execution itself fails", async () => {
    const fetchMock = stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/query/execute": {
        body: {
          error: { code: "upstream_error", message: "catalog down" },
        },
        ok: false,
        status: 502,
      },
      "/query/analyze": { body: analysisResult() },
    });
    render(<QueryPanel />);

    runFullQuery();

    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent("catalog down"),
    );
    expect(
      fetchMock.mock.calls.some((c) => String(c[0]).includes("/query/analyze")),
    ).toBe(false);
  });
});

const NDWI_MEASUREMENTS = [
  { name: "ndwi_valid_pixel_count", value: 1234, unit: "pixels" },
  { name: "ndwi_mean", value: 0.2777, unit: "index" },
  { name: "ndwi_min", value: -0.41, unit: "index" },
  { name: "ndwi_max", value: 0.5, unit: "index" },
  { name: "ndwi_percent_above_index_threshold_0.3", value: 12.5, unit: "%" },
];

function enableNdwi() {
  fireEvent.click(
    screen.getByLabelText("Compute NDWI index statistics (Sentinel-2)"),
  );
}

function analyzeBody(fetchMock: ReturnType<typeof stubRouter>) {
  const call = fetchMock.mock.calls.find((c) =>
    String(c[0]).includes("/query/analyze"),
  );
  expect(call).toBeDefined();
  return JSON.parse((call![1] as RequestInit).body as string);
}

describe("QueryPanel - NDWI opt-in", () => {
  it("offers an unchecked NDWI control by default", () => {
    stubRouter({ "/geospatial/resolve": { body: CHENNAI } });
    render(<QueryPanel />);

    expect(
      screen.getByLabelText("Compute NDWI index statistics (Sentinel-2)"),
    ).not.toBeChecked();
  });

  it("omits include_ndwi entirely when the control is off", async () => {
    const execution = executionResult();
    const fetchMock = stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/query/execute": { body: execution },
      "/query/analyze": { body: analysisResult() },
    });
    render(<QueryPanel />);

    runFullQuery();
    await waitFor(() =>
      expect(screen.getByRole("heading", { name: "Analysis" })).toBeInTheDocument(),
    );

    // Byte-identical to the pre-NDWI request: the key is absent, not false.
    const body = analyzeBody(fetchMock);
    expect(body).toEqual({ execution });
    expect("include_ndwi" in body).toBe(false);
  });

  it("sends include_ndwi=true when the control is on", async () => {
    const execution = executionResult();
    const fetchMock = stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/query/execute": { body: execution },
      "/query/analyze": {
        body: analysisResult({ measurements: NDWI_MEASUREMENTS }),
      },
    });
    render(<QueryPanel />);

    setPlace("Chennai");
    fireEvent.change(screen.getByLabelText("Observation date"), {
      target: { value: "2024-07-01" },
    });
    enableNdwi();
    fireEvent.click(screen.getByRole("button", { name: /run full query/i }));

    await waitFor(() =>
      expect(screen.getByRole("heading", { name: "Analysis" })).toBeInTheDocument(),
    );
    expect(analyzeBody(fetchMock)).toEqual({ execution, include_ndwi: true });
  });

  it("renders the NDWI measurements returned by the backend", async () => {
    stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/query/execute": { body: executionResult() },
      "/query/analyze": {
        body: analysisResult({ measurements: NDWI_MEASUREMENTS }),
      },
    });
    render(<QueryPanel />);

    setPlace("Chennai");
    fireEvent.change(screen.getByLabelText("Observation date"), {
      target: { value: "2024-07-01" },
    });
    enableNdwi();
    fireEvent.click(screen.getByRole("button", { name: /run full query/i }));

    await waitFor(() =>
      expect(screen.getByText("ndwi_mean")).toBeInTheDocument(),
    );
    for (const name of NDWI_MEASUREMENTS.map((m) => m.name)) {
      expect(screen.getByText(name)).toBeInTheDocument();
    }
    expect(screen.getByText("0.2777 index")).toBeInTheDocument();
    expect(screen.getByText("1,234 pixels")).toBeInTheDocument();
    expect(screen.getByText("12.5 %")).toBeInTheDocument();
    // The index threshold is never presented as water/flood detection.
    expect(screen.queryByText(/water detect/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/flood/i)).not.toBeInTheDocument();
  });

  it("still renders the analysis when NDWI produced no measurements", async () => {
    stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/query/execute": { body: executionResult() },
      "/query/analyze": {
        body: analysisResult({
          measurements: [],
          warnings: ["NDWI was requested but no Sentinel-2 window was available."],
        }),
      },
    });
    render(<QueryPanel />);

    setPlace("Chennai");
    fireEvent.change(screen.getByLabelText("Observation date"), {
      target: { value: "2024-07-01" },
    });
    enableNdwi();
    fireEvent.click(screen.getByRole("button", { name: /run full query/i }));

    await waitFor(() =>
      expect(
        screen.getByText(/Warning: NDWI was requested but no Sentinel-2 window/),
      ).toBeInTheDocument(),
    );
    expect(screen.getByRole("heading", { name: "Analysis" })).toBeInTheDocument();
  });

  it("does not change the /query/execute request", async () => {
    const fetchMock = stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/query/execute": { body: executionResult() },
      "/query/analyze": { body: analysisResult() },
    });
    render(<QueryPanel />);

    setPlace("Chennai");
    fireEvent.change(screen.getByLabelText("Observation date"), {
      target: { value: "2024-07-01" },
    });
    enableNdwi();
    fireEvent.click(screen.getByRole("button", { name: /run full query/i }));

    await waitFor(() =>
      expect(screen.getByRole("heading", { name: "Analysis" })).toBeInTheDocument(),
    );
    const call = fetchMock.mock.calls.find((c) =>
      String(c[0]).includes("/query/execute"),
    );
    // NDWI is an analysis concern only - execution is untouched.
    expect(JSON.parse((call![1] as RequestInit).body as string)).toEqual({
      intent: {
        location_query: "Chennai",
        temporal_mode: "single",
        time_windows: [{ start_date: "2024-07-01", end_date: "2024-07-01" }],
        modalities: ["sentinel-2-optical"],
        task: "visualize",
      },
      include_imagery: false,
    });
  });
});

const TEMPORAL_COMPARISON = {
  first: {
    window_label: "baseline",
    scene_id: "S2B_44PLA_20240115_0_L2A",
    acquired_at: "2024-01-15T05:12:34Z",
    cloud_cover: 3.1,
    measurements: [
      { name: "ndwi_valid_pixel_count", value: 1000, unit: "pixels" },
      { name: "ndwi_mean", value: -0.42, unit: "index" },
    ],
  },
  second: {
    window_label: "target",
    scene_id: "S2B_44PLA_20240715_0_L2A",
    acquired_at: "2024-07-15T05:12:34Z",
    cloud_cover: 8.4,
    measurements: [
      { name: "ndwi_valid_pixel_count", value: 1000, unit: "pixels" },
      { name: "ndwi_mean", value: 0.18, unit: "index" },
    ],
  },
  compatibility: {
    same_modality: true,
    temporal_separation_days: 182,
    bbox_overlap: "full",
    crs_match: "unknown",
    resolution_match: "unknown",
    processing_level_match: "same",
    limitations: ["This report is derived from metadata only."],
    co_registration_status: "not_evaluated",
  },
  differences: [{ name: "mean_ndwi_difference", value: 0.6, unit: "index" }],
  warnings: ["mean_ndwi_difference is the difference between two aggregates."],
};

function enableTemporalNdwi() {
  fireEvent.click(
    screen.getByLabelText("Compute temporal NDWI statistics (two Sentinel-2 dates)"),
  );
}

describe("QueryPanel - temporal NDWI opt-in", () => {
  it("offers an unchecked temporal NDWI control by default", () => {
    stubRouter({ "/geospatial/resolve": { body: CHENNAI } });
    render(<QueryPanel />);

    expect(
      screen.getByLabelText(
        "Compute temporal NDWI statistics (two Sentinel-2 dates)",
      ),
    ).not.toBeChecked();
  });

  it("omits include_temporal_ndwi entirely when the control is off", async () => {
    const execution = executionResult();
    const fetchMock = stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/query/execute": { body: execution },
      "/query/analyze": { body: analysisResult() },
    });
    render(<QueryPanel />);

    runFullQuery();
    await waitFor(() =>
      expect(screen.getByRole("heading", { name: "Analysis" })).toBeInTheDocument(),
    );

    const body = analyzeBody(fetchMock);
    expect(body).toEqual({ execution });
    expect("include_temporal_ndwi" in body).toBe(false);
  });

  it("sends include_temporal_ndwi=true when the control is on", async () => {
    const execution = executionResult();
    const fetchMock = stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/query/execute": { body: execution },
      "/query/analyze": {
        body: analysisResult({ temporal_comparison: TEMPORAL_COMPARISON }),
      },
    });
    render(<QueryPanel />);

    enableTemporalNdwi();
    runFullQuery();
    await waitFor(() =>
      expect(screen.getByRole("heading", { name: "Analysis" })).toBeInTheDocument(),
    );

    expect(analyzeBody(fetchMock)).toEqual({
      execution,
      include_temporal_ndwi: true,
    });
  });

  it("renders both observations, the difference and the limitations", async () => {
    stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/query/execute": { body: executionResult() },
      "/query/analyze": {
        body: analysisResult({ temporal_comparison: TEMPORAL_COMPARISON }),
      },
    });
    render(<QueryPanel />);

    enableTemporalNdwi();
    runFullQuery();
    await waitFor(() =>
      expect(
        screen.getByRole("heading", { name: "Temporal NDWI Statistics" }),
      ).toBeInTheDocument(),
    );

    expect(screen.getByText("baseline")).toBeInTheDocument();
    expect(screen.getByText("target")).toBeInTheDocument();
    expect(screen.getByText("Mean NDWI Difference")).toBeInTheDocument();
    expect(
      screen.getByText(/This report is derived from metadata only/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/difference between two aggregates/),
    ).toBeInTheDocument();
  });

  it("renders no temporal section when the backend returns none", async () => {
    stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/query/execute": { body: executionResult() },
      "/query/analyze": { body: analysisResult({ temporal_comparison: null }) },
    });
    render(<QueryPanel />);

    runFullQuery();
    await waitFor(() =>
      expect(screen.getByRole("heading", { name: "Analysis" })).toBeInTheDocument(),
    );

    expect(
      screen.queryByRole("heading", { name: "Temporal NDWI Statistics" }),
    ).not.toBeInTheDocument();
  });
});

// ===========================================================================
// Phase 14.1 - presentation fixes
//
// Formatting is a PRESENTATION concern only: the API representation and the
// backend values are unchanged, and these tests assert the rendered text.
// ===========================================================================

const UGLY_COMPARISON = {
  ...TEMPORAL_COMPARISON,
  first: {
    ...TEMPORAL_COMPARISON.first,
    cloud_cover: 3.14159265,
    measurements: [
      { name: "ndwi_valid_pixel_count", value: 1234567, unit: "pixels" },
      { name: "ndwi_mean", value: -0.5794832165498732, unit: "index" },
      {
        name: "ndwi_percent_above_index_threshold_0.3",
        value: 44.03472222222222,
        unit: "%",
      },
    ],
  },
  compatibility: {
    ...TEMPORAL_COMPARISON.compatibility,
    temporal_separation_days: 44.03472222222222,
  },
  differences: [
    { name: "mean_ndwi_difference", value: -0.5794832165498732, unit: "index" },
  ],
};

describe("QueryPanel - numeric presentation", () => {
  it("formats index, percentage, count and day values readably", async () => {
    stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/query/execute": { body: executionResult() },
      "/query/analyze": {
        body: analysisResult({ temporal_comparison: UGLY_COMPARISON }),
      },
    });
    render(<QueryPanel />);

    enableTemporalNdwi();
    runFullQuery();
    await waitFor(() =>
      expect(
        screen.getByRole("heading", { name: "Temporal NDWI Statistics" }),
      ).toBeInTheDocument(),
    );

    // Machine precision must not reach the user.
    expect(screen.queryByText(/0\.5794832165498732/)).not.toBeInTheDocument();
    expect(screen.queryByText(/44\.03472222222222/)).not.toBeInTheDocument();

    // Index values keep useful precision.
    expect(screen.getAllByText(/-0\.5795/).length).toBeGreaterThan(0);
    // Percentages to one decimal.
    expect(screen.getByText(/44\.0\s*%/)).toBeInTheDocument();
    // Counts as integers with separators.
    expect(screen.getByText(/1,234,567/)).toBeInTheDocument();
    // Day intervals to sensible precision.
    expect(screen.getByText(/44\.0 days apart/)).toBeInTheDocument();
    // Cloud cover to one decimal.
    expect(screen.getByText(/3\.1%/)).toBeInTheDocument();
  });

  it("formats single-scene NDWI measurements too", async () => {
    stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/query/execute": { body: executionResult() },
      "/query/analyze": {
        body: analysisResult({
          measurements: [
            { name: "ndwi_mean", value: 0.27770000123456, unit: "index" },
            { name: "ndwi_valid_pixel_count", value: 1234567, unit: "pixels" },
          ],
        }),
      },
    });
    render(<QueryPanel />);

    enableNdwi();
    runFullQuery();
    await waitFor(() =>
      expect(screen.getByRole("heading", { name: "Analysis" })).toBeInTheDocument(),
    );

    expect(screen.queryByText(/0\.27770000123456/)).not.toBeInTheDocument();
    expect(screen.getByText(/0\.2777/)).toBeInTheDocument();
    expect(screen.getByText(/1,234,567/)).toBeInTheDocument();
  });
});

describe("QueryPanel - unsupported task labelling", () => {
  it("marks change detection and object identification as unavailable", () => {
    stubRouter({ "/geospatial/resolve": { body: CHENNAI } });
    render(<QueryPanel />);

    const select = screen.getByLabelText("Task") as HTMLSelectElement;
    const labels = Array.from(select.options).map((o) => o.text);

    expect(labels.some((l) => /Visualize/.test(l))).toBe(true);
    // The user must not read "Change Detection" as something the system does.
    expect(
      labels.some((l) => /Change Detection.*(unavailable|not implemented)/i.test(l)),
    ).toBe(true);
    expect(
      labels.some((l) =>
        /Object Identification.*(unavailable|not implemented)/i.test(l),
      ),
    ).toBe(true);
  });
});

describe("QueryPanel - map imagery handoff", () => {
  // The map draws whatever imagery it was last handed. Resetting the panel's
  // own imagery state is not enough: without telling the parent, the map keeps
  // showing a scene from a location the user has already moved on from.

  it("clears map imagery when a new place is resolved", async () => {
    stubRouter({ "/geospatial/resolve": { body: CHENNAI } });
    const onImagery = vi.fn();
    render(<QueryPanel onImagery={onImagery} />);

    resolvePlace();

    await waitFor(() => expect(onImagery).toHaveBeenCalledWith(null));
  });

  it("clears map imagery when a new scene search starts", async () => {
    stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/satellite/search": { body: searchResponse([SCENE]) },
    });
    const onImagery = vi.fn();
    render(<QueryPanel onImagery={onImagery} />);

    await resolveAndAwaitBbox();
    onImagery.mockClear();
    fillDates();
    fireEvent.click(
      screen.getByRole("button", { name: /search sentinel-2 scenes/i }),
    );

    await waitFor(() => expect(onImagery).toHaveBeenCalledWith(null));
  });

  it("still hands over the imagery on a successful preview", async () => {
    const imagery = imageryResponse(SCENE.id);
    stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/satellite/search": { body: searchResponse([SCENE]) },
      "/satellite/imagery": { body: imagery },
    });
    const onImagery = vi.fn();
    render(<QueryPanel onImagery={onImagery} />);

    await resolveSearchAndAwaitScene();
    onImagery.mockClear();
    fireEvent.click(screen.getByRole("button", { name: /load image/i }));

    await waitFor(() => expect(onImagery).toHaveBeenCalledWith(imagery));
  });

  it("works without the callback", async () => {
    stubRouter({ "/geospatial/resolve": { body: CHENNAI } });
    expect(() => render(<QueryPanel />)).not.toThrow();
    resolvePlace();
    await waitFor(() =>
      expect(screen.getByText("Chennai, Tamil Nadu, India")).toBeInTheDocument(),
    );
  });
});

describe("QueryPanel - a new preview invalidates the old one", () => {
  it("clears map imagery when a different preview starts", async () => {
    const imagery = imageryResponse(SCENE.id);
    stubRouter({
      "/geospatial/resolve": { body: CHENNAI },
      "/satellite/search": { body: searchResponse([SCENE]) },
      "/satellite/imagery": { body: imagery },
    });
    const onImagery = vi.fn();
    render(<QueryPanel onImagery={onImagery} />);

    await resolveSearchAndAwaitScene();
    onImagery.mockClear();
    fireEvent.click(screen.getByRole("button", { name: /load image/i }));

    // Cleared first, so the map never shows a stale scene while the new one
    // is still in flight.
    expect(onImagery).toHaveBeenCalledWith(null);
    await waitFor(() => expect(onImagery).toHaveBeenLastCalledWith(imagery));
  });
});
