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
    expect(
      screen.getByRole("option", { name: "Change Detection" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("option", { name: "Object Identification" }),
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
