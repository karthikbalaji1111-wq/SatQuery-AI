import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
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


// --------------------------------------------------------------------------- #
// The App -> QueryPanel -> MapPanel seam
// --------------------------------------------------------------------------- #
//
// QueryPanel owns the request and App holds the result; the map only renders
// what it is handed. These tests exercise that whole path through the real
// components, so a break in the wiring cannot pass by being mocked out.
//
// jsdom has no WebGL2, so the real MapLibre map never starts. The panel still
// reports which scene reached it, which is what makes the seam observable here.

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
  thumbnail_url: null,
  assets: [],
};

const IMAGERY = {
  scene_id: SCENE.id,
  bbox: CHENNAI.bbox,
  asset: "visual",
  asset_href: "https://example.test/TCI.tif",
  width: 128,
  height: 96,
  format: "png",
  media_type: "image/png",
  bands: ["red", "green", "blue"],
  crs: "EPSG:32644",
  resolution: 10,
  normalization: "none (source is 8-bit RGB)",
  window: { col_off: 1328, row_off: 5830, width: 128, height: 96 },
  source_shape: [10980, 10980],
  transform: [10, 0, 421900, 0, -10, 1444560],
  corners_wgs84: [
    [80.279621036, 13.066131716],
    [80.289951131, 13.066160297],
    [80.29002857, 13.039034352],
    [80.279699601, 13.039005833],
  ],
  image_base64: "iVBORw0KGgo=",
};

/** Canned responses by URL substring, mirroring the QueryPanel tests. */
function stubRouter(routes: Record<string, unknown>) {
  const fn = vi.fn((url: string) => {
    const match = Object.keys(routes).find((key) => String(url).includes(key));
    const body = match ? routes[match] : {};
    return Promise.resolve({
      ok: true,
      status: 200,
      text: () => Promise.resolve(JSON.stringify(body)),
    } as Response);
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

async function runQueryToPreview() {
  fireEvent.change(screen.getByLabelText("Place name"), {
    target: { value: "Chennai" },
  });
  fireEvent.click(screen.getByRole("button", { name: /resolve location/i }));
  await waitFor(() =>
    expect(screen.getByText("Chennai, Tamil Nadu, India")).toBeInTheDocument(),
  );

  fireEvent.change(screen.getByLabelText("Start date"), {
    target: { value: "2024-06-01" },
  });
  fireEvent.change(screen.getByLabelText("End date"), {
    target: { value: "2024-08-31" },
  });
  fireEvent.click(
    screen.getByRole("button", { name: /search sentinel-2 scenes/i }),
  );
  await waitFor(() =>
    expect(screen.getByText(SCENE.id)).toBeInTheDocument(),
  );

  fireEvent.click(screen.getByRole("button", { name: /load image/i }));
}

describe("App - query to map seam", () => {
  it("passes a successful preview through to the map panel", async () => {
    stubRouter({
      "/health": { status: "ok", service: "SatQuery API", version: "0.1.0", environment: "test" },
      "/geospatial/resolve": CHENNAI,
      "/satellite/search": { scenes: [SCENE], count: 1, catalog: "https://example.test/v1" },
      "/satellite/imagery": IMAGERY,
    });
    render(<App />);

    await runQueryToPreview();

    // The map panel names the scene it received - proof it crossed the seam.
    const mapPanel = screen
      .getByRole("heading", { name: "Map" })
      .closest("section") as HTMLElement;
    await waitFor(() =>
      expect(within(mapPanel).getByRole("status")).toHaveTextContent(SCENE.id),
    );
  });

  it("clears stale imagery from the map when a new search starts", async () => {
    stubRouter({
      "/health": { status: "ok", service: "SatQuery API", version: "0.1.0", environment: "test" },
      "/geospatial/resolve": CHENNAI,
      "/satellite/search": { scenes: [SCENE], count: 1, catalog: "https://example.test/v1" },
      "/satellite/imagery": IMAGERY,
    });
    render(<App />);

    await runQueryToPreview();
    const mapPanel = screen
      .getByRole("heading", { name: "Map" })
      .closest("section") as HTMLElement;
    await waitFor(() =>
      expect(within(mapPanel).getByRole("status")).toHaveTextContent(SCENE.id),
    );

    // A new search invalidates it: the map must not keep the old scene.
    fireEvent.click(
      screen.getByRole("button", { name: /search sentinel-2 scenes/i }),
    );

    await waitFor(() =>
      expect(within(mapPanel).getByRole("status")).not.toHaveTextContent(
        SCENE.id,
      ),
    );
  });
});

describe("App - NDWI overlay reaches the map", () => {
  const NDWI = {
    scene_id: SCENE.id,
    window_label: "single",
    media_type: "image/png",
    image_base64: "TkRXSQ==",
    width: 2,
    height: 2,
    crs: "EPSG:32644",
    transform: [10, 0, 399960, 0, -10, 1500000],
    corners_wgs84: [
      [80.2, 13.06],
      [80.29, 13.061],
      [80.291, 13.03],
      [80.201, 13.029],
    ],
    value_min: -0.5,
    value_max: 0.8,
    valid_pixel_count: 3,
  };

  const EXECUTION = {
    plan: {
      intent: {
        location_query: "Chennai",
        temporal_mode: "single",
        time_windows: [{ start_date: "2024-07-01", end_date: "2024-07-01" }],
        modalities: ["sentinel-2-optical"],
        task: "visualize",
      },
      bbox: CHENNAI.bbox,
    },
    executed_modalities: ["sentinel-2-optical"],
    skipped_modalities: [],
    windows: [
      {
        modality: "sentinel-2-optical",
        label: "single",
        time_range: { start_date: "2024-07-01", end_date: "2024-07-01" },
        scene_count: 1,
        scenes: [SCENE],
        selected_scene_id: SCENE.id,
        imagery: null,
        imagery_error: null,
      },
    ],
    catalog: "https://example.test/v1",
  };

  function analysis(overlay: unknown) {
    return {
      status: "ok",
      task: "visualize",
      answer: "Analysed.",
      windows_considered: [],
      warnings: [],
      measurements: [],
      temporal_comparison: null,
      ndwi_overlay: overlay,
    };
  }

  async function runNdwiQuery() {
    fireEvent.change(screen.getByLabelText("Place name"), {
      target: { value: "Chennai" },
    });
    fireEvent.change(screen.getByLabelText("Observation date"), {
      target: { value: "2024-07-01" },
    });
    fireEvent.click(
      screen.getByLabelText("Compute NDWI index statistics (Sentinel-2)"),
    );
    fireEvent.click(screen.getByRole("button", { name: /run full query/i }));
  }

  it("passes a produced NDWI overlay through App to the map panel", async () => {
    stubRouter({
      "/health": { status: "ok", service: "SatQuery API", version: "0.1.0", environment: "test" },
      "/geospatial/resolve": CHENNAI,
      "/query/execute": EXECUTION,
      "/query/analyze": analysis(NDWI),
    });
    render(<App />);

    await runNdwiQuery();

    const mapPanel = screen
      .getByRole("heading", { name: "Map" })
      .closest("section") as HTMLElement;
    await waitFor(() =>
      expect(within(mapPanel).getByText(/NDWI/i)).toBeInTheDocument(),
    );
  });

  it("shows no NDWI on the map when the analysis produced none", async () => {
    stubRouter({
      "/health": { status: "ok", service: "SatQuery API", version: "0.1.0", environment: "test" },
      "/geospatial/resolve": CHENNAI,
      "/query/execute": EXECUTION,
      "/query/analyze": analysis(null),
    });
    render(<App />);

    await runNdwiQuery();

    const mapPanel = screen
      .getByRole("heading", { name: "Map" })
      .closest("section") as HTMLElement;
    await waitFor(() =>
      expect(screen.getByRole("heading", { name: "Analysis" })).toBeInTheDocument(),
    );
    expect(within(mapPanel).queryByText(/NDWI/i)).not.toBeInTheDocument();
  });
});

describe("App - temporal change overlay reaches the map", () => {
  const CHANGE_OVERLAY = {
    scene_id: "S2_TARGET",
    window_label: "baseline→target",
    media_type: "image/png",
    image_base64: "Q0hBTkdF",
    width: 2,
    height: 2,
    crs: "EPSG:32644",
    transform: [10, 0, 399960, 0, -10, 1500000],
    corners_wgs84: [
      [80.1, 13.1],
      [80.3, 13.101],
      [80.301, 12.9],
      [80.101, 12.899],
    ],
    value_min: -0.72,
    value_max: 0.81,
    valid_pixel_count: 33600,
  };

  function comparison(overlay: unknown) {
    return {
      first: {
        window_label: "baseline",
        scene_id: "S2_BASE",
        acquired_at: null,
        cloud_cover: null,
        measurements: [],
        transform: null,
      },
      second: {
        window_label: "target",
        scene_id: "S2_TARGET",
        acquired_at: null,
        cloud_cover: null,
        measurements: [],
        transform: null,
      },
      compatibility: {
        same_modality: true,
        temporal_separation_days: 189,
        bbox_overlap: "full",
        crs_match: "unknown",
        resolution_match: "unknown",
        processing_level_match: "same",
        limitations: [],
        co_registration_status: "not_evaluated",
      },
      differences: [],
      change:
        overlay === null
          ? null
          : {
              baseline_scene_id: "S2_BASE",
              target_scene_id: "S2_TARGET",
              baseline_acquired_at: null,
              target_acquired_at: null,
              window_label: "baseline→target",
              paired_valid_pixel_count: 33600,
              change_mean: 0.118,
              change_min: -0.72,
              change_max: 0.81,
              crs: "EPSG:32644",
              transform: [10, 0, 399960, 0, -10, 1500000],
              corners_wgs84: CHANGE_OVERLAY.corners_wgs84,
              overlay,
            },
      warnings: [],
    };
  }

  function analysisWith(change: unknown) {
    return {
      status: "ok",
      task: "visualize",
      answer: "Analysed.",
      windows_considered: [],
      warnings: [],
      measurements: [],
      ndwi_overlay: null,
      spatial_measurement: null,
      temporal_comparison: comparison(change),
    };
  }

  const EXECUTION = {
    plan: {
      intent: {
        location_query: "Chennai",
        temporal_mode: "single",
        time_windows: [{ start_date: "2024-07-01", end_date: "2024-07-01" }],
        modalities: ["sentinel-2-optical"],
        task: "visualize",
      },
      bbox: CHENNAI.bbox,
    },
    executed_modalities: ["sentinel-2-optical"],
    skipped_modalities: [],
    windows: [],
    catalog: "https://example.test/v1",
  };

  async function runQuery(change: unknown) {
    stubRouter({
      "/health": { status: "ok", service: "SatQuery API", version: "0.1.0", environment: "test" },
      "/geospatial/resolve": CHENNAI,
      "/query/execute": EXECUTION,
      "/query/analyze": analysisWith(change),
    });
    render(<App />);
    fireEvent.change(screen.getByLabelText("Place name"), {
      target: { value: "Chennai" },
    });
    fireEvent.change(screen.getByLabelText("Observation date"), {
      target: { value: "2024-07-01" },
    });
    fireEvent.click(screen.getByRole("button", { name: /run full query/i }));
  }

  it("passes a change overlay through App to the map panel", async () => {
    await runQuery(CHANGE_OVERLAY);

    const mapPanel = screen
      .getByRole("heading", { name: "Map" })
      .closest("section") as HTMLElement;
    await waitFor(() =>
      expect(within(mapPanel).getByText(/NDWI change/i)).toBeInTheDocument(),
    );
  });

  it("shows no change on the map when the grids were not comparable", async () => {
    await runQuery(null);

    await waitFor(() =>
      expect(screen.getByRole("heading", { name: "Analysis" })).toBeInTheDocument(),
    );
    const mapPanel = screen
      .getByRole("heading", { name: "Map" })
      .closest("section") as HTMLElement;
    expect(within(mapPanel).queryByText(/NDWI change/i)).not.toBeInTheDocument();
  });
});
