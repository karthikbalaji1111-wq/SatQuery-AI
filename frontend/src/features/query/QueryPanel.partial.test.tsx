/**
 * A partial run must read as partial.
 *
 * Execution used to abort on the first catalog failure, so there was nothing
 * partial to render. Now a window that could not be searched is carried beside
 * the windows that succeeded, and the interface has to say which is which -
 * otherwise the reader sees "Scenes found: 0" and concludes the archive holds
 * nothing there, when in fact nothing was ever asked.
 *
 * Also pinned here: a mixed run names EVERY catalog that answered. Sentinel-2
 * comes from Earth Search and Sentinel-1 RTC from the Planetary Computer, and
 * naming one implies the other's data came from it.
 */

import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { QueryPanel } from "./QueryPanel";

afterEach(() => {
  vi.restoreAllMocks();
});

const BBOX = { west: 80.1, south: 12.9, east: 80.3, north: 13.2 };
const EARTH_SEARCH = "https://earth-search.aws.element84.com/v1";
const PLANETARY = "https://planetarycomputer.microsoft.com/api/stac/v1";
const OUTAGE = "The satellite catalog is unavailable.";

const INTENT = {
  location_query: "Chennai",
  temporal_mode: "single",
  time_windows: [{ start_date: "2024-07-01", end_date: "2024-07-01" }],
  modalities: ["sentinel-2-optical"],
  task: "visualize",
};

const SCENE = {
  id: "S2B_44PLA_20240715_0_L2A",
  datetime: "2024-07-15T05:12:34Z",
  bbox: BBOX,
  geometry: null,
  cloud_cover: 12.3,
  collection: "sentinel-2-l2a",
  platform: "sentinel-2b",
  processing_level: "L2A",
  thumbnail_url: null,
  assets: [],
};

function window_(overrides: Record<string, unknown> = {}) {
  return {
    modality: "sentinel-2-optical",
    label: "single",
    time_range: { start_date: "2024-07-01", end_date: "2024-07-01" },
    scene_count: 1,
    scenes: [SCENE],
    selected_scene_id: SCENE.id,
    imagery: null,
    imagery_error: null,
    catalog: EARTH_SEARCH,
    error: null,
    ...overrides,
  };
}

function partialExecution() {
  return {
    plan: { intent: INTENT, bbox: BBOX },
    executed_modalities: ["sentinel-2-optical", "sentinel-1-sar"],
    skipped_modalities: [],
    windows: [
      window_(),
      window_({
        modality: "sentinel-1-sar",
        scene_count: 0,
        scenes: [],
        selected_scene_id: null,
        catalog: null,
        error: OUTAGE,
      }),
    ],
    catalog: EARTH_SEARCH,
    catalogs: [EARTH_SEARCH],
    status: "partial",
  };
}

const ANALYSIS = {
  status: "ok",
  task: "visualize",
  answer: "Retrieved 1 window(s).",
  windows_considered: [],
  warnings: [],
  measurements: [],
};

function stubRouter(routes: Record<string, unknown>) {
  const fn = vi.fn().mockImplementation((url: string) => {
    const key = Object.keys(routes).find((k) => url.includes(k));
    if (key === undefined) return Promise.reject(new Error(`no route: ${url}`));
    return Promise.resolve({
      ok: true,
      status: 200,
      text: () => Promise.resolve(JSON.stringify(routes[key])),
    } as Response);
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

async function runQuery(execution: unknown) {
  stubRouter({
    "/geospatial/resolve": {
      query_type: "place",
      display_name: "Chennai",
      center: { lat: 13.08, lon: 80.27 },
      bbox: BBOX,
      source: "nominatim",
    },
    "/query/execute": execution,
    "/query/analyze": ANALYSIS,
  });
  render(<QueryPanel />);

  fireEvent.change(screen.getByLabelText("Place name"), {
    target: { value: "Chennai" },
  });
  fireEvent.change(screen.getByLabelText("Observation date"), {
    target: { value: "2024-07-01" },
  });
  fireEvent.click(screen.getByRole("button", { name: /run full query/i }));

  await waitFor(() =>
    expect(
      screen.getByRole("heading", { name: /sentinel-2-optical/ }),
    ).toBeInTheDocument(),
  );
}

describe("QueryPanel - partial execution", () => {
  it("says how much of the run succeeded", async () => {
    await runQuery(partialExecution());

    expect(screen.getByText(/Partial result: 1 of 2 windows/)).toBeInTheDocument();
  });

  it("keeps the completed window's results", async () => {
    await runQuery(partialExecution());

    // The whole point of partial results: what succeeded is still reported.
    expect(
      screen.getAllByText("S2B_44PLA_20240715_0_L2A").length,
    ).toBeGreaterThan(0);
  });

  it("distinguishes a catalog outage from an empty archive", async () => {
    await runQuery(partialExecution());

    const notice = screen.getByRole("alert");
    expect(notice).toHaveTextContent(OUTAGE);
    expect(notice).toHaveTextContent(/not a finding that the archive is empty/);
  });

  it("says nothing about partial results when every window succeeded", async () => {
    const complete = partialExecution();
    complete.windows = [window_()];
    complete.executed_modalities = ["sentinel-2-optical"];
    complete.status = "completed";
    await runQuery(complete);

    expect(screen.queryByText(/Partial result:/)).not.toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("names every catalog a mixed run actually used", async () => {
    const mixed = partialExecution();
    mixed.windows = [
      window_(),
      window_({ modality: "sentinel-1-sar", catalog: PLANETARY }),
    ];
    mixed.catalogs = [EARTH_SEARCH, PLANETARY];
    mixed.status = "completed";
    await runQuery(mixed);

    const summary = screen.getByText(/^Executed:/);
    expect(summary).toHaveTextContent(EARTH_SEARCH);
    expect(summary).toHaveTextContent(PLANETARY);
    expect(summary).toHaveTextContent("sources:");
  });
});
