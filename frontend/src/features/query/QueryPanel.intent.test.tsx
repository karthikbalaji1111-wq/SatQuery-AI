/**
 * A manual run must execute the request that was parsed - all of it.
 *
 * The form is a REVIEW of a parsed intent, and `currentIntent()` rebuilt that
 * intent from the form's own controls. Anything the form had no control for was
 * therefore dropped between parsing and executing, silently and with a
 * plausible-looking result on the other side:
 *
 *   - a TIME SERIES was collapsed to `windows[0]`, so "monthly analysis,
 *     January through March" executed as January alone;
 *   - an NDWI THRESHOLD ("NDWI above 0.3") never reached the wire, so the
 *     backend - which counts those pixels only when the intent carries the
 *     threshold - answered a plain index question instead.
 *
 * Neither failed. Both answered a different question from the one asked, which
 * is the harder kind to notice.
 *
 * These are contract tests over the whole path - parse -> edit -> serialize ->
 * execute - and they assert on the REQUEST BODY, because that is the only place
 * the preserved intent can be observed. Every expectation is written from
 * literals rather than from the fixture object, so a fix that preserved the
 * shape while losing a value cannot pass.
 */

import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { QueryPanel } from "./QueryPanel";

afterEach(() => {
  vi.restoreAllMocks();
});

const BBOX = { west: 80.1, south: 12.9, east: 80.3, north: 13.2 };

const CHENNAI = {
  query_type: "place",
  display_name: "Chennai, Tamil Nadu, India",
  center: { lat: 13.0837, lon: 80.2702 },
  bbox: BBOX,
  source: "nominatim",
};

/** Three monthly windows and an explicit threshold - the audited example. */
const SERIES_INTENT = {
  location_query: "Chennai",
  temporal_mode: "timeseries",
  time_windows: [
    { start_date: "2025-01-01", end_date: "2025-01-31" },
    { start_date: "2025-02-01", end_date: "2025-02-28" },
    { start_date: "2025-03-01", end_date: "2025-03-31" },
  ],
  modalities: ["sentinel-2-optical"],
  task: "visualize",
  ndwi_threshold: { operator: "gt", value: 0.3 },
};

const PLAIN_INTENT = {
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

function executionResult(intent: Record<string, unknown>) {
  return {
    plan: { intent, bbox: BBOX },
    executed_modalities: ["sentinel-2-optical"],
    skipped_modalities: [],
    windows: [
      {
        modality: "sentinel-2-optical",
        label: "series[0]",
        time_range: { start_date: "2025-01-01", end_date: "2025-01-31" },
        scene_count: 1,
        scenes: [SCENE],
        selected_scene_id: SCENE.id,
        imagery: null,
        imagery_error: null,
      },
    ],
    catalog: "https://earth-search.aws.element84.com/v1",
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

type RouteResult = { body: unknown; ok?: boolean; status?: number };

function stubRouter(routes: Record<string, RouteResult>) {
  const fn = vi.fn().mockImplementation((url: string) => {
    const key = Object.keys(routes).find((k) => url.includes(k));
    if (key === undefined) return Promise.reject(new Error(`no route: ${url}`));
    const route = routes[key];
    return Promise.resolve({
      ok: route.ok ?? true,
      status: route.status ?? 200,
      text: () => Promise.resolve(JSON.stringify(route.body)),
    } as Response);
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

/** The JSON body of the last request whose URL contains `fragment`. */
function lastBody(
  fetchMock: ReturnType<typeof stubRouter>,
  fragment: string,
): Record<string, unknown> {
  const call = [...fetchMock.mock.calls]
    .reverse()
    .find((args) => String(args[0]).includes(fragment));
  if (call === undefined) throw new Error(`no request to ${fragment}`);
  return JSON.parse(String((call[1] as RequestInit | undefined)?.body ?? "{}"));
}

async function parse(intent: Record<string, unknown>, fetchMock: unknown) {
  void fetchMock;
  fireEvent.change(screen.getByLabelText("Natural Language Request"), {
    target: { value: "monthly NDWI above 0.3 for Chennai, January to March" },
  });
  fireEvent.click(screen.getByRole("button", { name: /parse request/i }));
  await waitFor(() =>
    expect((screen.getByLabelText("Place name") as HTMLInputElement).value).toBe(
      intent.location_query,
    ),
  );
}

async function runFullQuery(fetchMock: ReturnType<typeof stubRouter>) {
  fireEvent.click(screen.getByRole("button", { name: /run full query/i }));
  await waitFor(() =>
    expect(
      fetchMock.mock.calls.some((c) => String(c[0]).includes("/query/execute")),
    ).toBe(true),
  );
}

function routes(intent: Record<string, unknown>) {
  return {
    "/geospatial/resolve": { body: CHENNAI },
    "/query/parse": { body: intent },
    "/query/execute": { body: executionResult(intent) },
    "/query/analyze": { body: ANALYSIS },
  };
}

describe("QueryPanel - the parsed intent survives to execution", () => {
  it("executes every window of a parsed time series, not just the first", async () => {
    const fetchMock = stubRouter(routes(SERIES_INTENT));
    render(<QueryPanel />);

    await parse(SERIES_INTENT, fetchMock);
    await runFullQuery(fetchMock);

    const intent = lastBody(fetchMock, "/query/execute").intent as {
      temporal_mode: string;
      time_windows: { start_date: string; end_date: string }[];
    };

    expect(intent.temporal_mode).toBe("timeseries");
    expect(intent.time_windows).toHaveLength(3);
    // Written from literals: the last window is the one the old code dropped.
    expect(intent.time_windows[0]).toEqual({
      start_date: "2025-01-01",
      end_date: "2025-01-31",
    });
    expect(intent.time_windows[1]).toEqual({
      start_date: "2025-02-01",
      end_date: "2025-02-28",
    });
    expect(intent.time_windows[2]).toEqual({
      start_date: "2025-03-01",
      end_date: "2025-03-31",
    });
  });

  it("carries a parsed NDWI threshold through to execution", async () => {
    const fetchMock = stubRouter(routes(SERIES_INTENT));
    render(<QueryPanel />);

    await parse(SERIES_INTENT, fetchMock);
    await runFullQuery(fetchMock);

    const intent = lastBody(fetchMock, "/query/execute").intent as Record<
      string,
      unknown
    >;
    expect(intent.ndwi_threshold).toEqual({ operator: "gt", value: 0.3 });
  });

  it("sends the parsed intent unchanged when nothing was edited", async () => {
    const fetchMock = stubRouter(routes(SERIES_INTENT));
    render(<QueryPanel />);

    await parse(SERIES_INTENT, fetchMock);
    await runFullQuery(fetchMock);

    // Whole-object equality: no field added, dropped or rewritten.
    expect(lastBody(fetchMock, "/query/execute").intent).toEqual(SERIES_INTENT);
  });

  it("changes only what the user edited", async () => {
    const fetchMock = stubRouter(routes(SERIES_INTENT));
    render(<QueryPanel />);

    await parse(SERIES_INTENT, fetchMock);
    fireEvent.change(screen.getByLabelText("Place name"), {
      target: { value: "Marina Beach, Chennai" },
    });
    await runFullQuery(fetchMock);

    expect(lastBody(fetchMock, "/query/execute").intent).toEqual({
      ...SERIES_INTENT,
      location_query: "Marina Beach, Chennai",
    });
  });

  it("shows the series rather than a single date it did not parse", async () => {
    const fetchMock = stubRouter(routes(SERIES_INTENT));
    render(<QueryPanel />);

    await parse(SERIES_INTENT, fetchMock);

    expect(screen.getByLabelText(/time series \(3 windows\)/i)).toBeChecked();
    expect(screen.getByText("2025-03-01 → 2025-03-31")).toBeInTheDocument();
  });

  it("replaces the series only when the user explicitly picks one date", async () => {
    const fetchMock = stubRouter(routes(SERIES_INTENT));
    render(<QueryPanel />);

    await parse(SERIES_INTENT, fetchMock);
    fireEvent.click(screen.getByLabelText("Single date"));
    fireEvent.change(screen.getByLabelText("Observation date"), {
      target: { value: "2025-05-04" },
    });
    await runFullQuery(fetchMock);

    const intent = lastBody(fetchMock, "/query/execute").intent as Record<
      string,
      unknown
    >;
    expect(intent.temporal_mode).toBe("single");
    expect(intent.time_windows).toEqual([
      { start_date: "2025-05-04", end_date: "2025-05-04" },
    ]);
    // The threshold was not what the user edited, so it still stands.
    expect(intent.ndwi_threshold).toEqual({ operator: "gt", value: 0.3 });
  });

  it("stays byte-identical to the previous contract without a threshold", async () => {
    const fetchMock = stubRouter(routes(PLAIN_INTENT));
    render(<QueryPanel />);

    await parse(PLAIN_INTENT, fetchMock);
    await runFullQuery(fetchMock);

    const intent = lastBody(fetchMock, "/query/execute").intent as Record<
      string,
      unknown
    >;
    expect(intent).toEqual(PLAIN_INTENT);
    // Absent, not null: an intent that states no threshold must not start
    // carrying the key.
    expect("ndwi_threshold" in intent).toBe(false);
  });
});
