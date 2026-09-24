import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";

afterEach(() => {
  vi.restoreAllMocks();
});

/**
 * The satellite-scene region, resolved at call time.
 *
 * The frame holds an intro before a run and the map after one, so it is a
 * different DOM node either side of a query. Capturing it once and asserting
 * into that reference later searches a detached element, which is why this is
 * re-queried inside every `waitFor` rather than hoisted above it.
 */
function mapPanelNow(): HTMLElement {
  return screen
    .getByRole("heading", { name: "Satellite scene" })
    .closest("section") as HTMLElement;
}

describe("App", () => {
  it("is map-first: the map region is there before any query, the intro layered over it", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      text: () => Promise.resolve("{}"),
    } as Response));

    render(<App />);

    // Exactly one satellite-scene region, and it is the map's - not an intro
    // standing in for it.
    const headings = screen.getAllByRole("heading", { name: "Satellite scene" });
    expect(headings).toHaveLength(1);
    expect(headings[0].closest("section")).not.toHaveClass("workspace-intro");
    // The intro is an aside on top of the map, before anything has run.
    expect(
      screen.getByRole("complementary", { name: "How SatQuery works" }),
    ).toBeInTheDocument();

    await waitFor(() =>
      expect(screen.queryByText(/Checking backend/i)).not.toBeInTheDocument(),
    );
  });

  it("renders the title and panels", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      text: () => Promise.resolve("{}"),
    } as Response));

    render(<App />);

    expect(screen.getByRole("heading", { name: "SatQuery", level: 1 })).toBeInTheDocument();
    // Direction B names the bands for what they hold: the question, and the
    // form that configures it.
    expect(
      screen.getByRole("heading", { name: "Natural-language query" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Query configuration" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Satellite scene" })).toBeInTheDocument();

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
      // The header shows state at a glance ("Operational", per Direction B §4);
      // the service, version and environment stay in the element's title so the
      // build environment is not the widest thing on screen.
      expect(screen.getByText(/^Operational$/i)).toBeInTheDocument(),
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
/**
 * The two endpoints the shell calls on mount, whatever a test is about.
 *
 * An unmatched route used to answer `200 {}`, which meant every test silently
 * received an empty object for the model catalog. Nothing noticed until
 * responses were checked at the boundary - and an empty object is not a catalog
 * the application could ever be sent. Defaults belong here rather than in each
 * test, because no test is ABOUT these two.
 */
const SHELL_ROUTES: Record<string, unknown> = {
  "/health": {
    status: "ok",
    service: "SatQuery API",
    version: "0.1.0",
    environment: "test",
  },
  "/ai/models": {
    role: "visual",
    default_provider: "gemini",
    default_model: "gemini-3.6-flash",
    models: [],
  },
};

function stubRouter(callerRoutes: Record<string, unknown>) {
  const fn = vi.fn((url: string) => {
    // A caller's route always wins, and is searched FIRST rather than merged:
    // matching is by substring, so a test spelling its key differently
    // ("/api/v1/ai/models") would otherwise lose to a default ("/ai/models")
    // that also matches - and silently receive the default instead of its own.
    const caller = Object.keys(callerRoutes).find((key) =>
      String(url).includes(key),
    );
    const fallback = Object.keys(SHELL_ROUTES).find((key) =>
      String(url).includes(key),
    );
    const body =
      caller !== undefined
        ? callerRoutes[caller]
        : fallback !== undefined
          ? SHELL_ROUTES[fallback]
          : {};
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
    expect(screen.getAllByText(SCENE.id).length).toBeGreaterThan(0),
  );

  fireEvent.click(screen.getByRole("button", { name: /load image/i }));
}

describe("App - a completed manual result is a completed result", () => {
  // The defect: export and the answer rail were gated on the AGENT result, so a
  // successful manual analysis left "Run an analysis" on screen with export
  // unavailable - while its own measurements sat in the panel beside it.

  it("offers evidence export after a manual run, with no agent run at all", async () => {
    stubRouter({
      "/health": { status: "ok", service: "SatQuery API", version: "0.1.0", environment: "test" },
      "/geospatial/resolve": CHENNAI,
      "/satellite/search": { scenes: [SCENE], scene_count: 1, catalog: "https://example.test/v1" },
      "/satellite/imagery": IMAGERY,
    });
    render(<App />);

    expect(
      screen.queryByRole("button", { name: /export evidence/i }),
    ).not.toBeInTheDocument();

    await runQueryToPreview();

    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: /export evidence/i }),
      ).toBeInTheDocument(),
    );
  });

  it("stops instructing a run once the manual result exists", async () => {
    stubRouter({
      "/health": { status: "ok", service: "SatQuery API", version: "0.1.0", environment: "test" },
      "/geospatial/resolve": CHENNAI,
      "/satellite/search": { scenes: [SCENE], scene_count: 1, catalog: "https://example.test/v1" },
      "/satellite/imagery": IMAGERY,
    });
    render(<App />);

    await runQueryToPreview();

    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: /export evidence/i }),
      ).toBeInTheDocument(),
    );
    expect(document.body.textContent).not.toMatch(
      /Run an analysis to produce an answer/i,
    );
  });
});

describe("App - query to map seam", () => {
  it("passes a successful preview through to the map panel", async () => {
    stubRouter({
      "/health": { status: "ok", service: "SatQuery API", version: "0.1.0", environment: "test" },
      "/geospatial/resolve": CHENNAI,
      "/satellite/search": { scenes: [SCENE], scene_count: 1, catalog: "https://example.test/v1" },
      "/satellite/imagery": IMAGERY,
    });
    render(<App />);

    await runQueryToPreview();

    // The map panel names the scene it received - proof it crossed the seam.
    await waitFor(() =>
      expect(within(mapPanelNow()).getByRole("status")).toHaveTextContent(SCENE.id),
    );
  });

  it("clears stale imagery from the map when a new search starts", async () => {
    stubRouter({
      "/health": { status: "ok", service: "SatQuery API", version: "0.1.0", environment: "test" },
      "/geospatial/resolve": CHENNAI,
      "/satellite/search": { scenes: [SCENE], scene_count: 1, catalog: "https://example.test/v1" },
      "/satellite/imagery": IMAGERY,
    });
    render(<App />);

    await runQueryToPreview();
    await waitFor(() =>
      expect(within(mapPanelNow()).getByRole("status")).toHaveTextContent(SCENE.id),
    );

    // A new search invalidates it: the map must not keep the old scene.
    fireEvent.click(
      screen.getByRole("button", { name: /search sentinel-2 scenes/i }),
    );

    await waitFor(() =>
      expect(within(mapPanelNow()).getByRole("status")).not.toHaveTextContent(
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

    await waitFor(() =>
      expect(within(mapPanelNow()).getByRole("status")).toHaveTextContent(/NDWI/i),
    );
  });

  it("keeps execution imagery and scene evidence when optional analysis fails", async () => {
    const execution = {
      ...EXECUTION,
      windows: [{ ...EXECUTION.windows[0], imagery: IMAGERY }],
    };
    const router = stubRouter({ "/query/execute": execution });
    const route = router.getMockImplementation()!;
    router.mockImplementation((url: string) =>
      String(url).includes("/query/analyze")
        ? Promise.reject(new Error("Analysis unavailable"))
        : route(url),
    );
    render(<App />);
    await runNdwiQuery();
    await waitFor(() =>
      expect(within(mapPanelNow()).getByRole("status")).toHaveTextContent(SCENE.id),
    );
    await screen.findByText(/Network request to .*query\/analyze failed/);
    const evidence = screen.getByRole("heading", { name: "Deterministic evidence" })
      .closest("section") as HTMLElement;
    expect(within(evidence).getByText(SCENE.id)).toBeInTheDocument();
    expect(within(evidence).getByText(/128 × 96/)).toBeInTheDocument();
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

    await waitFor(() =>
      expect(screen.getByRole("heading", { name: "Analysis result" })).toBeInTheDocument(),
    );
    expect(within(mapPanelNow()).queryByText(/NDWI/i)).not.toBeInTheDocument();
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
              first_scene_id: "S2_BASE",
              second_scene_id: "S2_TARGET",
              first_acquired_at: null,
              second_acquired_at: null,
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

    await waitFor(() =>
      expect(within(mapPanelNow()).getByRole("status")).toHaveTextContent(/NDWI change/i),
    );
  });

  it("shows no change on the map when the grids were not comparable", async () => {
    await runQuery(null);

    await waitFor(() =>
      expect(screen.getByRole("heading", { name: "Analysis result" })).toBeInTheDocument(),
    );
    expect(within(mapPanelNow()).queryByText(/NDWI change/i)).not.toBeInTheDocument();
  });
});

// --------------------------------------------------------------------------- #
// The AgentPanel -> App -> MapPanel seam
// --------------------------------------------------------------------------- #
//
// The agent retrieves its own imagery as part of `execute_query`, so asking a
// question must place a scene on the same map the manual flow uses. This runs
// the real components end to end; jsdom has no WebGL2, so the map reports the
// scene it received rather than drawing it, which is what makes the seam
// observable here.

describe("App - agent question reaches the map", () => {
  const AGENT_SCENE = "S2A_44PMV_20240115_0_L2A";

  const AGENT_RESULT = {
    status: "ok",
    answer: "Water is visible along the eastern shoreline.",
    trace: {
      plan: {
        steps: [
          {
            tool: "execute_query",
            intent: {
              location_query: "Marina Beach, Chennai",
              temporal_mode: "single",
              time_windows: [
                { start_date: "2024-01-01", end_date: "2024-01-31" },
              ],
              modalities: ["sentinel-2-optical"],
              task: "visualize",
            },
            include_imagery: true,
            max_cloud_cover: null,
          },
          { tool: "rs_model_analysis", question: "Is there visible water?" },
        ],
      },
      steps: [],
      evidence_refs: [],
      answer_validation: {
        numeric_grounding: "pass",
        forbidden_terms: "pass",
        evidence_refs: "pass",
        visual_claims: "attributed",
      },
    },
    evidence: {
      items: [],
      execution: {
        plan: { intent: null, bbox: null },
        executed_modalities: ["sentinel-2-optical"],
        skipped_modalities: [],
        windows: [
          {
            modality: "sentinel-2-optical",
            label: "single",
            time_range: { start_date: "2024-01-01", end_date: "2024-01-31" },
            scene_count: 6,
            scenes: [],
            selected_scene_id: AGENT_SCENE,
            imagery: { ...IMAGERY, scene_id: AGENT_SCENE },
            imagery_error: null,
          },
        ],
        catalog: "https://example.test/v1",
      },
      analysis: null,
    },
  };

  it("places the scene the agent retrieved on the map", async () => {
    stubRouter({
      "/health": { status: "ok", service: "SatQuery API", version: "0.1.0", environment: "test" },
      "/query/agent": AGENT_RESULT,
    });
    render(<App />);

    fireEvent.change(screen.getByLabelText("Question"), {
      target: {
        value: "Is there visible water in Marina Beach, Chennai?",
      },
    });
    fireEvent.click(screen.getByRole("button", { name: /^run analysis$/i }));

    await waitFor(() =>
      expect(within(mapPanelNow()).getByRole("status")).toHaveTextContent(
        AGENT_SCENE,
      ),
    );
  });

  it("discards a pending manual preview when an agent run takes ownership", async () => {
    let finishPreview!: (response: Response) => void;
    const pending = new Promise<Response>((resolve) => { finishPreview = resolve; });
    const router = stubRouter({
      "/geospatial/resolve": CHENNAI,
      "/satellite/search": { scenes: [SCENE], scene_count: 1, catalog: "https://example.test/v1" },
      "/query/agent": AGENT_RESULT,
    });
    const route = router.getMockImplementation()!;
    router.mockImplementation((url: string) =>
      String(url).includes("/satellite/imagery") ? pending : route(url),
    );
    render(<App />);
    await runQueryToPreview();
    fireEvent.change(screen.getByLabelText("Question"), {
      target: { value: "Show Marina Beach in January 2024" },
    });
    fireEvent.click(screen.getByRole("button", { name: /^run analysis$/i }));
    await screen.findByText(AGENT_RESULT.answer);
    await act(async () => {
      finishPreview({
        ok: true, status: 200,
        text: () => Promise.resolve(JSON.stringify(IMAGERY)),
      } as Response);
    });
    expect(within(mapPanelNow()).getByRole("status")).toHaveTextContent(AGENT_SCENE);
    expect(within(mapPanelNow()).getByRole("status")).not.toHaveTextContent(SCENE.id);
  });

  it("clears an agent answer when a manual location run takes ownership", async () => {
    stubRouter({ "/query/agent": AGENT_RESULT, "/geospatial/resolve": CHENNAI });
    render(<App />);
    fireEvent.change(screen.getByLabelText("Question"), {
      target: { value: "Show Marina Beach in January 2024" },
    });
    fireEvent.click(screen.getByRole("button", { name: /^run analysis$/i }));
    await screen.findByText(AGENT_RESULT.answer);
    fireEvent.change(screen.getByLabelText("Place name"), { target: { value: "Chennai" } });
    fireEvent.click(screen.getByRole("button", { name: /resolve location/i }));
    await screen.findByText(CHENNAI.display_name);
    expect(screen.queryByText(AGENT_RESULT.answer)).not.toBeInTheDocument();
    // Assert on the whole panel rather than on one status node. Which hint the
    // map renders depends on the imagery state it happens to be in, so
    // getByRole("status") threw whenever that state rendered no hint - passing
    // alone and failing in a full-file run. The panel-wide assertion is what
    // this test actually means, and is strictly stronger: the agent's scene
    // must appear NOWHERE in the map, not merely be absent from one node.
    expect(mapPanelNow()).not.toHaveTextContent(AGENT_SCENE);
  });

  it("shows the resolved query context the server planned", async () => {
    stubRouter({
      "/health": { status: "ok", service: "SatQuery API", version: "0.1.0", environment: "test" },
      "/query/agent": AGENT_RESULT,
    });
    render(<App />);

    fireEvent.change(screen.getByLabelText("Question"), {
      target: { value: "Is there visible water in Marina Beach, Chennai?" },
    });
    fireEvent.click(screen.getByRole("button", { name: /^run analysis$/i }));

    // Read back from the validated plan - never from the question text.
    await waitFor(() =>
      expect(screen.getAllByText("Marina Beach, Chennai").length).toBeGreaterThan(0),
    );
    // Same-year windows abbreviate the end date, as the design writes them.
    expect(screen.getByText("2024-01-01 → 01-31")).toBeInTheDocument();
    expect(screen.getByText("Sentinel-2 L2A")).toBeInTheDocument();
  });
});

// --------------------------------------------------------------------------- #
// Provider selection
// --------------------------------------------------------------------------- #
//
// Switching the vision-language backend changes the inference provider for the
// NEXT run and nothing else. It must never disturb the scene, the evidence, the
// map or the query already on screen, and no credential may reach the browser.

describe("App - AI provider and model selection", () => {
  const CATALOG = {
        role: "visual",
        default_provider: "gemini",
        default_model: "gemini-3.6-flash",
        models: [
          {
            provider: "gemini", model_id: "gemini-3.6-flash",
            display_name: "Gemini 3.6 Flash", modality: "multimodal",
            supports_image: true, supports_text: true, supports_video: false,
            supports_tools: true, supports_structured_output: true,
            endpoint_type: "gemini-genai", configured: true, compatible: true,
            status: "Ready",
          },
          {
            // Retired by NVIDIA (410 Gone). Still image-capable, hence
            // compatible: true - availability is the separate fact that makes
            // it unselectable.
            provider: "nvidia", model_id: "nvidia/nemotron-nano-12b-v2-vl",
            display_name: "Nemotron Nano 12B v2 VL", modality: "multimodal",
            supports_image: true, supports_text: true, supports_video: false,
            supports_tools: false, supports_structured_output: true,
            endpoint_type: "openai-compatible", configured: true,
            compatible: true, available: false,
            retired_reason: "Retired by NVIDIA on 2026-08-26 (HTTP 410 Gone).",
            status: "Retired by provider",
          },
          {
            provider: "nvidia",
            model_id: "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            display_name: "Nemotron 3 Nano Omni 30B A3B Reasoning",
            modality: "multimodal",
            supports_image: true, supports_text: true, supports_video: true,
            supports_tools: true, supports_structured_output: false,
            endpoint_type: "openai-compatible", configured: true,
            compatible: true, available: true, status: "Ready",
          },
          {
            provider: "nvidia", model_id: "nvidia/nemotron-3-super-120b-a12b",
            display_name: "Nemotron 3 Super 120B A12B", modality: "text",
            supports_image: false, supports_text: true, supports_video: false,
            supports_tools: true, supports_structured_output: false,
            endpoint_type: "openai-compatible", configured: true,
            compatible: false, status: "Unsupported for visual analysis",
          },
        ],
      };

  const AGENT_STUB = {
    status: "planner_unavailable",
    answer: null,
    trace: { plan: null, steps: [], evidence_refs: [], answer_validation: null },
    evidence: { items: [], execution: null, analysis: null },
  };

  const HEALTH = {
    status: "ok", service: "SatQuery API", version: "0.1.0", environment: "test",
  };

  const OK_RESULT = {
    status: "ok",
    answer: "Water is visible along the eastern shoreline.",
    failure: null,
    trace: { plan: null, steps: [], evidence_refs: [], answer_validation: null },
    evidence: { items: [], execution: null, analysis: null },
  };

  it("exports the executed question, not a draft edited afterwards", async () => {
    // Astra P1 #2. The question box stays editable after a run, and the export
    // read that live input - so typing a new place over a finished result
    // produced a report pairing the NEW question with the OLD measurements.
    // Every number in it was real; the record was still false.
    const blobs: string[] = [];
    const originalCreate = URL.createObjectURL;
    const originalRevoke = URL.revokeObjectURL;
    URL.createObjectURL = ((blob: Blob) => {
      // jsdom Blobs expose their parts synchronously via this internal-free
      // route: re-read what was handed in.
      blobs.push((blob as unknown as { _text?: string })._text ?? "");
      return "blob:stub";
    }) as typeof URL.createObjectURL;
    URL.revokeObjectURL = (() => {}) as typeof URL.revokeObjectURL;

    // Capture the JSON at construction instead of fighting jsdom's Blob.
    const OriginalBlob = globalThis.Blob;
    class CapturingBlob extends OriginalBlob {
      _text: string;
      constructor(parts: BlobPart[], options?: BlobPropertyBag) {
        super(parts, options);
        this._text = parts.map(String).join("");
      }
    }
    globalThis.Blob = CapturingBlob as unknown as typeof Blob;

    try {
      stubRouter({ "/health": HEALTH, "/query/agent": OK_RESULT });
      render(<App />);

      fireEvent.change(screen.getByLabelText("Question"), {
        target: { value: "NDWI Marina Beach January 2025" },
      });
      fireEvent.click(screen.getByRole("button", { name: /^run analysis$/i }));
      await screen.findByText(OK_RESULT.answer);

      // Edit the draft WITHOUT submitting.
      fireEvent.change(screen.getByLabelText("Question"), {
        target: { value: "NDWI Hussain Sagar February 2025" },
      });

      fireEvent.click(screen.getByRole("button", { name: /export evidence/i }));

      expect(blobs.length).toBeGreaterThan(0);
      const report = JSON.parse(blobs[blobs.length - 1]);
      expect(report.question).toBe("NDWI Marina Beach January 2025");
      expect(report.question).not.toMatch(/Hussain Sagar/);
    } finally {
      globalThis.Blob = OriginalBlob;
      URL.createObjectURL = originalCreate;
      URL.revokeObjectURL = originalRevoke;
    }
  });

  it("names the provider the run actually used, not the current selector", async () => {
    // Astra P2 #5. The selector configures the NEXT run. A result already on
    // screen must keep naming the provider that produced it.
    stubRouter({
      "/health": HEALTH,
      "/ai/models": CATALOG,
      "/query/agent": OK_RESULT,
    });
    render(<App />);

    // AI is opted into per run (M5.5): pick Gemini explicitly first.
    const picker = (await screen.findByLabelText(
      /ai provider and model/i,
    )) as HTMLSelectElement;
    fireEvent.change(picker, { target: { value: "gemini-3.6-flash" } });

    fireEvent.change(screen.getByLabelText("Question"), {
      target: { value: "Is there visible water in Marina Beach?" },
    });
    fireEvent.click(screen.getByRole("button", { name: /^run analysis$/i }));
    await screen.findByText(OK_RESULT.answer);

    const attribution = await screen.findByTestId("run-attribution");
    expect(attribution.textContent).toMatch(/Gemini/);

    // Now switch the selector to a DIFFERENT provider without re-running.
    const select = (await screen.findByLabelText(
      /ai provider and model/i,
    )) as HTMLSelectElement;
    fireEvent.change(select, {
      target: { value: "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning" },
    });

    // The completed result must still be attributed to Gemini.
    expect(screen.getByTestId("run-attribution").textContent).toMatch(/Gemini/);
    expect(screen.getByTestId("run-attribution").textContent).not.toMatch(
      /NVIDIA/,
    );
  });

  it("attributes a default run to the standard workflow, never to the default AI", async () => {
    // M5.5: a run that names no provider is interpreted without any model.
    // Crediting it to the deployment's default provider (Gemini here) would
    // name a service that was never called.
    stubRouter({
      "/health": HEALTH,
      "/ai/models": CATALOG,
      "/query/agent": OK_RESULT,
    });
    render(<App />);
    await screen.findByLabelText(/ai provider and model/i);

    fireEvent.change(screen.getByLabelText("Question"), {
      target: { value: "What is the NDWI of Marina Beach, Chennai in January 2025?" },
    });
    fireEvent.click(screen.getByRole("button", { name: /^run analysis$/i }));
    await screen.findByText(OK_RESULT.answer);

    const attribution = screen.getByTestId("run-attribution").textContent;
    expect(attribution).toMatch(/standard workflow · local intent model, no external AI/);
    expect(attribution).not.toMatch(/Gemini/);
  });

  it("offers the standard workflow first, then the catalogued models", async () => {
    stubRouter({ "/health": HEALTH, "/ai/models": CATALOG });
    render(<App />);

    const select = (await screen.findByLabelText(
      /ai provider and model/i,
    )) as HTMLSelectElement;

    // The default names no provider: the server's standard workflow (no external AI).
    expect(select.value).toBe("");
    const options = [...select.options];
    expect(options[0].textContent).toBe("Standard · no external AI");
    expect(options.map((option) => option.value)).toEqual([
      "",
      "gemini-3.6-flash",
      "nvidia/nemotron-nano-12b-v2-vl",
      "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
      "nvidia/nemotron-3-super-120b-a12b",
    ]);
    // A text-only model stays visible but cannot be chosen for a visual run.
    const textOnly = options[4];
    expect(textOnly.disabled).toBe(true);
    expect(textOnly.textContent).toMatch(/Unsupported for visual analysis/);
  });

  it("marks a local model that is not installed as not ready", async () => {
    // Configured, compatible and not retired used to be enough for a green
    // dot, so an uninstalled local model read "Not installed" beside the
    // colour that means ready. Readiness now follows the server's own status.
    const local = {
      provider: "local", model_id: "qwen3-vl:4b-instruct",
      display_name: "Qwen3-VL 4B Instruct (local)", modality: "multimodal",
      supports_image: true, supports_text: true, supports_video: false,
      supports_tools: false, supports_structured_output: true,
      endpoint_type: "ollama-chat", configured: true, compatible: true,
      available: true, status: "Not installed",
    };
    stubRouter({
      "/health": HEALTH,
      "/ai/models": {
        ...CATALOG,
        default_provider: "local",
        default_model: "qwen3-vl:4b-instruct",
        models: [...CATALOG.models, local],
      },
    });
    render(<App />);

    const select = (await screen.findByLabelText(
      /ai provider and model/i,
    )) as HTMLSelectElement;
    // Selectable - and its badge is only shown once it is the selection.
    fireEvent.change(select, { target: { value: "qwen3-vl:4b-instruct" } });
    expect(select.value).toBe("qwen3-vl:4b-instruct");
    const option = [...select.options].find(
      (candidate) => candidate.value === "qwen3-vl:4b-instruct",
    );
    expect(option?.textContent).toMatch(/Local · Qwen3-VL 4B Instruct.*Not installed/);

    const badge = document.querySelector(".model-status");
    expect(badge?.textContent).toBe("Not installed");
    expect(badge?.getAttribute("data-ready")).toBe("false");
  });

  it.each(["Ollama not running", "Ollama cannot read models"])(
    "shows a local model whose Ollama reports %s as not ready",
    async (status) => {
      // Two different fixes - start Ollama, or reconnect the drive holding its
      // models - so the server says which, and neither is ever green.
      const local = {
        provider: "local", model_id: "qwen3-vl:4b-instruct",
        display_name: "Qwen3-VL 4B Instruct (local)", modality: "multimodal",
        supports_image: true, supports_text: true, supports_video: false,
        supports_tools: true, supports_structured_output: true,
        endpoint_type: "ollama-chat", configured: true, compatible: true,
        available: true, status,
      };
      stubRouter({
        "/health": HEALTH,
        "/ai/models": {
          ...CATALOG,
          default_provider: "local",
          default_model: "qwen3-vl:4b-instruct",
          models: [...CATALOG.models, local],
        },
      });
      render(<App />);
      const select = await screen.findByLabelText(/ai provider and model/i);
      fireEvent.change(select, { target: { value: "qwen3-vl:4b-instruct" } });

      const badge = document.querySelector(".model-status");
      expect(badge?.textContent).toBe(status);
      expect(badge?.getAttribute("data-ready")).toBe("false");
    },
  );

  it("re-reads model statuses when the window regains focus", async () => {
    // Statuses are a snapshot: pulling a model or starting Ollama in another
    // window left the badge reading "Not installed" beside a successful local
    // answer (observed live). Coming back to the window re-reads them.
    const catalog = (status: string) => ({
      ...CATALOG,
      default_provider: "local",
      default_model: "qwen3-vl:4b-instruct",
      models: [
        ...CATALOG.models,
        {
          provider: "local", model_id: "qwen3-vl:4b-instruct",
          display_name: "Qwen3-VL 4B Instruct (local)", modality: "multimodal",
          supports_image: true, supports_text: true, supports_video: false,
          supports_tools: true, supports_structured_output: true,
          endpoint_type: "ollama-chat", configured: true, compatible: true,
          available: true, status,
        },
      ],
    });
    const routes: Record<string, unknown> = {
      "/health": HEALTH,
      "/ai/models": catalog("Not installed"),
    };
    stubRouter(routes);
    render(<App />);
    const select = await screen.findByLabelText(/ai provider and model/i);
    fireEvent.change(select, { target: { value: "qwen3-vl:4b-instruct" } });
    const badge = () => document.querySelector(".model-status");
    await waitFor(() => expect(badge()?.textContent).toBe("Not installed"));

    routes["/ai/models"] = catalog("Ready");
    fireEvent(window, new Event("focus"));

    await waitFor(() => expect(badge()?.textContent).toBe("Ready"));
    expect(badge()?.getAttribute("data-ready")).toBe("true");
  });

  it("keeps a ready model's badge green", async () => {
    stubRouter({ "/health": HEALTH, "/ai/models": CATALOG });
    render(<App />);
    await screen.findByLabelText(/ai provider and model/i);
    expect(document.querySelector(".model-status")?.getAttribute("data-ready")).toBe(
      "true",
    );
  });

  it("never offers a provider-retired model as selectable", async () => {
    // The defect this guards: the retired model reported compatible AND
    // configured, so the selector enabled it and the run failed with 410. It
    // stays LISTED - hiding it would invite someone to re-add it - but it must
    // never be pickable, and it must say why.
    stubRouter({ "/health": HEALTH, "/ai/models": CATALOG });
    render(<App />);

    const select = (await screen.findByLabelText(
      /ai provider and model/i,
    )) as HTMLSelectElement;
    const retired = [...select.options].find(
      (option) => option.value === "nvidia/nemotron-nano-12b-v2-vl",
    );

    expect(retired).toBeDefined();
    expect(retired!.disabled).toBe(true);
    expect(retired!.textContent).toMatch(/Retired by provider/);
    expect(retired!.textContent).not.toMatch(/Ready/);
  });

  it("still offers a live model of the same provider", async () => {
    // The counter-case. Disabling every NVIDIA option would satisfy the test
    // above while removing the provider from the product.
    stubRouter({ "/health": HEALTH, "/ai/models": CATALOG });
    render(<App />);

    const select = (await screen.findByLabelText(
      /ai provider and model/i,
    )) as HTMLSelectElement;
    const live = [...select.options].find(
      (option) =>
        option.value === "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    );

    expect(live).toBeDefined();
    expect(live!.disabled).toBe(false);
    expect(live!.textContent).not.toMatch(/Retired/);
  });

  it("sends the selected provider and model with the run", async () => {
    const fetchMock = stubRouter({
      "/health": HEALTH, "/ai/models": CATALOG, "/query/agent": AGENT_STUB,
    });
    render(<App />);

    fireEvent.change(await screen.findByLabelText(/ai provider and model/i), {
      target: { value: "nvidia/nemotron-nano-12b-v2-vl" },
    });
    fireEvent.change(screen.getByLabelText("Question"), {
      target: { value: "Any water near Kochi?" },
    });
    fireEvent.click(screen.getByRole("button", { name: /^run analysis$/i }));

    await waitFor(() => {
      const call = (fetchMock.mock.calls as unknown[][]).find((entry) =>
        String(entry[0]).includes("/query/agent"),
      );
      expect(call).toBeDefined();
      expect(JSON.parse((call![1] as { body: string }).body)).toEqual({
        question: "Any water near Kochi?",
        provider: "nvidia",
        model: "nvidia/nemotron-nano-12b-v2-vl",
      });
    });
  });

  it("omits provider and model when the default is left alone", async () => {
    const fetchMock = stubRouter({
      "/health": HEALTH, "/ai/models": CATALOG, "/query/agent": AGENT_STUB,
    });
    render(<App />);
    await screen.findByLabelText(/ai provider and model/i);

    fireEvent.change(screen.getByLabelText("Question"), {
      target: { value: "Any water near Kochi?" },
    });
    fireEvent.click(screen.getByRole("button", { name: /^run analysis$/i }));

    await waitFor(() => {
      const call = (fetchMock.mock.calls as unknown[][]).find((entry) =>
        String(entry[0]).includes("/query/agent"),
      );
      expect(call).toBeDefined();
      // Byte-identical to the pre-provider contract.
      expect(JSON.parse((call![1] as { body: string }).body)).toEqual({
        question: "Any water near Kochi?",
      });
    });
  });

  it("keeps the loaded scene when the model changes", async () => {
    stubRouter({
      "/health": HEALTH,
      "/ai/models": CATALOG,
      "/geospatial/resolve": CHENNAI,
      "/satellite/search": { scenes: [SCENE], scene_count: 1, catalog: "https://example.test/v1" },
      "/satellite/imagery": IMAGERY,
    });
    render(<App />);
    await screen.findByLabelText(/ai provider and model/i);

    await runQueryToPreview();
    await waitFor(() =>
      expect(within(mapPanelNow()).getAllByText(new RegExp(SCENE.id)).length)
        .toBeGreaterThan(0),
    );
    const before = within(mapPanelNow()).getAllByText(new RegExp(SCENE.id)).length;

    fireEvent.change(screen.getByLabelText(/ai provider and model/i), {
      target: { value: "nvidia/nemotron-nano-12b-v2-vl" },
    });

    // Only the next run's backend moved; the deterministic result is untouched.
    expect(within(mapPanelNow()).getAllByText(new RegExp(SCENE.id)).length).toBe(
      before,
    );
  });

  it("degrades honestly when the catalog cannot be read", async () => {
    stubRouter({ "/health": HEALTH, "/ai/models": { nope: true } });
    render(<App />);

    // Only the optional AI models are unknown; the standard workflow the next
    // question uses is named as still in effect.
    expect(
      await screen.findByText(/Standard · AI models unavailable/i),
    ).toBeInTheDocument();
    // The workspace still renders around it.
    expect(
      screen.getByRole("heading", { name: "Natural-language query" }),
    ).toBeInTheDocument();
  });

  it("never puts a credential in the document", async () => {
    stubRouter({ "/health": HEALTH, "/ai/models": CATALOG });
    const { container } = render(<App />);
    await screen.findByLabelText(/ai provider and model/i);

    const markup = container.innerHTML.toLowerCase();
    for (const secret of ["api_key", "apikey", "bearer", "nvapi-"]) {
      expect(markup).not.toContain(secret);
    }
  });
});
