/**
 * The typed client is a claim, not a check.
 *
 * `apiRequest<AgentResult>(...)` erases at runtime, so a response that does not
 * match the contract enters state unnoticed and fails later, somewhere else, as
 * `undefined is not an object` inside a component. That is the worst possible
 * place to learn that a server is a version behind, or that a proxy rewrote the
 * body.
 *
 * These pin the two halves that matter: a usable response passes through
 * unchanged (identity, not a copy - nothing is rebuilt or defaulted), and an
 * unusable one becomes an `ApiError` that every existing caller already knows
 * how to handle.
 */

import { describe, expect, it } from "vitest";

import { ApiError } from "./client";
import {
  asAgentResult,
  asAnalysisResult,
  asExecutionResult,
  asModelCatalog,
  asResolveResponse,
  asSceneSearchResponse,
} from "./validate";

const AGENT = {
  status: "ok",
  answer: "The mean NDWI was 0.1464 index.",
  trace: { plan: null, steps: [], evidence_refs: [], answer_validation: null },
  evidence: { items: [], execution: null, analysis: null },
};

const EXECUTION = {
  plan: {
    intent: { location_query: "Chennai" },
    bbox: { west: 80.1, south: 12.9, east: 80.3, north: 13.2 },
  },
  executed_modalities: ["sentinel-2-optical"],
  skipped_modalities: [],
  windows: [],
  catalog: "https://earth-search.aws.element84.com/v1",
};

const ANALYSIS = {
  status: "ok",
  task: "visualize",
  answer: "Retrieved 1 window(s).",
  windows_considered: [],
  warnings: [],
  measurements: [],
};

const CATALOG = {
  role: "visual",
  default_provider: "local",
  default_model: "qwen3-vl:4b-instruct",
  models: [],
};

const RESOLVED = {
  query_type: "place",
  display_name: "Chennai",
  center: { lat: 13.08, lon: 80.27 },
  bbox: { west: 80.1, south: 12.9, east: 80.3, north: 13.2 },
  source: "nominatim",
};

const SCENES = {
  query: {},
  scene_count: 0,
  scenes: [],
  catalog: "https://earth-search.aws.element84.com/v1",
};

function rejects(run: () => unknown): ApiError {
  try {
    run();
  } catch (error) {
    expect(error).toBeInstanceOf(ApiError);
    return error as ApiError;
  }
  throw new Error("expected the response to be refused");
}

describe("response validation - a usable response is untouched", () => {
  it.each([
    ["agent", () => asAgentResult(AGENT), AGENT],
    ["execution", () => asExecutionResult(EXECUTION), EXECUTION],
    ["analysis", () => asAnalysisResult(ANALYSIS), ANALYSIS],
    ["model catalog", () => asModelCatalog(CATALOG), CATALOG],
    ["location", () => asResolveResponse(RESOLVED), RESOLVED],
    ["scene search", () => asSceneSearchResponse(SCENES), SCENES],
  ])("passes a well-formed %s response straight through", (_label, run, body) => {
    // Identity: nothing is rebuilt, defaulted or reordered on the way past.
    expect(run()).toBe(body);
  });
});

describe("response validation - an unusable response is refused", () => {
  it("refuses a body that is not an object at all", () => {
    // What an HTML page or a bare string would look like by the time it gets
    // here.
    expect(rejects(() => asAgentResult("<!DOCTYPE html>")).code).toBe(
      "invalid_response",
    );
    expect(rejects(() => asExecutionResult(null)).code).toBe("invalid_response");
  });

  it("refuses an agent status this build does not understand", () => {
    // A newer server with a fifth status: better to say so than to render a
    // result whose meaning is unknown.
    const error = rejects(() => asAgentResult({ ...AGENT, status: "queued" }));
    expect(error.message).toMatch(/status/);
  });

  it("accepts a clarification that carries its message and options", () => {
    const body = {
      ...AGENT,
      status: "needs_clarification",
      answer: null,
      clarification: {
        reason: "date_missing",
        message: "For which date or period?",
        options: [],
        understood_analyses: ["vegetation (NDVI)"],
        understood_location: "Chennai",
        understood_periods: [],
      },
    };
    expect(asAgentResult(body)).toBe(body);
  });

  it("accepts a location outage that carries its failure", () => {
    const body = {
      ...AGENT,
      status: "location_unavailable",
      answer: null,
      failure: {
        stage: "location",
        code: "geocoding_unavailable",
        dependency: "geocoder",
        message: "Location service temporarily unavailable.",
        retry_after_seconds: 45,
      },
    };
    expect(asAgentResult(body)).toBe(body);
  });

  it("refuses a location outage with no failure to show", () => {
    const error = rejects(() =>
      asAgentResult({ ...AGENT, status: "location_unavailable", answer: null }),
    );
    expect(error.code).toBe("invalid_response");
  });

  it("refuses a clarification status with no clarification to show", () => {
    const error = rejects(() =>
      asAgentResult({ ...AGENT, status: "needs_clarification", answer: null }),
    );
    expect(error.code).toBe("invalid_response");
  });

  it("refuses an agent result whose evidence is missing", () => {
    expect(rejects(() => asAgentResult({ ...AGENT, evidence: {} })).message).toMatch(
      /items/,
    );
  });

  it("refuses an execution result with no windows list", () => {
    const { windows, ...withoutWindows } = EXECUTION;
    void windows;
    expect(rejects(() => asExecutionResult(withoutWindows)).message).toMatch(
      /windows/,
    );
  });

  it("refuses an execution result whose plan has no bbox", () => {
    expect(
      rejects(() =>
        asExecutionResult({ ...EXECUTION, plan: { intent: {} } }),
      ).code,
    ).toBe("invalid_response");
  });

  it("refuses an analysis result with no answer", () => {
    const { answer, ...withoutAnswer } = ANALYSIS;
    void answer;
    expect(rejects(() => asAnalysisResult(withoutAnswer)).message).toMatch(
      /answer/,
    );
  });

  it("refuses coordinates that are not numbers", () => {
    const error = rejects(() =>
      asResolveResponse({ ...RESOLVED, center: { lat: "13.08", lon: 80.27 } }),
    );
    // Without this the map camera would be moved to NaN rather than fail.
    expect(error.message).toMatch(/coordinates/);
  });

  it("refuses a bounding box missing an edge", () => {
    expect(
      rejects(() =>
        asResolveResponse({
          ...RESOLVED,
          bbox: { west: 80.1, south: 12.9, east: 80.3 },
        }),
      ).message,
    ).toMatch(/north/);
  });

  it("refuses a model catalog with no models list", () => {
    const { models, ...withoutModels } = CATALOG;
    void models;
    expect(rejects(() => asModelCatalog(withoutModels)).message).toMatch(/models/);
  });
});
