/**
 * A manual run's export has to be as auditable as an agent run's.
 *
 * The exported record carried, for the manual path, a scene, a picture and some
 * numbers. It could not say what was asked, which catalog answered, which
 * windows failed, or what the analysis warned about - all of which the browser
 * already held. A report that omits its own limitations reads as more
 * authoritative than the run that produced it, and this one is meant to be
 * handed to someone else.
 *
 * Pinned here: the submitted intent, per-window catalog provenance, discovery
 * failures kept apart from imagery failures, and the analysis's own warnings
 * and completeness - for a run with no agent involved at all.
 */

import { describe, expect, it } from "vitest";

import { buildEvidenceReport } from "./evidenceReport";
import type { ManualEvidenceInput } from "./evidenceReport";

const BBOX = { west: 80.1, south: 12.9, east: 80.3, north: 13.2 };
const EARTH_SEARCH = "https://earth-search.aws.element84.com/v1";
const PLANETARY = "https://planetarycomputer.microsoft.com/api/stac/v1";
const OUTAGE = "The satellite catalog is unavailable.";

const INTENT = {
  location_query: "Marina Beach, Chennai",
  temporal_mode: "single" as const,
  time_windows: [{ start_date: "2025-01-01", end_date: "2025-01-31" }],
  modalities: ["sentinel-2-optical" as const, "sentinel-1-sar" as const],
  task: "visualize" as const,
  ndwi_threshold: { operator: "gt" as const, value: 0.3 },
};

const SCENE = {
  id: "S2B_44PMV_20250104_0_L2A",
  datetime: "2025-01-04T05:12:34Z",
  bbox: BBOX,
  geometry: null,
  cloud_cover: 4.2,
  collection: "sentinel-2-l2a",
  platform: "sentinel-2b",
  processing_level: "L2A",
  thumbnail_url: null,
  assets: [],
};

function manualRun(): ManualEvidenceInput {
  return {
    scene: SCENE,
    imagery: null,
    measurements: [{ name: "ndwi_mean", value: 0.1464, unit: "index" }],
    intent: INTENT,
    execution: {
      plan: { intent: INTENT, bbox: BBOX },
      executed_modalities: ["sentinel-2-optical", "sentinel-1-sar"],
      skipped_modalities: [],
      windows: [
        {
          modality: "sentinel-2-optical",
          label: "single",
          time_range: { start_date: "2025-01-01", end_date: "2025-01-31" },
          scene_count: 1,
          scenes: [SCENE],
          selected_scene_id: SCENE.id,
          imagery: null,
          imagery_error: "The imagery could not be rendered.",
          catalog: EARTH_SEARCH,
          error: null,
        },
        {
          modality: "sentinel-1-sar",
          label: "single",
          time_range: { start_date: "2025-01-01", end_date: "2025-01-31" },
          scene_count: 0,
          scenes: [],
          selected_scene_id: null,
          imagery: null,
          imagery_error: null,
          catalog: null,
          error: OUTAGE,
        },
      ],
      catalog: EARTH_SEARCH,
      catalogs: [EARTH_SEARCH, PLANETARY],
      status: "partial",
    },
    analysis: {
      status: "ok",
      task: "visualize",
      answer: "Retrieved 2 window(s).",
      windows_considered: [],
      warnings: ["NDWI values are a spectral index, not a water classification."],
      measurements: [{ name: "ndwi_mean", value: 0.1464, unit: "index" }],
      analysis_outcomes: [
        { name: "ndwi", status: "completed" },
        {
          name: "sar_backscatter",
          status: "unavailable",
          reason: "no SAR window with a selected scene was available",
        },
      ],
      completeness: "partial",
    },
  };
}

describe("evidence report - a manual run", () => {
  it("records what was actually asked", async () => {
    const report = buildEvidenceReport(null, manualRun(), null);

    expect(report.manual?.intent).toEqual(INTENT);
    // Including the threshold, which the form has no control for and which used
    // to be dropped on the way to the server.
    expect(report.manual?.intent?.ndwi_threshold).toEqual({
      operator: "gt",
      value: 0.3,
    });
  });

  it("records which catalogs answered", async () => {
    const report = buildEvidenceReport(null, manualRun(), null);

    expect(report.manual?.execution?.catalogs).toEqual([EARTH_SEARCH, PLANETARY]);
    expect(report.manual?.execution?.status).toBe("partial");
  });

  it("keeps a catalog outage apart from an imagery failure", async () => {
    const report = buildEvidenceReport(null, manualRun(), null);

    expect(report.discovery_failures).toEqual([
      { modality: "sentinel-1-sar", window: "single", reason: OUTAGE },
    ]);
    expect(report.imagery_errors).toEqual([
      {
        modality: "sentinel-2-optical",
        window: "single",
        reason: "The imagery could not be rendered.",
      },
    ]);
  });

  it("carries the analysis's own warnings and completeness", async () => {
    const report = buildEvidenceReport(null, manualRun(), null);

    expect(report.warnings).toContain(
      "NDWI values are a spectral index, not a water classification.",
    );
    expect(report.manual?.analysis?.completeness).toBe("partial");
    // The unflattering half: an analysis that was asked for and not produced.
    expect(report.manual?.analysis?.analysis_outcomes).toContainEqual(
      expect.objectContaining({ name: "sar_backscatter", status: "unavailable" }),
    );
  });

  it("still builds a report when there is nothing to record", async () => {
    // Non-vacuity in the other direction: the additions must not require the
    // new fields to be present.
    const report = buildEvidenceReport(null, null, "what is the NDWI?");

    expect(report.manual).toBeNull();
    expect(report.discovery_failures).toEqual([]);
    expect(report.imagery_errors).toEqual([]);
    expect(report.question).toBe("what is the NDWI?");
  });
});
