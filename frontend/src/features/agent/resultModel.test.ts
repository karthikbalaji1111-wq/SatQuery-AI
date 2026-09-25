import { describe, expect, it } from "vitest";

import {
  AREA_TOO_LARGE_CHENNAI,
  CLARIFY_CHENNAI,
  LOCATION_UNAVAILABLE,
  NDBI_AMEERPET,
  NDVI_CUBBON,
  NDWI_MARINA,
  NO_SCENES,
  NOT_FOUND,
  REFUSED_BY_RADIOMETRY,
  SAHARA_POINT,
  SAR_MARINA,
  TEMPORAL_MARINA,
  UNSUPPORTED_SHIPS,
} from "./m6Fixtures";
import {
  formatPeriod,
  outcomeOf,
  resultContext,
  resultSummary,
  runStages,
} from "./resultModel";

describe("outcomeOf - every kind of result stays its own kind", () => {
  it.each([
    ["NDVI", NDVI_CUBBON, "success"],
    ["NDWI", NDWI_MARINA, "success"],
    ["NDBI", NDBI_AMEERPET, "success"],
    ["SAR", SAR_MARINA, "success"],
    ["temporal NDWI", TEMPORAL_MARINA, "success"],
    ["a missing detail", CLARIFY_CHENNAI, "clarification"],
    ["a place not found", NOT_FOUND, "location_not_found"],
    ["a geocoder outage", LOCATION_UNAVAILABLE, "location_unavailable"],
    ["a whole city", AREA_TOO_LARGE_CHENNAI, "area_too_large"],
    ["a place that is only a point", SAHARA_POINT, "location_is_point"],
    ["counting ships", UNSUPPORTED_SHIPS, "unsupported"],
    ["no matching scene", NO_SCENES, "insufficient_evidence"],
    ["a radiometric refusal", REFUSED_BY_RADIOMETRY, "analysis_refused"],
  ])("%s -> %s", (_, result, kind) => {
    expect(outcomeOf(result).kind).toBe(kind);
  });

  it("keeps the server's own reason for a validated refusal, verbatim", () => {
    expect(outcomeOf(REFUSED_BY_RADIOMETRY).reason).toMatch(
      /^NDVI was not computed: Radiometric validation failed/,
    );
  });
});

describe("resultSummary - the operation's own values, never another's", () => {
  it("reads an index from its own evidence ids", () => {
    const body = resultSummary(NDVI_CUBBON);
    expect(body.kind).toBe("index");
    if (body.kind !== "index") return;
    const [reading] = body.readings;
    expect(reading).toMatchObject({ key: "ndvi", label: "NDVI", title: "Vegetation index" });
    expect(reading.mean).toBe(0.5204459203473597);
    expect(reading.validPixels).toBe(16988);
    expect(reading.quality).toMatchObject({ valid: 16988, total: 17080 });
    expect(body.scene).toEqual({
      id: "S2B_43PGQ_20241208_0_L2A",
      acquired: "2024-12-08",
      platform: "sentinel-2b",
    });
  });

  it("reads backscatter per polarization, in dB", () => {
    const body = resultSummary(SAR_MARINA);
    expect(body.kind).toBe("sar");
    if (body.kind !== "sar") return;
    expect(body.reading).toMatchObject({
      vv: -5.4437196664666985,
      vh: -17.849565,
      difference: 12.405845,
      validPixels: 33600,
      polarizations: ["VV", "VH"],
    });
    expect(body.scene?.acquired).toBe("2025-01-11");
  });

  it("reads a comparison as two periods and the change between them", () => {
    const body = resultSummary(TEMPORAL_MARINA);
    expect(body.kind).toBe("temporal");
    if (body.kind !== "temporal") return;
    const { earlier, later, difference, pairedChange, pairedPixels } = body.reading;
    expect(earlier).toMatchObject({ role: "Earlier", period: "January 2024", acquired: "2024-01-15", mean: 0.02658240949942237 });
    expect(later).toMatchObject({ role: "Later", period: "January 2025", acquired: "2025-01-04", mean: 0.14657038314284795 });
    expect(difference).toBe(0.11998797364342559);
    expect(pairedChange).toBe(0.12069164662445794);
    expect(pairedPixels).toBe(33420);
  });

  it("claims nothing when nothing was measured", () => {
    expect(resultSummary(NO_SCENES).kind).toBe("none");
    expect(resultSummary(REFUSED_BY_RADIOMETRY).kind).toBe("none");
    expect(resultSummary(CLARIFY_CHENNAI).kind).toBe("none");
  });
});

describe("resultContext and formatPeriod", () => {
  it("names the place and the requested period as planned", () => {
    expect(resultContext(NDVI_CUBBON)).toEqual({
      location: "Cubbon Park, Bengaluru",
      matched: {
        name: "Cubbon Park, Sampangirama Nagar",
        full: "Cubbon Park, Sampangirama Nagar, Bengaluru Central City Corporation, Bengaluru, Karnataka, India",
        kind: "leisure · park",
      },
      periods: ["December 2024"],
      sceneCount: 7,
    });
    expect(resultContext(TEMPORAL_MARINA).periods).toEqual(["January 2024", "January 2025"]);
  });

  it("reads the place from the plan even when nothing was executed", () => {
    expect(resultContext(LOCATION_UNAVAILABLE).location).toBe("Dal Lake");
  });

  it.each([
    [{ start_date: "2024-12-01", end_date: "2024-12-31" }, "December 2024"],
    [{ start_date: "2024-02-01", end_date: "2024-02-29" }, "February 2024"],
    [{ start_date: "2023-02-01", end_date: "2023-02-28" }, "February 2023"],
    [{ start_date: "2024-01-01", end_date: "2024-12-31" }, "2024"],
    [{ start_date: "2025-01-15", end_date: "2025-01-15" }, "2025-01-15"],
    [{ start_date: "2024-12-01", end_date: "2025-01-31" }, "2024-12-01 → 2025-01-31"],
    [{ start_date: "2024-12-02", end_date: "2024-12-31" }, "2024-12-02 → 2024-12-31"],
  ])("formats %j as %s - never moving a date", (range, text) => {
    expect(formatPeriod(range)).toBe(text);
  });
});

describe("runStages - only the stages the response shows happened", () => {
  it("walks a full successful run", () => {
    expect(runStages(NDVI_CUBBON).map((stage) => [stage.name, stage.state, stage.detail])).toEqual([
      ["Understand question", "done", null],
      ["Resolve location", "done", "Cubbon Park, Bengaluru"],
      // Updated deliberately (layperson first): plain stage words.
      ["Find satellite images", "done", "7 found · 1 selected"],
      ["Check the images", "done", "data suitable & correctly aligned"],
      ["Run analysis", "done", "vegetation index ✓"],
      ["Check answer", "done", "matches the measurements"],
    ]);
  });

  it("sees a comparison's validation, published per observation", () => {
    expect(runStages(TEMPORAL_MARINA).map((stage) => stage.name)).toContain("Check the images");
    expect(runStages(TEMPORAL_MARINA).find((stage) => stage.name === "Check the images")?.state).toBe("done");
  });

  it("stops at the stage that did not complete", () => {
    expect(runStages(LOCATION_UNAVAILABLE).map((stage) => [stage.name, stage.state])).toEqual([
      ["Understand question", "done"],
      ["Resolve location", "failed"],
    ]);
    expect(runStages(AREA_TOO_LARGE_CHENNAI).map((stage) => [stage.name, stage.state])).toEqual([
      ["Understand question", "attention"],
      ["Resolve location", "attention"],
    ]);
    expect(runStages(SAHARA_POINT).map((stage) => [stage.name, stage.state, stage.detail])).toEqual([
      ["Understand question", "attention", runStages(SAHARA_POINT)[0].detail],
      ["Resolve location", "attention", "a single point, not an area"],
    ]);
    expect(runStages(CLARIFY_CHENNAI).map((stage) => stage.name)).toEqual(["Understand question"]);
    expect(runStages(NO_SCENES).at(-1)).toMatchObject({ name: "Find satellite images", state: "attention" });
  });

  it("reports a refused analysis as refused, not as done", () => {
    const stages = runStages(REFUSED_BY_RADIOMETRY);
    expect(stages.find((stage) => stage.name === "Run analysis")).toMatchObject({
      state: "failed",
      detail: "vegetation index ✕",
    });
  });
});

describe("runStages - the pipeline speaks plainly", () => {
  const JARGON = /\b(NDVI|NDWI|NDBI|SAR|VV|VH|backscatter|radiometr\w*|geometr\w*|scenes?|grounded|temporal)\b/i;

  it.each([
    ["vegetation", NDVI_CUBBON],
    ["water", NDWI_MARINA],
    ["built-up", NDBI_AMEERPET],
    ["radar", SAR_MARINA],
    ["two dates", TEMPORAL_MARINA],
    ["a refusal", REFUSED_BY_RADIOMETRY],
    ["no image", NO_SCENES],
  ])("%s: no stage name or detail needs remote-sensing knowledge", (_, result) => {
    const words = runStages(result)
      .map((stage) => `${stage.name} ${stage.detail ?? ""}`)
      .join(" ");
    expect(words).not.toMatch(JARGON);
  });

  it("names each analysis in everyday words", () => {
    const detail = (result: typeof NDVI_CUBBON) =>
      runStages(result).find((stage) => stage.name === "Run analysis")?.detail;
    expect(detail(NDWI_MARINA)).toBe("water index ✓");
    expect(detail(NDBI_AMEERPET)).toBe("built-up index ✓");
    expect(detail(SAR_MARINA)).toBe("radar ✓");
    expect(detail(TEMPORAL_MARINA)).toBe("water change ✓");
  });
});
