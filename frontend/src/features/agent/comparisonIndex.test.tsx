import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { AgentResult } from "../../api/types";
import { AgentAnswerPanel } from "./AgentPanel";
import { interpretResult, technicalDetails } from "./interpretation";
import { TEMPORAL_MARINA } from "./m6Fixtures";
import { resultSummary } from "./resultModel";

/**
 * The comparison tool keeps its NDWI name, but it now compares vegetation
 * (NDVI) and built-up area (NDBI) too. Every word a reader sees must name the
 * index that was compared, never "water" for a vegetation comparison.
 */
function asIndex(key: "ndvi" | "ndbi"): AgentResult {
  const result = structuredClone(TEMPORAL_MARINA);
  result.evidence.items = result.evidence.items.map((item) => {
    const renamed = item.id
      .replace(".ndwi_mean", `.${key}_mean`)
      .replace(".ndwi_valid_pixel_count", `.${key}_valid_pixel_count`)
      .replace("mean_ndwi_difference", `mean_${key}_difference`)
      .replace("ndwi_change_", `${key}_change_`);
    return {
      ...item,
      id: renamed,
      measurement: item.measurement
        ? { ...item.measurement, name: renamed.split(".").pop() ?? item.measurement.name }
        : item.measurement,
    };
  });
  return result;
}

describe("a comparison names the index it compared", () => {
  it.each([
    ["ndvi", "vegetation", "Vegetation change", "NDVI"],
    ["ndbi", "built-up", "Built-up change", "NDBI"],
  ] as const)("%s", (key, word, title, code) => {
    const result = asIndex(key);
    const summary = resultSummary(result);
    expect(summary.kind).toBe("temporal");
    if (summary.kind !== "temporal") return;
    expect(summary.reading.index).toBe(key);
    expect(summary.reading.difference).not.toBeNull();

    const reading = interpretResult(result);
    expect(reading).not.toBeNull();
    const plain = [reading?.change, reading?.headline, reading?.explanation, reading?.howWeKnow]
      .join(" ")
      .toLowerCase();
    expect(plain).toContain(`${word} signal`);
    expect(plain).not.toContain("water");

    const technical = (technicalDetails(result) ?? []).map((row) => `${row.label} ${row.value}`).join(" ");
    expect(technical).toContain(code);
    expect(technical).not.toContain("NDWI");

    render(<AgentAnswerPanel result={result} />);
    expect(screen.getByText(title)).toBeInTheDocument();
    expect(screen.queryByText("Water change")).toBeNull();
  });

  it("water comparisons read exactly as before", () => {
    const summary = resultSummary(TEMPORAL_MARINA);
    expect(summary.kind === "temporal" && summary.reading.index).toBe("ndwi");
    expect(interpretResult(TEMPORAL_MARINA)?.change).toMatch(/water signal/);
  });
});
