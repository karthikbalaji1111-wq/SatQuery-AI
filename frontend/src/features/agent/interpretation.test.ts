import { describe, expect, it } from "vitest";

import type { AgentResult } from "../../api/types";
import { SMALL_SAMPLE_PIXELS, interpretResult, type Interpretation } from "./interpretation";
import {
  AREA_TOO_LARGE_CHENNAI,
  CLARIFY_CHENNAI,
  LOCATION_UNAVAILABLE,
  NDBI_AMEERPET,
  NDVI_CUBBON,
  NDVI_LALBAGH_STOP,
  NDWI_MARINA,
  NO_SCENES,
  NOT_FOUND,
  REFUSED_BY_RADIOMETRY,
  SAR_MARINA,
  TEMPORAL_MARINA,
  UNSUPPORTED_SHIPS,
} from "./m6Fixtures";

type Json = Record<string, unknown>;
type Item = { id: string; measurement: { value: number } | null };

/** A deep copy with evidence values replaced (or removed, with `null`). */
function withEvidence(result: AgentResult, values: Record<string, number | null>): AgentResult {
  const copy = structuredClone(result) as unknown as { evidence: { items: Item[] } };
  copy.evidence.items = copy.evidence.items
    .filter((item) => values[item.id] !== null)
    .map((item) =>
      item.id in values && item.measurement
        ? { ...item, measurement: { ...item.measurement, value: values[item.id] as number } }
        : item,
    );
  return copy as unknown as AgentResult;
}

function interpreted(result: AgentResult): Interpretation {
  const interpretation = interpretResult(result);
  expect(interpretation).not.toBeNull();
  return interpretation as Interpretation;
}

function allText(interpretation: Interpretation): string {
  return [
    interpretation.headline,
    ...(interpretation.sampleWarnings ?? []),
    interpretation.explanation,
    interpretation.caveat ?? "",
  ].join(" ");
}

/** Dal Lake, Srinagar, January 2024 vs January 2025 - live production values. */
const TEMPORAL_DECREASE = withEvidence(TEMPORAL_MARINA, {
  "temporal_ndwi.first.ndwi_mean": 0.04028,
  "temporal_ndwi.second.ndwi_mean": 0.03169,
  "temporal_ndwi.difference.mean_ndwi_difference": -0.008593,
  "temporal_ndwi.change.ndwi_change_mean": -0.0102,
});

const TEMPORAL_FLAT = withEvidence(TEMPORAL_MARINA, {
  "temporal_ndwi.first.ndwi_mean": 0.14655,
  "temporal_ndwi.second.ndwi_mean": 0.146583,
  "temporal_ndwi.difference.mean_ndwi_difference": 0.000033,
});

const TEMPORAL_FLAT_NEGATIVE = withEvidence(TEMPORAL_MARINA, {
  "temporal_ndwi.difference.mean_ndwi_difference": -0.00004,
});

/** Marina Beach NDVI, January 2025 - live: below zero. */
const NDVI_NEGATIVE = withEvidence(NDVI_CUBBON, { "ndvi.ndvi_mean": -0.06131 });
const NDVI_ZERO = withEvidence(NDVI_CUBBON, { "ndvi.ndvi_mean": 0.00002 });

describe("interpretResult - single-scene indices", () => {
  it("NDVI: the measured mean, its sign, the period, the scene and the place", () => {
    const { headline, explanation, caveat } = interpreted(NDVI_CUBBON);
    expect(headline).toBe("Positive vegetation-related signal");
    expect(explanation).toContain(
      "In the Sentinel-2 scene of 2024-12-08, selected for December 2024, the analysed area for Cubbon Park, Bengaluru had an average NDVI of +0.5204 across 16,988 valid pixels.",
    );
    expect(explanation).toContain("An average above zero indicates a positive vegetation-related signal");
    expect(explanation).toContain("ranged from -0.0877 to +0.8780");
    expect(explanation).toContain("92 of 17,080 pixels were excluded by pixel-quality masking");
    expect(caveat).toMatch(/^NDVI is a spectral index/);
  });

  it("NDVI below zero is not called a vegetation signal", () => {
    const { headline, explanation } = interpreted(NDVI_NEGATIVE);
    expect(headline).toBe("No positive vegetation-related signal on average");
    expect(explanation).toContain("average NDVI of -0.0613");
    expect(explanation).toContain("did not show a positive vegetation-related signal on average");
    expect(explanation).not.toContain("indicates a positive");
  });

  it("a mean shown as +0.0000 is zero, not positive", () => {
    const { headline, explanation } = interpreted(NDVI_ZERO);
    expect(headline).toBe("NDVI of zero on average");
    expect(explanation).toContain("+0.0000");
    expect(explanation).toContain("neither a positive nor a negative vegetation-related signal");
  });

  it("NDWI: a positive water-related signal, never a water body", () => {
    const { headline, explanation, caveat } = interpreted(NDWI_MARINA);
    expect(headline).toBe("Positive water-related signal");
    expect(explanation).toContain("selected for January 2025");
    expect(explanation).toContain("average NDWI of +0.1466 across 33,524 valid pixels");
    expect(caveat).toBe(
      "NDWI is a spectral index of the measured pixels, not a validated water classification.",
    );
  });

  it("NDBI: the measured built-up-related signal, never a land-cover class", () => {
    const { headline, explanation, caveat } = interpreted(NDBI_AMEERPET);
    expect(headline).toBe("No positive built-up-related signal on average");
    expect(explanation).toContain("Ameerpet, Hyderabad had an average NDBI of -0.0248 across 196,620 valid pixels");
    // Nothing was masked at Ameerpet; no masking sentence is invented.
    expect(explanation).not.toContain("excluded");
    expect(caveat).toContain("not a land-cover classification");
  });

  it("states only the optional fields the result carries", () => {
    const bare = withEvidence(NDVI_CUBBON, {
      "ndvi.ndvi_min": null,
      "ndvi.ndvi_max": null,
      "ndvi.ndvi_valid_pixel_count": null,
    });
    const copy = structuredClone(bare) as unknown as { evidence: { analysis: Json } };
    copy.evidence.analysis.pixel_quality = [];
    const { explanation } = interpreted(copy as unknown as AgentResult);
    expect(explanation).toContain("had an average NDVI of +0.5204.");
    expect(explanation).not.toMatch(/ranged|across|excluded/);
  });
});

describe("interpretResult - Sentinel-1 backscatter", () => {
  it("states VV, VH and their difference in dB, as measured", () => {
    const { headline, explanation, caveat } = interpreted(SAR_MARINA);
    expect(headline).toBe("Radar backscatter: VV -5.44 dB, VH -17.85 dB");
    expect(explanation).toContain(
      "In the Sentinel-1 scene of 2025-01-11, selected for January 2025, the analysed area for Marina Beach, Chennai recorded a mean VV backscatter of -5.44 dB and a mean VH backscatter of -17.85 dB, across 33,600 valid pixels.",
    );
    expect(explanation).toContain("The VV–VH difference was 12.41 dB: VV backscatter was stronger than VH on average.");
    expect(caveat).toContain("terrain-corrected");
  });

  it("with one polarization, says only that one", () => {
    const vvOnly = withEvidence(SAR_MARINA, {
      "sar_backscatter.vh_mean_db": null,
      "sar_backscatter.vv_minus_vh_mean_db": null,
    });
    const { headline, explanation } = interpreted(vvOnly);
    expect(headline).toBe("Radar backscatter: VV -5.44 dB");
    expect(explanation).not.toContain("VH");
  });
});

describe("interpretResult - what changed between two periods", () => {
  it("an increase: increased, stronger, both values, both periods, in order", () => {
    const { headline, explanation, caveat } = interpreted(TEMPORAL_MARINA);
    expect(headline).toBe("The water-related signal increased from January 2024 to January 2025");
    expect(explanation).toContain(
      "Between January 2024 and January 2025, the measured water-related signal increased from +0.0266 to +0.1466, a change of +0.1200.",
    );
    expect(explanation).toContain(
      "the later observation (2025-01-04) shows a stronger water-related signal than the earlier observation (2024-01-15)",
    );
    expect(explanation).toContain("Over the 33,420 pixels usable on both dates, the average per-pixel change was +0.1207.");
    expect(explanation).not.toContain("decreased");
    expect(caveat).toContain("it does not by itself establish the cause of that change");
  });

  it("a decrease: decreased, weaker - the same pattern the other way", () => {
    const { headline, explanation } = interpreted(TEMPORAL_DECREASE);
    expect(headline).toBe("The water-related signal decreased from January 2024 to January 2025");
    expect(explanation).toContain("decreased from +0.0403 to +0.0317, a change of -0.0086.");
    expect(explanation).toContain("shows a weaker water-related signal than the earlier observation");
    expect(explanation).not.toContain("increased");
  });

  it.each([
    ["+0.0000", TEMPORAL_FLAT],
    ["-0.0000", TEMPORAL_FLAT_NEGATIVE],
  ])("a difference shown as %s is little measured change - no direction claimed", (shown, result) => {
    const { headline, explanation } = interpreted(result);
    expect(headline).toBe("Little measured change in the water-related signal from January 2024 to January 2025");
    expect(explanation).toContain("showed little measured change");
    expect(explanation).toContain(`a difference of ${shown} at the displayed precision`);
    expect(allText(interpreted(result))).not.toMatch(/increased|decreased|stronger|weaker/);
  });

  it("keeps earlier and later in acquisition order, whatever the request's roles", () => {
    const { explanation } = interpreted(TEMPORAL_MARINA);
    expect(explanation.indexOf("January 2024")).toBeLessThan(explanation.indexOf("January 2025"));
    expect(explanation.indexOf("+0.0266")).toBeLessThan(explanation.indexOf("+0.1466"));
  });

  it("a withheld difference states no change at all", () => {
    const withheld = withEvidence(TEMPORAL_MARINA, {
      "temporal_ndwi.difference.mean_ndwi_difference": null,
    });
    const { headline, explanation } = interpreted(withheld);
    expect(headline).toBe("Change not reported");
    expect(explanation).toContain("withheld the difference");
    expect(explanation).not.toMatch(/increased|decreased|little measured change/);
  });

  it("without a paired-pixel change, says nothing about one", () => {
    const unpaired = withEvidence(TEMPORAL_MARINA, {
      "temporal_ndwi.change.ndwi_change_mean": null,
      "temporal_ndwi.change.paired_valid_pixel_count": null,
    });
    expect(interpreted(unpaired).explanation).not.toContain("per-pixel");
  });
});

describe("interpretResult - only a measured result is interpreted", () => {
  it.each([
    ["a clarification", CLARIFY_CHENNAI],
    ["a place not found", NOT_FOUND],
    ["a location outage", LOCATION_UNAVAILABLE],
    ["an area too large", AREA_TOO_LARGE_CHENNAI],
    ["an unsupported request", UNSUPPORTED_SHIPS],
    ["no measurement", NO_SCENES],
    ["an analysis not computed", REFUSED_BY_RADIOMETRY],
  ])("%s -> null (keeps its own wording)", (_, result) => {
    expect(interpretResult(result)).toBeNull();
  });

  it("a provider failure is not interpreted, even with measurements in its evidence", () => {
    const withheld = {
      ...NDVI_CUBBON,
      status: "answer_withheld",
      answer: null,
      failure: { stage: "answer_validation", message: "withheld" },
    } as unknown as AgentResult;
    expect(interpretResult(withheld)).toBeNull();
  });
});

describe("interpretResult - scientific grounding", () => {
  const MEASURED: [string, AgentResult][] = [
    ["NDVI", NDVI_CUBBON],
    ["NDVI over 4 pixels", NDVI_LALBAGH_STOP],
    ["NDVI below zero", NDVI_NEGATIVE],
    ["NDVI zero", NDVI_ZERO],
    ["NDWI", NDWI_MARINA],
    ["NDBI", NDBI_AMEERPET],
    ["SAR", SAR_MARINA],
    ["temporal increase", TEMPORAL_MARINA],
    ["temporal decrease", TEMPORAL_DECREASE],
    ["temporal little change", TEMPORAL_FLAT],
  ];

  // Causes, events, judgements and qualities no measurement here can support.
  const PROHIBITED =
    /\b(flood\w*|drought\w*|rain\w*|monsoon\w*|season\w*|expan\w*|shr[ai]nk\w*|health\w*|dense|density|lush|thriv\w*|urbani[sz]\w*|significant\w*|caused|causing|because|due to|result(ed)? of|improv\w*|deteriorat\w*|degrad\w*|dried|climate|pollut\w*|encroach\w*)\b/i;

  it.each(MEASURED)("%s: no causal, event or judgement claim", (_, result) => {
    expect(allText(interpreted(result))).not.toMatch(PROHIBITED);
  });

  it.each(MEASURED)("%s: every number stated is one the result carries", (_, result) => {
    const card = new Set(
      (result.evidence.items as Item[])
        .filter((item) => item.measurement)
        .flatMap((item) => {
          const value = (item.measurement as { value: number }).value;
          return [
            `${value >= 0 ? "+" : ""}${value.toFixed(4)}`,
            value.toFixed(2),
            Math.round(value).toLocaleString("en-US"),
          ];
        }),
    );
    const quality = (result.evidence.analysis?.pixel_quality ?? []).flatMap((q) => [
      q.masked_pixels.toLocaleString("en-US"),
      q.total_pixels.toLocaleString("en-US"),
    ]);
    // Dates, period years and sensor names are context, not measurements.
    const text = allText(interpreted(result)).replace(
      /\d{4}-\d{2}-\d{2}|\b(19|20)\d{2}\b|Sentinel-\d/g,
      "",
    );
    const numbers = text.match(/[+-]?\d[\d,]*(\.\d+)?/g) ?? [];
    for (const number of numbers) {
      expect([...card, ...quality]).toContain(number);
    }
  });
});

describe("interpretResult - a small sample is said, never judged", () => {
  /** A copy with extra evidence items, e.g. a second index in the same run. */
  function withItems(result: AgentResult, items: Json[]): AgentResult {
    const copy = structuredClone(result) as unknown as { evidence: { items: Json[] } };
    copy.evidence.items.push(...items);
    return copy as unknown as AgentResult;
  }
  const measured = (id: string, name: string, value: number, unit: string): Json => ({
    id, source: id.split(".")[0], measurement: { name, value, unit }, text: null, produced_by: "analysis", visual: null,
  });

  it("the live railway-stop case: 4 pixels, with the count as returned", () => {
    const interpretation = interpreted(NDVI_LALBAGH_STOP);
    expect(interpretation.sampleWarnings).toEqual([
      "Small sample: only 4 valid pixels contributed to this result, so interpret it cautiously.",
    ]);
    // The measurement and its sentence are exactly as before.
    expect(interpretation.headline).toBe("Positive vegetation-related signal");
    expect(interpretation.explanation).toContain("average NDVI of +0.3728 across 4 valid pixels");
  });

  it("a large sample carries no warning", () => {
    for (const result of [NDVI_CUBBON, NDWI_MARINA, NDBI_AMEERPET, SAR_MARINA, TEMPORAL_MARINA]) {
      expect(interpreted(result).sampleWarnings).toBeUndefined();
    }
  });

  it("the boundary is the documented presentation rule, and it is only that", () => {
    expect(SMALL_SAMPLE_PIXELS).toBe(100);
    const at = (count: number) =>
      interpreted(withEvidence(NDVI_CUBBON, { "ndvi.ndvi_valid_pixel_count": count })).sampleWarnings;
    expect(at(SMALL_SAMPLE_PIXELS - 1)?.[0]).toContain("only 99 valid pixels");
    expect(at(SMALL_SAMPLE_PIXELS)).toBeUndefined();
    expect(at(1)?.[0]).toContain("only 1 valid pixel contributed");
  });

  it("no count, no warning - a count is never guessed", () => {
    const uncounted = withEvidence(NDVI_LALBAGH_STOP, { "ndvi.ndvi_valid_pixel_count": null });
    expect(interpreted(uncounted).sampleWarnings).toBeUndefined();
  });

  it("names which index, when several were measured", () => {
    const both = withItems(NDVI_CUBBON, [
      measured("ndwi.ndwi_mean", "ndwi_mean", -0.2, "index"),
      measured("ndwi.ndwi_valid_pixel_count", "ndwi_valid_pixel_count", 12, "pixels"),
    ]);
    expect(interpreted(both).sampleWarnings).toEqual([
      "Small sample: only 12 valid pixels contributed to the NDWI result, so interpret it cautiously.",
    ]);
  });

  it("SAR: the returned count", () => {
    const few = withEvidence(SAR_MARINA, {
      "sar_backscatter.vv_valid_pixel_count": 9,
      "sar_backscatter.vh_valid_pixel_count": 9,
    });
    expect(interpreted(few).sampleWarnings).toEqual([
      "Small sample: only 9 valid pixels contributed to this result, so interpret it cautiously.",
    ]);
  });

  it("temporal: each observation on its own, and the paired change as a third sample", () => {
    const earlierOnly = withEvidence(TEMPORAL_MARINA, {
      "temporal_ndwi.first.ndwi_valid_pixel_count": 6,
    });
    expect(interpreted(earlierOnly).sampleWarnings).toEqual([
      "Small sample: only 6 valid pixels contributed to the earlier observation (January 2024), so interpret it cautiously.",
    ]);

    const all = withEvidence(TEMPORAL_MARINA, {
      "temporal_ndwi.first.ndwi_valid_pixel_count": 6,
      "temporal_ndwi.second.ndwi_valid_pixel_count": 8,
      "temporal_ndwi.change.paired_valid_pixel_count": 5,
    });
    expect(interpreted(all).sampleWarnings).toEqual([
      "Small sample: only 6 valid pixels contributed to the earlier observation (January 2024), so interpret it cautiously.",
      "Small sample: only 8 valid pixels contributed to the later observation (January 2025), so interpret it cautiously.",
      "Small sample: only 5 pixels were usable on both dates for the paired-pixel change, so interpret it cautiously.",
    ]);
    // The change itself is stated exactly as before.
    expect(interpreted(all).explanation).toContain("increased from +0.0266 to +0.1466, a change of +0.1200.");
  });

  it("a withheld difference still says how few pixels each observation had", () => {
    const withheld = withEvidence(TEMPORAL_MARINA, {
      "temporal_ndwi.difference.mean_ndwi_difference": null,
      "temporal_ndwi.second.ndwi_valid_pixel_count": 3,
    });
    expect(interpreted(withheld).sampleWarnings).toEqual([
      "Small sample: only 3 valid pixels contributed to the later observation (January 2025), so interpret it cautiously.",
    ]);
  });

  it("never declares invalidity, significance or confidence", () => {
    const warnings = [
      ...(interpreted(NDVI_LALBAGH_STOP).sampleWarnings ?? []),
      ...(interpreted(
        withEvidence(TEMPORAL_MARINA, {
          "temporal_ndwi.first.ndwi_valid_pixel_count": 1,
          "temporal_ndwi.change.paired_valid_pixel_count": 1,
        }),
      ).sampleWarnings ?? []),
    ].join(" ");
    expect(warnings).not.toMatch(/invalid|unreliable|significan|insignifican|confiden|statistic|meaningless|not meaningful|error bar|uncertain/i);
  });
});
