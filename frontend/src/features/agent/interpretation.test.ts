import { describe, expect, it } from "vitest";

import type { AgentResult } from "../../api/types";
import {
  SMALL_SAMPLE_PIXELS,
  formatDay,
  interpretResult,
  technicalDetails,
  type Interpretation,
} from "./interpretation";
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
  SAHARA_POINT,
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

/** A copy with extra evidence items, e.g. a second index in the same run. */
function withItems(result: AgentResult, items: Json[]): AgentResult {
  const copy = structuredClone(result) as unknown as { evidence: { items: Json[] } };
  copy.evidence.items.push(...items);
  return copy as unknown as AgentResult;
}

const measured = (id: string, name: string, value: number, unit: string): Json => ({
  id, source: id.split(".")[0], measurement: { name, value, unit }, text: null, produced_by: "analysis", visual: null,
});

function interpreted(result: AgentResult): Interpretation {
  const interpretation = interpretResult(result);
  expect(interpretation).not.toBeNull();
  return interpretation as Interpretation;
}

/** Everything a person reads before opening Technical details. */
function primaryText(interpretation: Interpretation): string {
  return [
    interpretation.change ?? "",
    interpretation.headline,
    ...(interpretation.sampleWarnings ?? []),
    interpretation.explanation,
    interpretation.caveat ?? "",
    interpretation.howWeKnow,
  ].join(" ");
}

function technicalText(result: AgentResult): string {
  return (technicalDetails(result) ?? []).map((row) => `${row.label}: ${row.value}`).join("\n");
}

/** The required scientific boundary: it names a cause only to deny knowing it. */
const CHANGE_BOUNDARY = "It does not tell us what caused the change.";

/** Why pixels were left out - a fact about the data, not a claim about the place. */
const EXCLUSION_SENTENCE =
  /\d[\d,]* of \d[\d,]* pixels were left out because the satellite data there was not usable - for example because of cloud, shadow or missing data\./g;

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

// =========================================================================== //
// Layer 1 + 2, single-image indices
// =========================================================================== //

describe("interpretResult - vegetation, water, built-up, in plain English", () => {
  it("NDVI: a plain answer first, the number with its date and place, then how we know", () => {
    const { headline, explanation, howWeKnow, caveat } = interpreted(NDVI_CUBBON);
    expect(headline).toBe("The selected area shows a positive vegetation signal.");
    expect(explanation).toBe(
      "In the satellite image of Cubbon Park, Bengaluru taken on 8 December 2024, the vegetation index was +0.5204. Above zero means a positive vegetation signal.",
    );
    expect(howWeKnow).toContain("red light and invisible near-infrared light");
    expect(howWeKnow).toContain("The vegetation index of +0.5204 is the average over 16,988 usable pixels.");
    expect(howWeKnow).toContain(
      "92 of 17,080 pixels were left out because the satellite data there was not usable",
    );
    expect(caveat).toMatch(/^The vegetation index is an indicator/);
  });

  it("NDVI below zero is not called a vegetation signal", () => {
    const { headline, explanation } = interpreted(NDVI_NEGATIVE);
    expect(headline).toBe("The selected area does not show a positive vegetation signal on average.");
    expect(explanation).toContain("the vegetation index was -0.0613.");
    expect(explanation).toContain("Below zero means no positive vegetation signal on average.");
    expect(explanation).not.toContain("Above zero");
  });

  it("a mean shown as +0.0000 is zero, not positive", () => {
    const { headline, explanation } = interpreted(NDVI_ZERO);
    expect(headline).toBe("The vegetation index is zero on average.");
    expect(explanation).toContain("+0.0000");
    expect(explanation).toContain("neither a positive nor a negative vegetation signal");
  });

  it("NDWI: a positive water signal - an indicator, not a map of water", () => {
    const { headline, explanation, howWeKnow, caveat } = interpreted(NDWI_MARINA);
    expect(headline).toBe("The selected area shows a positive water signal.");
    expect(explanation).toContain("Marina Beach, Chennai taken on 4 January 2025, the water index was +0.1466.");
    expect(howWeKnow).toContain("green light with invisible near-infrared light");
    expect(howWeKnow).toContain("33,524 usable pixels");
    expect(caveat).toBe("The water index is an indicator, not a map of where water is.");
  });

  it("NDBI: no positive built-up signal - an indicator, not a building detector", () => {
    const { headline, explanation, howWeKnow, caveat } = interpreted(NDBI_AMEERPET);
    expect(headline).toBe("The selected area does not show a positive built-up signal on average.");
    expect(explanation).toContain("the built-up index was -0.0248.");
    expect(howWeKnow).toContain("196,620 usable pixels");
    // Nothing was masked at Ameerpet; no masking sentence is invented.
    expect(howWeKnow).not.toContain("left out");
    expect(caveat).toBe("The built-up index is an indicator, not a building detector.");
  });

  it("states only the optional fields the result carries", () => {
    const bare = withEvidence(NDVI_CUBBON, {
      "ndvi.ndvi_min": null,
      "ndvi.ndvi_max": null,
      "ndvi.ndvi_valid_pixel_count": null,
    });
    const copy = structuredClone(bare) as unknown as { evidence: { analysis: Json } };
    copy.evidence.analysis.pixel_quality = [];
    const { explanation, howWeKnow } = interpreted(copy as unknown as AgentResult);
    expect(explanation).toContain("the vegetation index was +0.5204.");
    expect(howWeKnow).not.toMatch(/average over|left out/);
  });

  it("formats a date the way people write it, never moving it", () => {
    expect(formatDay("2024-12-08")).toBe("8 December 2024");
    expect(formatDay("2025-01-04T05:15:13Z")).toBe("4 January 2025");
    expect(formatDay(null)).toBeNull();
    expect(formatDay("not a date")).toBe("not a date");
  });
});

// =========================================================================== //
// Radar
// =========================================================================== //

describe("interpretResult - radar, in plain English", () => {
  it("two measurements, their difference, and what they describe", () => {
    const { headline, explanation, howWeKnow, caveat } = interpreted(SAR_MARINA);
    expect(headline).toBe(
      "The radar image gave two measurements for the selected area: -5.44 dB and -17.85 dB.",
    );
    expect(explanation).toContain(
      "The difference between the two measurements was 12.41 dB: the first was the stronger of the two.",
    );
    expect(explanation).toContain(
      "In the radar image of Marina Beach, Chennai taken on 11 January 2025, these values describe how strongly the area reflected the radar signal back to the satellite.",
    );
    expect(howWeKnow).toContain("We used a radar image rather than a normal camera image");
    expect(howWeKnow).toContain("They are averages over 33,600 usable pixels.");
    expect(caveat).toContain("does not use them to decide what is on the ground");
  });

  it("with one polarization, says one measurement and no difference", () => {
    const vvOnly = withEvidence(SAR_MARINA, {
      "sar_backscatter.vh_mean_db": null,
      "sar_backscatter.vv_minus_vh_mean_db": null,
    });
    const { headline, explanation, howWeKnow } = interpreted(vvOnly);
    expect(headline).toBe("The radar image gave one measurement for the selected area: -5.44 dB.");
    expect(explanation).not.toContain("difference");
    expect(howWeKnow).not.toContain("two ways");
  });
});

// =========================================================================== //
// Two dates: what changed
// =========================================================================== //

describe("interpretResult - what changed, in plain English", () => {
  it("an increase: stronger, increased, both values, both dates, in order", () => {
    const { change, headline, explanation, howWeKnow, caveat } = interpreted(TEMPORAL_MARINA);
    expect(change).toBe("The water signal was stronger in January 2025 than in January 2024.");
    expect(headline).toBe("The later satellite image shows a stronger water signal than the earlier image.");
    expect(explanation).toBe(
      "Between the two dates, the water index increased from +0.0266 to +0.1466, a change of +0.1200.",
    );
    expect(howWeKnow).toContain("taken on 15 January 2024 and 4 January 2025");
    expect(howWeKnow).toContain(
      "Looking only at the 33,420 pixels usable in both images, the average change per pixel was +0.1207.",
    );
    expect(caveat).toBe(`This shows a change in the satellite measurement. ${CHANGE_BOUNDARY}`);
    expect(primaryText(interpreted(TEMPORAL_MARINA))).not.toMatch(/decreased|weaker/);
  });

  it("a decrease: weaker, decreased - the same pattern the other way", () => {
    const { change, headline, explanation } = interpreted(TEMPORAL_DECREASE);
    expect(change).toBe("The water signal was weaker in January 2025 than in January 2024.");
    expect(headline).toBe("The later satellite image shows a weaker water signal than the earlier image.");
    expect(explanation).toContain("decreased from +0.0403 to +0.0317, a change of -0.0086.");
    expect(primaryText(interpreted(TEMPORAL_DECREASE))).not.toMatch(/increased|stronger/);
  });

  it.each([
    ["+0.0000", TEMPORAL_FLAT],
    ["-0.0000", TEMPORAL_FLAT_NEGATIVE],
  ])("a difference shown as %s is little measured change - no direction claimed", (shown, result) => {
    const interpretation = interpreted(result);
    expect(interpretation.change).toBe("The water signal was about the same in January 2025 as in January 2024.");
    expect(interpretation.headline).toBe(
      "The two satellite images show little measured change in the water signal.",
    );
    expect(interpretation.explanation).toContain("little measured change");
    expect(interpretation.explanation).toContain(`a difference of ${shown} at the precision shown`);
    expect(primaryText(interpretation)).not.toMatch(/increased|decreased|stronger|weaker/);
  });

  it("keeps earlier and later in acquisition order, whatever the request's roles", () => {
    const { change, explanation } = interpreted(TEMPORAL_MARINA);
    expect(explanation.indexOf("+0.0266")).toBeLessThan(explanation.indexOf("+0.1466"));
    expect(change).toMatch(/January 2025 than in January 2024/);
  });

  it("a withheld difference states no change at all", () => {
    const withheld = withEvidence(TEMPORAL_MARINA, {
      "temporal_ndwi.difference.mean_ndwi_difference": null,
    });
    const interpretation = interpreted(withheld);
    expect(interpretation.change).toBe("No change is reported.");
    expect(interpretation.explanation).toContain("withheld the difference");
    expect(primaryText(interpretation)).not.toMatch(/increased|decreased|little measured change|stronger|weaker/);
  });

  it("without a paired-pixel change, says nothing about one", () => {
    const unpaired = withEvidence(TEMPORAL_MARINA, {
      "temporal_ndwi.change.ndwi_change_mean": null,
      "temporal_ndwi.change.paired_valid_pixel_count": null,
    });
    expect(interpreted(unpaired).howWeKnow).not.toContain("per pixel");
  });
});

// =========================================================================== //
// Status safety
// =========================================================================== //

describe("interpretResult - only a measured result is interpreted", () => {
  it.each([
    ["a clarification", CLARIFY_CHENNAI],
    ["a place not found", NOT_FOUND],
    ["a location outage", LOCATION_UNAVAILABLE],
    ["an area too large", AREA_TOO_LARGE_CHENNAI],
    ["a place that is only a point", SAHARA_POINT],
    ["an unsupported request", UNSUPPORTED_SHIPS],
    ["no measurement", NO_SCENES],
    ["an analysis not computed", REFUSED_BY_RADIOMETRY],
  ])("%s -> null (keeps its own wording), and no technical details", (_, result) => {
    expect(interpretResult(result)).toBeNull();
    expect(technicalDetails(result)).toBeNull();
  });

  it("a provider failure is not interpreted, even with measurements in its evidence", () => {
    const withheld = {
      ...NDVI_CUBBON,
      status: "answer_withheld",
      answer: null,
      failure: { stage: "answer_validation", message: "withheld" },
    } as unknown as AgentResult;
    expect(interpretResult(withheld)).toBeNull();
    expect(technicalDetails(withheld)).toBeNull();
  });
});

// =========================================================================== //
// Plain words first, specialist words last - and nothing invented anywhere
// =========================================================================== //

describe("interpretResult - layperson first, technical depth kept", () => {
  // Terms a person without remote-sensing training should not meet before
  // opening Technical details. Allowed there - and required there, below.
  const JARGON =
    /\b(NDVI|NDWI|NDBI|SAR|VV|VH|NIR|SWIR|AOI|STAC|RTC|L2A|B0\d|B1\d|backscatter|reflectance|radiometr\w*|affine|georeferenc\w*|multispectral|normali[sz]ed difference|pixel-wise|spectral|scene|gamma|polari[sz]ation|valid pixels?)\b/i;

  it.each(MEASURED)("%s: no jargon in the plain layers", (_, result) => {
    expect(primaryText(interpreted(result))).not.toMatch(JARGON);
  });

  it.each(MEASURED)("%s: the specialist terms are all in Technical details", (_, result) => {
    expect(technicalText(result)).toMatch(JARGON);
    expect(technicalText(result)).toMatch(/Radiometric check|Scene ID|observation/);
  });

  it("Technical details carry the index names, formulas, exact values and checks", () => {
    const ndvi = technicalText(NDVI_CUBBON);
    expect(ndvi).toContain("Index: NDVI - Normalized Difference Vegetation Index");
    expect(ndvi).toContain("NDVI formula: (NIR − Red) / (NIR + Red) · Sentinel-2 bands B08 and B04");
    expect(ndvi).toContain("NDVI mean: +0.5204 (exact 0.5204459203473597)");
    expect(ndvi).toContain("NDVI valid pixels: 16,988 of 17,080 (92 masked · sentinel-2-scl)");
    expect(ndvi).toContain("Scene ID: S2B_43PGQ_20241208_0_L2A");
    expect(ndvi).toContain("Sensor: Sentinel-2B · Level-2A");
    expect(ndvi).toMatch(/Radiometric check: verified/);
    expect(ndvi).toMatch(/Geometric check: grid verified \(ndvi\) · EPSG:32643 · 10 m/);

    const sar = technicalText(SAR_MARINA);
    expect(sar).toContain("VV (first measurement): -5.44 dB (exact -5.4437196664666985)");
    expect(sar).toContain("VH (second measurement): -17.85 dB (exact -17.849565)");
    expect(sar).toContain("VV − VH: 12.41 dB (exact 12.405845)");
    expect(sar).toContain("γ⁰");

    const change = technicalText(TEMPORAL_MARINA);
    expect(change).toContain("Mean difference: +0.1200 (exact 0.11998797364342559)");
    expect(change).toContain("Paired-pixel change: +0.1207 (exact 0.12069164662445794) over 33,420 pixels");
    expect(change).toMatch(/Earlier observation: January 2024 · S2A_44PMV_20240115_0_L2A/);
    expect(change).toMatch(/Later observation: January 2025 · S2B_44PMV_20250104_0_L2A/);
  });

  // Causes, events, judgements and qualities no measurement here can support.
  // The required boundary sentence names a cause only to deny knowing it.
  const PROHIBITED =
    /\b(flood\w*|drought\w*|rain\w*|monsoon\w*|season\w*|expan\w*|shr[ai]nk\w*|health\w*|dense|density|lush|thriv\w*|urbani[sz]\w*|significant\w*|caused|causing|because|due to|result(ed)? of|improv\w*|deteriorat\w*|degrad\w*|dried|climate|pollut\w*|encroach\w*|deforest\w*|yield|strong vegetation)\b/i;

  it.each(MEASURED)("%s: no causal, event or judgement claim, in any layer", (_, result) => {
    const everything = `${primaryText(interpreted(result))}\n${technicalText(result)}`;
    expect(everything.replace(CHANGE_BOUNDARY, "").replace(EXCLUSION_SENTENCE, "")).not.toMatch(
      PROHIBITED,
    );
  });

  it.each(MEASURED)("%s: every number in the plain layers is one the result carries", (_, result) => {
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
    // Dates and years are context, not measurements.
    const text = primaryText(interpreted(result)).replace(
      /\b\d{1,2} (January|February|March|April|May|June|July|August|September|October|November|December) \d{4}\b|\b(19|20)\d{2}\b/g,
      "",
    );
    const numbers = text.match(/[+-]?\d[\d,]*(\.\d+)?/g) ?? [];
    for (const number of numbers) {
      expect([...card, ...quality]).toContain(number);
    }
  });
});

// =========================================================================== //
// Small sample: said, never judged - behaviour unchanged, words plainer
// =========================================================================== //

describe("interpretResult - a small sample is said, never judged", () => {
  it("the live railway-stop case: 4 pixels, with the count as returned", () => {
    const interpretation = interpreted(NDVI_LALBAGH_STOP);
    expect(interpretation.sampleWarnings).toEqual([
      "Small sample: only 4 usable pixels contributed to this result, so interpret it cautiously.",
    ]);
    // The measurement and its sentence are exactly as before.
    expect(interpretation.headline).toBe("The selected area shows a positive vegetation signal.");
    expect(interpretation.explanation).toContain("the vegetation index was +0.3728.");
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
    expect(at(SMALL_SAMPLE_PIXELS - 1)?.[0]).toContain("only 99 usable pixels");
    expect(at(SMALL_SAMPLE_PIXELS)).toBeUndefined();
    expect(at(1)?.[0]).toContain("only 1 usable pixel contributed");
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
      "Small sample: only 12 usable pixels contributed to the water index, so interpret it cautiously.",
    ]);
  });

  it("radar: the returned count", () => {
    const few = withEvidence(SAR_MARINA, {
      "sar_backscatter.vv_valid_pixel_count": 9,
      "sar_backscatter.vh_valid_pixel_count": 9,
    });
    expect(interpreted(few).sampleWarnings).toEqual([
      "Small sample: only 9 usable pixels contributed to this result, so interpret it cautiously.",
    ]);
  });

  it("two dates: each image on its own, and the per-pixel change as a third sample", () => {
    const earlierOnly = withEvidence(TEMPORAL_MARINA, {
      "temporal_ndwi.first.ndwi_valid_pixel_count": 6,
    });
    expect(interpreted(earlierOnly).sampleWarnings).toEqual([
      "Small sample: only 6 usable pixels contributed to the earlier image (January 2024), so interpret it cautiously.",
    ]);

    const all = withEvidence(TEMPORAL_MARINA, {
      "temporal_ndwi.first.ndwi_valid_pixel_count": 6,
      "temporal_ndwi.second.ndwi_valid_pixel_count": 8,
      "temporal_ndwi.change.paired_valid_pixel_count": 5,
    });
    expect(interpreted(all).sampleWarnings).toEqual([
      "Small sample: only 6 usable pixels contributed to the earlier image (January 2024), so interpret it cautiously.",
      "Small sample: only 8 usable pixels contributed to the later image (January 2025), so interpret it cautiously.",
      "Small sample: only 5 pixels were usable in both images for the per-pixel change, so interpret it cautiously.",
    ]);
    // The change itself is stated exactly as before.
    expect(interpreted(all).explanation).toContain("increased from +0.0266 to +0.1466, a change of +0.1200.");
  });

  it("a withheld difference still says how few pixels each image had", () => {
    const withheld = withEvidence(TEMPORAL_MARINA, {
      "temporal_ndwi.difference.mean_ndwi_difference": null,
      "temporal_ndwi.second.ndwi_valid_pixel_count": 3,
    });
    expect(interpreted(withheld).sampleWarnings).toEqual([
      "Small sample: only 3 usable pixels contributed to the later image (January 2025), so interpret it cautiously.",
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
