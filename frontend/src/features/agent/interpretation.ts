/**
 * What a measured result means - in plain English first, technical terms last.
 *
 * Three layers, all derived, none generated:
 *
 *   1. What this means - one or two sentences a person with no remote-sensing
 *      background can read in five seconds. No acronyms, no formulas.
 *   2. How we know     - how the number was obtained, in everyday words.
 *   3. Technical details - the exact values, index names, formulas, sensor,
 *      scene, pixel counts and validation records, acronyms included
 *      (`technicalDetails`).
 *
 * Every sentence is a fixed template filled ONLY from values the result model
 * already read out of the backend's response: the operation, the place as
 * asked, the requested period, the satellite image's date, the measurements,
 * the pixel counts and the validation records. No model, no network, no
 * outside knowledge, and no cause.
 *
 * The only boundary used anywhere is ZERO, the natural midpoint of a
 * normalised difference - so the words are "a positive vegetation signal",
 * never "strong", "healthy" or "dense": no threshold for those exists in the
 * product, so none is implied. "Zero" means zero at the precision the card
 * displays (a mean shown as +0.0000 is not called positive), which is a
 * statement about the display, not a significance test; none exists in the
 * backend, so none is claimed.
 *
 * Only a successful measured result is interpreted. A clarification, a
 * refusal, a location problem, an analysis that was not computed, a provider
 * failure and an imagery-only result all return `null` and keep their own
 * wording.
 */

import type { AgentResult } from "../../api/types";
import { decibels, pixelCount, signedIndex } from "./format";
import {
  outcomeOf,
  resultContext,
  resultSummary,
  validationRecords,
  type IndexReading,
  type PeriodReading,
  type ResultBody,
  type ResultContext,
  type SarReading,
  type SceneRef,
  type TemporalReading,
} from "./resultModel";

export interface Interpretation {
  /** Comparisons only: the one-line answer to "what changed?". */
  change?: string;
  /** What this means - the one-line answer. */
  headline: string;
  /** What this means - the supporting sentence(s), with the number. */
  explanation: string;
  /** The limit of the statement, in plain words. */
  caveat?: string;
  /**
   * "Very few pixels contributed" - one per measurement that rests on fewer
   * than `SMALL_SAMPLE_PIXELS`. Present only when there is one to state.
   */
  sampleWarnings?: string[];
  /** How we know - how the number was obtained, in everyday words. */
  howWeKnow: string;
}

/**
 * A PRESENTATION safeguard, not a scientific validity threshold.
 *
 * The backend defines no minimum sample: its only sample rule is the exact
 * degenerate case of at most one pixel (min == max == mean, temporal
 * `_sample_warnings`), and a stronger one was deliberately not invented,
 * because a defensible minimum would depend on spatial autocorrelation and an
 * effective sample size this system does not estimate. So this number decides
 * WORDING only: below it, the reader is told how few pixels contributed and
 * asked to read the value cautiously. It changes no measurement, marks nothing
 * invalid, and implies no confidence level or significance. 100 pixels of the
 * 10 m analysis grid - every Sentinel-2 index grid and the Sentinel-1 RTC grid
 * are 10 m - is about one hectare: a round, conservative size that includes the
 * backend's one-pixel case.
 */
export const SMALL_SAMPLE_PIXELS = 100;

/** Each index in everyday words - a signal and an indicator, never a class. */
const PLAIN: Record<
  IndexReading["key"],
  { signal: string; index: string; how: string; caveat: string }
> = {
  ndvi: {
    signal: "vegetation signal",
    index: "vegetation index",
    how: "We compared two kinds of light the satellite records - red light and invisible near-infrared light - which plants reflect very differently.",
    caveat:
      "The vegetation index is an indicator: it does not tell us what kind of plants are there or what condition they are in.",
  },
  ndwi: {
    signal: "water signal",
    index: "water index",
    how: "We compared green light with invisible near-infrared light, which water and land reflect very differently.",
    caveat: "The water index is an indicator, not a map of where water is.",
  },
  ndbi: {
    signal: "built-up signal",
    index: "built-up index",
    how: "We compared two kinds of invisible infrared light that built-up surfaces and plants reflect differently.",
    caveat: "The built-up index is an indicator, not a building detector.",
  },
};

/** The same indices, named as a specialist would: for Technical details only. */
const TECHNICAL: Record<IndexReading["key"], { name: string; formula: string; grid: string }> = {
  ndvi: {
    name: "NDVI - Normalized Difference Vegetation Index",
    formula: "(NIR − Red) / (NIR + Red) · Sentinel-2 bands B08 and B04",
    grid: "10 m (native)",
  },
  ndwi: {
    name: "NDWI - Normalized Difference Water Index",
    formula: "(Green − NIR) / (Green + NIR) · Sentinel-2 bands B03 and B08",
    grid: "10 m (native)",
  },
  ndbi: {
    name: "NDBI - Normalized Difference Built-up Index",
    formula: "(SWIR − NIR) / (SWIR + NIR) · Sentinel-2 bands B11 (20 m) and B08 (10 m)",
    grid: "10 m grid; the 20 m SWIR band limits detail to 20 m",
  },
};

const INDEX_INPUTS =
  "Raw digital numbers: both bands share one scale, which cancels in the ratio; no offset is applied.";

const CHANGE_CAVEAT =
  "This shows a change in the satellite measurement. It does not tell us what caused the change.";

const SAR_CAVEAT =
  "These values come from the satellite provider's processed radar product. SatQuery does not use them to decide what is on the ground.";

type Sign = "positive" | "negative" | "zero";

/**
 * The sign a reader SEES. A value the card shows as +0.0000 (or 0.00 dB) is
 * zero here too, so a sentence never calls a displayed zero "positive".
 */
function shownSign(shown: string): Sign {
  const value = Number.parseFloat(shown);
  return value > 0 ? "positive" : value < 0 ? "negative" : "zero";
}

const MONTH_NAMES = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
];

/** "2024-12-08" -> "8 December 2024". Anything else is returned unchanged. */
export function formatDay(iso: string | null | undefined): string | null {
  if (!iso) return null;
  const match = /^(\d{4})-(\d{2})-(\d{2})/.exec(iso);
  if (!match) return iso;
  const month = MONTH_NAMES[Number(match[2]) - 1];
  return month ? `${Number(match[3])} ${month} ${match[1]}` : iso;
}

function pixels(count: number): string {
  return `${pixelCount(count)} usable ${count === 1 ? "pixel" : "pixels"}`;
}

/** "In the satellite image of <place> taken on <day>" - only what is known. */
function imageOf(context: ResultContext, scene: SceneRef | null, kind = "satellite image"): string {
  const place = context.location ? ` of ${context.location}` : "";
  const day = formatDay(scene?.acquired);
  const period = context.periods[0] ?? null;
  if (day) return `In the ${kind}${place} taken on ${day}`;
  if (period) return `In the ${kind}${place} for ${period}`;
  return `In the ${kind}${place}`;
}

function excluded(reading: { quality: IndexReading["quality"] }): string | null {
  const quality = reading.quality;
  if (quality === null || quality.masked <= 0) return null;
  return `${pixelCount(quality.masked)} of ${pixelCount(quality.total)} pixels were left out because the satellite data there was not usable - for example because of cloud, shadow or missing data.`;
}

// --------------------------------------------------------------------------- //
// Single-image indices
// --------------------------------------------------------------------------- //

function indexHeadline(reading: IndexReading): string {
  const { signal, index } = PLAIN[reading.key];
  switch (shownSign(signedIndex(reading.mean))) {
    case "positive":
      return `The selected area shows a positive ${signal}.`;
    case "negative":
      return `The selected area does not show a positive ${signal} on average.`;
    case "zero":
      return `The ${index} is zero on average.`;
  }
}

function indexMeaning(reading: IndexReading, lead: string | null): string {
  const { signal, index } = PLAIN[reading.key];
  const value = signedIndex(reading.mean);
  const first = lead ? `${lead}, the ${index} was ${value}.` : `The ${index} was ${value}.`;
  switch (shownSign(value)) {
    case "positive":
      return `${first} Above zero means a positive ${signal}.`;
    case "negative":
      return `${first} Below zero means no positive ${signal} on average.`;
    case "zero":
      return `${first} That is zero at the precision shown - neither a positive nor a negative ${signal}.`;
  }
}

function interpretIndices(
  readings: IndexReading[],
  scene: SceneRef | null,
  context: ResultContext,
): Interpretation {
  const lead = imageOf(context, scene);
  const how = readings.flatMap((reading) => {
    const { index, how: method } = PLAIN[reading.key];
    const counted =
      reading.validPixels !== null
        ? `The ${index} of ${signedIndex(reading.mean)} is the average over ${pixels(reading.validPixels)}.`
        : null;
    return [method, counted, excluded(reading)].filter((line): line is string => line !== null);
  });
  const caveats = [...new Set(readings.map((reading) => PLAIN[reading.key].caveat))];
  return {
    headline: readings.map(indexHeadline).join(" "),
    explanation: readings
      .map((reading, position) => indexMeaning(reading, position === 0 ? lead : null))
      .join(" "),
    caveat: caveats.join(" "),
    howWeKnow: how.join(" "),
  };
}

// --------------------------------------------------------------------------- //
// Radar
// --------------------------------------------------------------------------- //

function interpretSar(
  reading: SarReading,
  scene: SceneRef | null,
  context: ResultContext,
): Interpretation {
  const values = [reading.vv, reading.vh].filter((value): value is number => value !== null);
  const headline =
    values.length === 2
      ? `The radar image gave two measurements for the selected area: ${decibels(values[0])} and ${decibels(values[1])}.`
      : `The radar image gave one measurement for the selected area: ${decibels(values[0])}.`;

  const sentences: string[] = [];
  if (reading.vv !== null && reading.vh !== null && reading.difference !== null) {
    const shown = decibels(reading.difference);
    sentences.push(
      {
        positive: `The difference between the two measurements was ${shown}: the first was the stronger of the two.`,
        negative: `The difference between the two measurements was ${shown}: the second was the stronger of the two.`,
        zero: `The difference between the two measurements was ${shown}: they were the same at the precision shown.`,
      }[shownSign(shown)],
    );
  }
  sentences.push(
    `${imageOf(context, scene, "radar image")}, these values describe how strongly the area reflected the radar signal back to the satellite.`,
  );

  const how = [
    "We used a radar image rather than a normal camera image: the satellite sends out its own radar signal and records how much of it comes back.",
    values.length === 2 ? "It records this in two ways, which gives the two measurements." : null,
    reading.validPixels !== null
      ? `They are averages over ${pixels(reading.validPixels)}.`
      : null,
  ].filter((line): line is string => line !== null);

  return {
    headline,
    explanation: sentences.join(" "),
    caveat: SAR_CAVEAT,
    howWeKnow: how.join(" "),
  };
}

// --------------------------------------------------------------------------- //
// Two dates: what changed
// --------------------------------------------------------------------------- //

function interpretTemporal(reading: TemporalReading): Interpretation {
  const { earlier, later, difference, pairedChange, pairedPixels } = reading;
  const from = earlier.period ?? formatDay(earlier.acquired) ?? "the earlier date";
  const to = later.period ?? formatDay(later.acquired) ?? "the later date";
  const days = [formatDay(earlier.acquired), formatDay(later.acquired)];
  const compared =
    days[0] && days[1]
      ? `We compared satellite images of the same area taken on ${days[0]} and ${days[1]}, measuring the water index in each.`
      : "We compared satellite measurements from two different dates, measuring the water index in each.";
  const paired =
    pairedChange !== null
      ? `Looking only at the ${pairedPixels !== null ? `${pixelCount(pairedPixels)} ` : ""}pixels usable in both images, the average change per pixel was ${signedIndex(pairedChange)}.`
      : null;
  const howWeKnow = [compared, paired].filter((line): line is string => line !== null).join(" ");

  if (difference === null) {
    return {
      change: "No change is reported.",
      headline: "The difference between the two images was not reported.",
      explanation: `Both dates were measured, but the analysis withheld the difference between ${from} and ${to}, so no change is stated.`,
      caveat: "The technical details record why the difference was withheld.",
      howWeKnow,
    };
  }

  const shown = signedIndex(difference);
  const means =
    earlier.mean !== null && later.mean !== null
      ? ` from ${signedIndex(earlier.mean)} to ${signedIndex(later.mean)}`
      : "";
  const sign = shownSign(shown);
  if (sign === "zero") {
    return {
      change: `The water signal was about the same in ${to} as in ${from}.`,
      headline: "The two satellite images show little measured change in the water signal.",
      explanation: `Between the two dates, the water index showed little measured change${means ? `, going${means}` : ""}: a difference of ${shown} at the precision shown.`,
      caveat: CHANGE_CAVEAT,
      howWeKnow,
    };
  }

  const [verb, relative] = sign === "positive" ? ["increased", "stronger"] : ["decreased", "weaker"];
  return {
    change: `The water signal was ${relative} in ${to} than in ${from}.`,
    headline: `The later satellite image shows a ${relative} water signal than the earlier image.`,
    explanation: `Between the two dates, the water index ${verb}${means}, a change of ${shown}.`,
    caveat: CHANGE_CAVEAT,
    howWeKnow,
  };
}

// --------------------------------------------------------------------------- //
// How few pixels a value rests on
// --------------------------------------------------------------------------- //

/** The warning for one measurement, from its returned count - or none. */
function smallSample(count: number | null, subject: string): string | null {
  // No count, no claim: a missing count is never guessed at.
  if (count === null || count < 1 || count >= SMALL_SAMPLE_PIXELS) return null;
  return `Small sample: only ${pixels(count)} contributed to ${subject}, so interpret it cautiously.`;
}

function sampleWarnings(
  body: Extract<ResultBody, { kind: "index" | "sar" | "temporal" }>,
): string[] {
  switch (body.kind) {
    case "index":
      return body.readings
        .map((reading) =>
          smallSample(
            reading.validPixels,
            body.readings.length === 1 ? "this result" : `the ${PLAIN[reading.key].index}`,
          ),
        )
        .filter((warning): warning is string => warning !== null);
    case "sar":
      return [smallSample(body.reading.validPixels, "this result")].filter(
        (warning): warning is string => warning !== null,
      );
    case "temporal": {
      // Each observation is its own sample, and the paired change a third.
      const { earlier, later, pairedChange, pairedPixels } = body.reading;
      const side = (reading: PeriodReading, role: string) =>
        smallSample(
          reading.validPixels,
          `the ${role} image${reading.period ? ` (${reading.period})` : ""}`,
        );
      const paired =
        pairedChange !== null &&
        pairedPixels !== null &&
        pairedPixels >= 1 &&
        pairedPixels < SMALL_SAMPLE_PIXELS
          ? `Small sample: only ${pixelCount(pairedPixels)} ${pairedPixels === 1 ? "pixel was" : "pixels were"} usable in both images for the per-pixel change, so interpret it cautiously.`
          : null;
      return [side(earlier, "earlier"), side(later, "later"), paired].filter(
        (warning): warning is string => warning !== null,
      );
    }
  }
}

// --------------------------------------------------------------------------- //

/**
 * The plain-language reading of a successful measured result, or `null` when
 * there is nothing measured to interpret.
 */
export function interpretResult(result: AgentResult): Interpretation | null {
  if (outcomeOf(result).kind !== "success") return null;
  const body = resultSummary(result);
  if (body.kind === "imagery" || body.kind === "none") return null;
  const context = resultContext(result);
  const interpretation =
    body.kind === "index"
      ? interpretIndices(body.readings, body.scene, context)
      : body.kind === "sar"
        ? interpretSar(body.reading, body.scene, context)
        : interpretTemporal(body.reading);
  const warnings = sampleWarnings(body);
  return warnings.length > 0 ? { ...interpretation, sampleWarnings: warnings } : interpretation;
}

// --------------------------------------------------------------------------- //
// Layer 3: Technical details - specialist terms, exact values
// --------------------------------------------------------------------------- //

export interface TechnicalRow {
  label: string;
  value: string;
}

/** Exact as the API returned it, beside the rounded value the card shows. */
function exact(shown: string, value: number): string {
  return `${shown} (exact ${value})`;
}

function platformName(platform: string | null): string | null {
  if (!platform) return null;
  return platform.replace(
    /^sentinel-(\d)([a-z]?)$/i,
    (_, number: string, unit: string) => `Sentinel-${number}${unit.toUpperCase()}`,
  );
}

const RADIOMETRY_WORDS: Record<string, string> = {
  verified: "verified",
  verified_with_unknown_metadata: "verified · some metadata not published",
  incompatible: "refused · incompatible",
  undetermined: "refused · undetermined",
};

function checkRows(result: AgentResult): TechnicalRow[] {
  const records = validationRecords(result.evidence.analysis);
  const rows: TechnicalRow[] = [];
  if (records.radiometry.length > 0) {
    rows.push({
      label: "Radiometric check",
      value: records.radiometry
        .map(
          (state) =>
            `${RADIOMETRY_WORDS[state.status] ?? state.status}${state.processing_baseline ? ` (baseline ${state.processing_baseline})` : ""} · ${state.scene_id}`,
        )
        .join("; "),
    });
  }
  if (records.grids.length > 0) {
    rows.push({
      label: "Geometric check",
      value: records.grids
        .map((grid) =>
          grid.status === "valid"
            ? [
                `grid verified (${grid.analysis})`,
                grid.crs,
                grid.resolution_x !== null ? `${grid.resolution_x} m` : null,
                grid.width !== null && grid.height !== null ? `${grid.width} × ${grid.height} px` : null,
              ]
                .filter((part): part is string => Boolean(part))
                .join(" · ")
            : `refused (${grid.analysis}): ${grid.refusal ?? "no reason recorded"}`,
        )
        .join("; "),
    });
  }
  return rows;
}

function sceneRows(scene: SceneRef | null, product: string): TechnicalRow[] {
  if (scene === null) return [];
  return [
    { label: "Sensor", value: [platformName(scene.platform), product].filter(Boolean).join(" · ") },
    { label: "Scene ID", value: scene.id },
    ...(scene.acquired ? [{ label: "Acquired", value: scene.acquired }] : []),
  ];
}

function indexRows(readings: IndexReading[], scene: SceneRef | null): TechnicalRow[] {
  const rows = readings.flatMap((reading) => {
    const spec = TECHNICAL[reading.key];
    const own: TechnicalRow[] = [
      { label: "Index", value: spec.name },
      { label: `${reading.label} formula`, value: spec.formula },
      { label: `${reading.label} mean`, value: exact(signedIndex(reading.mean), reading.mean) },
    ];
    if (reading.min !== null && reading.max !== null) {
      own.push({
        label: `${reading.label} range`,
        value: `${signedIndex(reading.min)} to ${signedIndex(reading.max)}`,
      });
    }
    if (reading.validPixels !== null) {
      const quality = reading.quality;
      own.push({
        label: `${reading.label} valid pixels`,
        value:
          quality !== null
            ? `${pixelCount(reading.validPixels)} of ${pixelCount(quality.total)} (${pixelCount(quality.masked)} masked · ${quality.source})`
            : pixelCount(reading.validPixels),
      });
    }
    own.push({ label: `${reading.label} grid`, value: spec.grid });
    return own;
  });
  return [...rows, { label: "Inputs", value: INDEX_INPUTS }, ...sceneRows(scene, "Level-2A")];
}

function sarRows(reading: SarReading, scene: SceneRef | null): TechnicalRow[] {
  const rows: TechnicalRow[] = [
    {
      label: "Measurement",
      value: "Sentinel-1 backscatter, γ⁰ (gamma naught), provider radiometrically terrain-corrected (RTC)",
    },
  ];
  if (reading.vv !== null) {
    rows.push({ label: "VV (first measurement)", value: exact(decibels(reading.vv), reading.vv) });
  }
  if (reading.vh !== null) {
    rows.push({ label: "VH (second measurement)", value: exact(decibels(reading.vh), reading.vh) });
  }
  if (reading.difference !== null) {
    rows.push({ label: "VV − VH", value: exact(decibels(reading.difference), reading.difference) });
  }
  rows.push({
    label: "Formula",
    value: "dB = 10 · log10(mean linear power) - averaged in linear power, then converted",
  });
  if (reading.validPixels !== null) {
    rows.push({ label: "Valid pixels", value: pixelCount(reading.validPixels) });
  }
  return [...rows, ...sceneRows(scene, "RTC")];
}

function temporalRows(reading: TemporalReading): TechnicalRow[] {
  const side = (period: PeriodReading): TechnicalRow => ({
    label: period.role === "Earlier" ? "Earlier observation" : "Later observation",
    value: [
      period.period,
      period.sceneId,
      period.acquired ? `acquired ${period.acquired}` : null,
      period.mean !== null ? `NDWI mean ${exact(signedIndex(period.mean), period.mean)}` : null,
      period.validPixels !== null ? `${pixelCount(period.validPixels)} valid pixels` : null,
    ]
      .filter((part): part is string => Boolean(part))
      .join(" · "),
  });
  const rows: TechnicalRow[] = [
    { label: "Index", value: TECHNICAL.ndwi.name },
    { label: "NDWI formula", value: TECHNICAL.ndwi.formula },
    {
      label: "Change formulas",
      value:
        "Mean difference = later mean − earlier mean; paired-pixel change = mean of (later − earlier) over pixels valid on both dates, on a verified identical grid",
    },
    side(reading.earlier),
    side(reading.later),
  ];
  if (reading.difference !== null) {
    rows.push({
      label: "Mean difference",
      value: exact(signedIndex(reading.difference), reading.difference),
    });
  }
  if (reading.pairedChange !== null) {
    rows.push({
      label: "Paired-pixel change",
      value: `${exact(signedIndex(reading.pairedChange), reading.pairedChange)}${reading.pairedPixels !== null ? ` over ${pixelCount(reading.pairedPixels)} pixels` : ""}`,
    });
  }
  return rows;
}

/**
 * Layer 3 - the specialist record of a measured result: index names,
 * formulas, exact values, sensor, scene and validation checks. `null` exactly
 * when `interpretResult` is `null`.
 */
export function technicalDetails(result: AgentResult): TechnicalRow[] | null {
  if (outcomeOf(result).kind !== "success") return null;
  const body = resultSummary(result);
  switch (body.kind) {
    case "index":
      return [...indexRows(body.readings, body.scene), ...checkRows(result)];
    case "sar":
      return [...sarRows(body.reading, body.scene), ...checkRows(result)];
    case "temporal":
      return [...temporalRows(body.reading), ...checkRows(result)];
    case "imagery":
    case "none":
      return null;
  }
}
