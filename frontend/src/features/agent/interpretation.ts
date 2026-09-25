/**
 * What a measured result means, in plain English - derived, never generated.
 *
 * Every sentence here is a fixed template filled ONLY from values the result
 * model already read out of the backend's response: the operation, the place
 * as asked, the requested period, the selected scene's acquisition date, the
 * measurements and the pixel-quality counts. No model, no network, no outside
 * knowledge, and no cause: the sentences say what the numbers are and what
 * their sign means for that index, and stop there.
 *
 * The only boundary used anywhere is ZERO, the natural midpoint of a
 * normalised difference - no threshold is invented. "Zero" means zero at the
 * precision the card displays (a mean shown as +0.0000 is not called
 * positive), which is a statement about the display, not a significance test;
 * none exists in the backend, so none is claimed.
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
  type IndexReading,
  type ResultBody,
  type ResultContext,
  type SarReading,
  type SceneRef,
  type TemporalReading,
} from "./resultModel";

export interface Interpretation {
  /** One line, answering the question the result was asked. */
  headline: string;
  /** What the measured values say, in full sentences. */
  explanation: string;
  /** The scientific boundary of the statement, when there is one to state. */
  caveat?: string;
  /**
   * "Very few pixels contributed" - one per measurement that rests on fewer
   * than `SMALL_SAMPLE_PIXELS`. Present only when there is one to state.
   */
  sampleWarnings?: string[];
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

/** What each index measures, named as a signal - never as a land-cover class. */
const SIGNAL: Record<IndexReading["key"], string> = {
  ndvi: "vegetation-related signal",
  ndwi: "water-related signal",
  ndbi: "built-up-related signal",
};

const INDEX_CAVEAT: Record<IndexReading["key"], string> = {
  ndvi:
    "NDVI is a spectral index of the measured pixels; on its own it does not describe the condition or type of the vegetation on the ground.",
  ndwi: "NDWI is a spectral index of the measured pixels, not a validated water classification.",
  ndbi: "NDBI is a spectral index of the measured pixels, not a land-cover classification.",
};

const TEMPORAL_CAVEAT =
  "This measurement indicates a change in the satellite-derived water signal; it does not by itself establish the cause of that change. No statistical test was applied to the difference.";

const TEMPORAL_FLAT_CAVEAT =
  "This compares the satellite-derived water signal of two scenes; no statistical test was applied to the difference.";

const SAR_CAVEAT =
  "These are mean values from the provider's terrain-corrected Sentinel-1 product; SatQuery applies no speckle filtering and no land-cover classification to them.";

type Sign = "positive" | "negative" | "zero";

/**
 * The sign a reader SEES. A value the card shows as +0.0000 (or 0.00 dB) is
 * zero here too, so a sentence never calls a displayed zero "positive".
 */
function shownSign(shown: string): Sign {
  const value = Number.parseFloat(shown);
  return value > 0 ? "positive" : value < 0 ? "negative" : "zero";
}

/** The area as the question named it; the geocoder's match is shown beside it. */
function area(context: ResultContext): string {
  return context.location ? `the analysed area for ${context.location}` : "the analysed area";
}

/** When the numbers were measured: the scene's own date, and why that scene. */
function when(sensor: string, scene: SceneRef | null, period: string | null): string {
  const acquired = scene?.acquired ?? null;
  if (acquired && period) return `In the ${sensor} scene of ${acquired}, selected for ${period}`;
  if (acquired) return `In the ${sensor} scene of ${acquired}`;
  if (period) return `For ${period}`;
  return `In the selected ${sensor} scene`;
}

// --------------------------------------------------------------------------- //
// Single-scene spectral indices
// --------------------------------------------------------------------------- //

function indexHeadline(reading: IndexReading): string {
  const signal = SIGNAL[reading.key];
  switch (shownSign(signedIndex(reading.mean))) {
    case "positive":
      return `Positive ${signal}`;
    case "negative":
      return `No positive ${signal} on average`;
    case "zero":
      return `${reading.label} of zero on average`;
  }
}

function indexSentences(reading: IndexReading, lead: string | null): string[] {
  const signal = SIGNAL[reading.key];
  const across =
    reading.validPixels !== null
      ? ` across ${pixelCount(reading.validPixels)} valid pixels`
      : "";
  const value = `${signedIndex(reading.mean)}${across}`;
  const sentences = [
    lead
      ? `${lead} had an average ${reading.label} of ${value}.`
      : `The average ${reading.label} was ${value}.`,
  ];

  switch (shownSign(signedIndex(reading.mean))) {
    case "positive":
      sentences.push(`An average above zero indicates a positive ${signal} in the measured pixels.`);
      break;
    case "negative":
      sentences.push(
        `An average below zero means the measured pixels did not show a positive ${signal} on average.`,
      );
      break;
    case "zero":
      sentences.push(
        `An average of zero at the displayed precision shows neither a positive nor a negative ${signal}.`,
      );
      break;
  }

  if (reading.min !== null && reading.max !== null) {
    sentences.push(
      `Individual pixel values ranged from ${signedIndex(reading.min)} to ${signedIndex(reading.max)}.`,
    );
  }
  if (reading.quality !== null && reading.quality.masked > 0) {
    sentences.push(
      `${pixelCount(reading.quality.masked)} of ${pixelCount(reading.quality.total)} pixels were excluded by pixel-quality masking before the average was computed.`,
    );
  }
  return sentences;
}

function interpretIndices(
  readings: IndexReading[],
  scene: SceneRef | null,
  context: ResultContext,
): Interpretation {
  const lead = `${when("Sentinel-2", scene, context.periods[0] ?? null)}, ${area(context)}`;
  const sentences = readings.flatMap((reading, position) =>
    indexSentences(reading, position === 0 ? lead : null),
  );
  const caveats = [...new Set(readings.map((reading) => INDEX_CAVEAT[reading.key]))];
  return {
    headline: readings.map(indexHeadline).join(" · "),
    explanation: sentences.join(" "),
    caveat: caveats.join(" "),
  };
}

// --------------------------------------------------------------------------- //
// Sentinel-1 backscatter
// --------------------------------------------------------------------------- //

function interpretSar(
  reading: SarReading,
  scene: SceneRef | null,
  context: ResultContext,
): Interpretation {
  const parts = [
    reading.vv !== null ? `VV backscatter of ${decibels(reading.vv)}` : null,
    reading.vh !== null ? `VH backscatter of ${decibels(reading.vh)}` : null,
  ].filter((part): part is string => part !== null);
  const across =
    reading.validPixels !== null
      ? `, across ${pixelCount(reading.validPixels)} valid pixels`
      : "";
  const sentences = [
    `${when("Sentinel-1", scene, context.periods[0] ?? null)}, ${area(context)} recorded a mean ${parts.join(" and a mean ")}${across}.`,
  ];

  if (reading.difference !== null) {
    const shown = decibels(reading.difference);
    const stronger = {
      positive: "VV backscatter was stronger than VH on average.",
      negative: "VH backscatter was stronger than VV on average.",
      zero: "The two polarizations were equal on average at the displayed precision.",
    }[shownSign(shown)];
    sentences.push(`The VV–VH difference was ${shown}: ${stronger}`);
  }

  const headline = [
    reading.vv !== null ? `VV ${decibels(reading.vv)}` : null,
    reading.vh !== null ? `VH ${decibels(reading.vh)}` : null,
  ]
    .filter((part): part is string => part !== null)
    .join(", ");
  return {
    headline: `Radar backscatter: ${headline}`,
    explanation: sentences.join(" "),
    caveat: SAR_CAVEAT,
  };
}

// --------------------------------------------------------------------------- //
// Two-period NDWI: what changed
// --------------------------------------------------------------------------- //

function interpretTemporal(reading: TemporalReading): Interpretation {
  const { earlier, later, difference, pairedChange, pairedPixels } = reading;
  const from = earlier.period ?? (earlier.acquired ? `the scene of ${earlier.acquired}` : "the earlier observation");
  const to = later.period ?? (later.acquired ? `the scene of ${later.acquired}` : "the later observation");

  if (difference === null) {
    return {
      headline: "Change not reported",
      explanation: `Both periods were analysed, but the analysis withheld the difference between ${from} and ${to}, so no change is stated.`,
      caveat: "The evidence records why the difference was withheld.",
    };
  }

  const shown = signedIndex(difference);
  const means =
    earlier.mean !== null && later.mean !== null
      ? ` from ${signedIndex(earlier.mean)} to ${signedIndex(later.mean)}`
      : "";
  const earlierScene = earlier.acquired ? ` (${earlier.acquired})` : "";
  const laterScene = later.acquired ? ` (${later.acquired})` : "";
  const paired =
    pairedChange !== null
      ? `Over the ${pairedPixels !== null ? `${pixelCount(pairedPixels)} ` : ""}pixels usable on both dates, the average per-pixel change was ${signedIndex(pairedChange)}.`
      : null;

  const sign = shownSign(shown);
  if (sign === "zero") {
    return {
      headline: `Little measured change in the water-related signal from ${from} to ${to}`,
      explanation: [
        `Between ${from} and ${to}, the measured water-related signal showed little measured change${means ? `,${means}` : ""}: a difference of ${shown} at the displayed precision.`,
        paired,
      ]
        .filter((sentence): sentence is string => sentence !== null)
        .join(" "),
      caveat: TEMPORAL_FLAT_CAVEAT,
    };
  }

  const [verb, relative] = sign === "positive" ? ["increased", "stronger"] : ["decreased", "weaker"];
  return {
    headline: `The water-related signal ${verb} from ${from} to ${to}`,
    explanation: [
      `Between ${from} and ${to}, the measured water-related signal ${verb}${means}, a change of ${shown}.`,
      `Based on the measured NDWI values, the later observation${laterScene} shows a ${relative} water-related signal than the earlier observation${earlierScene}.`,
      paired,
    ]
      .filter((sentence): sentence is string => sentence !== null)
      .join(" "),
    caveat: TEMPORAL_CAVEAT,
  };
}

// --------------------------------------------------------------------------- //
// How few pixels a value rests on
// --------------------------------------------------------------------------- //

/** The warning for one measurement, from its returned count - or none. */
function smallSample(count: number | null, subject: string): string | null {
  // No count, no claim: a missing count is never guessed at.
  if (count === null || count < 1 || count >= SMALL_SAMPLE_PIXELS) return null;
  const pixels = `${pixelCount(count)} valid ${count === 1 ? "pixel" : "pixels"}`;
  return `Small sample: only ${pixels} contributed to ${subject}, so interpret it cautiously.`;
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
            body.readings.length === 1 ? "this result" : `the ${reading.label} result`,
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
      const side = (reading: typeof earlier, role: string) =>
        smallSample(
          reading.validPixels,
          `the ${role} observation${reading.period ? ` (${reading.period})` : ""}`,
        );
      const paired =
        pairedChange !== null &&
        pairedPixels !== null &&
        pairedPixels >= 1 &&
        pairedPixels < SMALL_SAMPLE_PIXELS
          ? `Small sample: only ${pixelCount(pairedPixels)} ${pairedPixels === 1 ? "pixel was" : "pixels were"} usable on both dates for the paired-pixel change, so interpret it cautiously.`
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
