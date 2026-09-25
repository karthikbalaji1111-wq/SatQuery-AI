/**
 * What a finished run MEANS, for the workspace to present.
 *
 * Pure selection over the response. Every number here is a value the backend
 * returned, found by the evidence id it was published under; every label names
 * that value, never a stronger claim. Nothing is computed from pixels,
 * interpolated, defaulted or rounded (rounding is the presentation layer's
 * job). A value the response does not carry is `null`, and the view omits it.
 *
 * Two questions are answered separately, because they fail separately:
 *
 *   outcomeOf()      - WHICH kind of result this is. Success, a question back,
 *                      a refusal, an outage and "the evidence did not answer
 *                      it" are different events and are never merged.
 *   resultSummary()  - WHAT was measured, where and when, read from the
 *                      operation's own fields (index, backscatter, or a
 *                      comparison of two periods).
 */

import type {
  AgentResult,
  AnalysisResult,
  Measurement,
  ClarificationReason,
  GridState,
  RadiometricState,
  EvidenceItem,
  ExecutedWindow,
  PixelQuality,
  SatQueryIntent,
  TimeRange,
} from "../../api/types";
import { executeStep, shownScene, shownWindow } from "./derive";

// --------------------------------------------------------------------------- //
// Outcome
// --------------------------------------------------------------------------- //

export type OutcomeKind =
  | "success"
  | "clarification"
  | "location_not_found"
  | "location_unavailable"
  | "location_is_point"
  | "area_too_large"
  | "unsupported"
  | "insufficient_evidence"
  | "analysis_refused"
  | "provider_failure";

export interface Outcome {
  kind: OutcomeKind;
  /** A short state name, for the chip above the result. */
  label: string;
  /**
   * For a validated refusal: the server's own reason, verbatim. Never a
   * second, friendlier explanation invented here.
   */
  reason: string | null;
}

const CLARIFICATION_KIND: Partial<Record<ClarificationReason, OutcomeKind>> = {
  location_not_found: "location_not_found",
  location_is_point: "location_is_point",
  area_too_large: "area_too_large",
  analysis_unsupported: "unsupported",
  requires_ai_model: "unsupported",
};

const OUTCOME_LABELS: Record<OutcomeKind, string> = {
  success: "Result",
  clarification: "One more detail",
  location_not_found: "Location not found",
  location_unavailable: "Location service unavailable",
  location_is_point: "A point, not an area",
  area_too_large: "Area too large",
  unsupported: "Not supported",
  insufficient_evidence: "No measurement",
  analysis_refused: "Analysis not computed",
  provider_failure: "Incomplete",
};

/** A measured value, as opposed to a count of scenes or pixels. */
function isAnalytical(item: EvidenceItem): boolean {
  const unit = item.measurement?.unit;
  return unit === "index" || unit === "dB";
}

export function outcomeOf(result: AgentResult): Outcome {
  const make = (kind: OutcomeKind, reason: string | null = null): Outcome => ({
    kind,
    label: OUTCOME_LABELS[kind],
    reason,
  });

  switch (result.status) {
    case "needs_clarification": {
      const reason = result.clarification?.reason;
      return make((reason && CLARIFICATION_KIND[reason]) ?? "clarification");
    }
    case "location_unavailable":
      return make("location_unavailable");
    case "planner_unavailable":
    case "synthesis_unavailable":
    case "answer_withheld":
      return make("provider_failure");
    case "ok":
      break;
  }

  if (result.evidence.items.some(isAnalytical)) return make("success");

  // Nothing was measured. A validation stage that REFUSED an analysis is a
  // different event from a search that found nothing to measure.
  const refused = (result.evidence.analysis?.analysis_outcomes ?? []).find(
    (outcome) => outcome.status === "unavailable",
  );
  if (refused) return make("analysis_refused", refused.reason ?? null);

  // Imagery alone was asked for and delivered: that IS the result.
  const step = executeStep(result);
  const onlyDiscovery =
    result.trace.plan !== null &&
    result.trace.plan.steps.every((candidate) => candidate.tool === "execute_query");
  if (step?.include_imagery && onlyDiscovery && shownWindow(result.evidence.execution)?.imagery) {
    return make("success");
  }
  return make("insufficient_evidence");
}

// --------------------------------------------------------------------------- //
// Context: where and when
// --------------------------------------------------------------------------- //

const MONTHS = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
];

function lastDayOfMonth(year: number, month: number): number {
  return new Date(Date.UTC(year, month, 0)).getUTCDate();
}

/**
 * A requested range in the words it was most likely asked in: a whole month
 * as "December 2024", a whole year as "2024", anything else verbatim. Pure
 * formatting of the range the server planned - no date is moved.
 */
export function formatPeriod(range: TimeRange): string {
  const start = /^(\d{4})-(\d{2})-(\d{2})$/.exec(range.start_date);
  const end = /^(\d{4})-(\d{2})-(\d{2})$/.exec(range.end_date);
  if (start && end) {
    const [sy, sm, sd] = start.slice(1).map(Number);
    const [ey, em, ed] = end.slice(1).map(Number);
    if (sy === ey && sm === em && sd === 1 && ed === lastDayOfMonth(ey, em)) {
      return `${MONTHS[sm - 1]} ${sy}`;
    }
    if (sy === ey && sm === 1 && sd === 1 && em === 12 && ed === 31) {
      return String(sy);
    }
  }
  if (range.start_date === range.end_date) return range.start_date;
  return `${range.start_date} → ${range.end_date}`;
}

/** The date part of an ISO instant, sliced - never re-zoned through `Date`. */
export function acquisitionDate(iso: string | null | undefined): string | null {
  if (!iso) return null;
  return /^\d{4}-\d{2}-\d{2}/.test(iso) ? iso.slice(0, 10) : iso;
}

export interface GeocoderMatch {
  /** The geocoder's own name for what it matched, first parts only. */
  name: string;
  /** The full name, verbatim, for the evidence and a tooltip. */
  full: string;
  /** Its own classification, e.g. "railway · stop" - recorded, not judged. */
  kind: string | null;
}

export interface ResultContext {
  /** The place as the question named it. */
  location: string | null;
  /** What the geocoder actually matched for it, when the server said. */
  matched: GeocoderMatch | null;
  /** The requested period(s), as planned. */
  periods: string[];
  /** Scenes the catalog returned for the (first) searched window. */
  sceneCount: number | null;
}

export function matchFrom(
  name: string | null | undefined,
  placeClass: string | null | undefined,
  placeType: string | null | undefined,
): GeocoderMatch | null {
  if (!name) return null;
  const parts = name.split(",").map((part) => part.trim()).filter(Boolean);
  const kind = [placeClass, placeType]
    .filter((part): part is string => Boolean(part))
    .map((part) => part.replace(/_/g, " "))
    .join(" · ");
  return { name: parts.slice(0, 2).join(", ") || name, full: name, kind: kind || null };
}

function intentOf(result: AgentResult): SatQueryIntent | null {
  return (
    result.evidence.execution?.plan.intent ?? executeStep(result)?.intent ?? null
  );
}

export function resultContext(result: AgentResult): ResultContext {
  const intent = intentOf(result);
  const windows = intent?.time_windows;
  const periods = windows
    ? Array.isArray(windows)
      ? windows.map(formatPeriod)
      : [formatPeriod(windows.baseline), formatPeriod(windows.target)]
    : [];
  // The window on screen, or - when no window selected a scene - the first one
  // searched, so "0 matched" is still stated rather than lost.
  const window =
    shownWindow(result.evidence.execution) ??
    result.evidence.execution?.windows[0] ??
    null;
  const plan = result.evidence.execution?.plan ?? null;
  return {
    location: intent?.location_query ?? null,
    matched: matchFrom(plan?.matched_name, plan?.matched_class, plan?.matched_type),
    periods,
    sceneCount: window ? window.scene_count : null,
  };
}

// --------------------------------------------------------------------------- //
// What was measured
// --------------------------------------------------------------------------- //

export interface SceneRef {
  id: string;
  acquired: string | null;
  platform: string | null;
}

export interface Quality {
  valid: number;
  total: number;
  /** As the backend computed it; `null` when the grid had no pixels. */
  validFraction: number | null;
  masked: number;
  source: string;
}

export interface IndexReading {
  key: "ndvi" | "ndwi" | "ndbi";
  label: string;
  title: string;
  mean: number;
  min: number | null;
  max: number | null;
  validPixels: number | null;
  quality: Quality | null;
}

export interface SarReading {
  vv: number | null;
  vh: number | null;
  difference: number | null;
  validPixels: number | null;
  polarizations: string[];
}

export interface PeriodReading {
  role: "Earlier" | "Later";
  /** The requested period this observation came from, when known. */
  period: string | null;
  acquired: string | null;
  sceneId: string | null;
  mean: number | null;
  validPixels: number | null;
  quality: Quality | null;
}

export interface TemporalReading {
  earlier: PeriodReading;
  later: PeriodReading;
  /** later mean − earlier mean, as the backend computed it; `null` if suppressed. */
  difference: number | null;
  /** Paired-pixel change, only when the grids were verified identical. */
  pairedChange: number | null;
  pairedPixels: number | null;
}

export type ResultBody =
  | { kind: "index"; readings: IndexReading[]; scene: SceneRef | null }
  | { kind: "sar"; reading: SarReading; scene: SceneRef | null }
  | { kind: "temporal"; reading: TemporalReading }
  | { kind: "imagery"; scene: SceneRef | null }
  | { kind: "none" };

const INDEX_NAMES: { key: IndexReading["key"]; label: string; title: string }[] = [
  { key: "ndvi", label: "NDVI", title: "Vegetation index" },
  { key: "ndwi", label: "NDWI", title: "Water index" },
  { key: "ndbi", label: "NDBI", title: "Built-up index" },
];

function valueOf(items: EvidenceItem[], id: string): number | null {
  const value = items.find((item) => item.id === id)?.measurement?.value;
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function qualityFrom(quality: PixelQuality | null | undefined): Quality | null {
  if (!quality) return null;
  return {
    valid: quality.valid_pixels,
    total: quality.total_pixels,
    validFraction: quality.valid_fraction,
    masked: quality.masked_pixels,
    source: quality.mask_source,
  };
}

function sceneFrom(window: ExecutedWindow | null): SceneRef | null {
  const scene = shownScene(window);
  const id = window?.selected_scene_id ?? scene?.id ?? null;
  if (id === null) return null;
  return {
    id,
    acquired: acquisitionDate(scene?.datetime ?? null),
    platform: scene?.platform ?? null,
  };
}

function periodFor(result: AgentResult, label: string | null | undefined): string | null {
  if (!label) return null;
  const window = result.evidence.execution?.windows.find((w) => w.label === label);
  return window ? formatPeriod(window.time_range) : null;
}

function temporalFrom(result: AgentResult): TemporalReading | null {
  const items = result.evidence.items;
  if (!items.some((item) => item.id.startsWith("temporal_ndwi."))) return null;
  const comparison = result.evidence.analysis?.temporal_comparison ?? null;

  const side = (key: "first" | "second"): PeriodReading => {
    const observation = comparison?.[key] ?? null;
    return {
      role: key === "first" ? "Earlier" : "Later",
      period: periodFor(result, observation?.window_label),
      acquired: acquisitionDate(observation?.acquired_at ?? null),
      sceneId: observation?.scene_id ?? null,
      mean: valueOf(items, `temporal_ndwi.${key}.ndwi_mean`),
      validPixels: valueOf(items, `temporal_ndwi.${key}.ndwi_valid_pixel_count`),
      quality: qualityFrom(observation?.pixel_quality),
    };
  };

  return {
    earlier: side("first"),
    later: side("second"),
    difference: valueOf(items, "temporal_ndwi.difference.mean_ndwi_difference"),
    pairedChange: valueOf(items, "temporal_ndwi.change.ndwi_change_mean"),
    pairedPixels: valueOf(items, "temporal_ndwi.change.paired_valid_pixel_count"),
  };
}

function indexReadings(result: AgentResult): IndexReading[] {
  const items = result.evidence.items;
  const quality = result.evidence.analysis?.pixel_quality ?? [];
  const readings: IndexReading[] = [];
  for (const { key, label, title } of INDEX_NAMES) {
    const mean = valueOf(items, `${key}.${key}_mean`);
    if (mean === null) continue;
    readings.push({
      key,
      label,
      title,
      mean,
      min: valueOf(items, `${key}.${key}_min`),
      max: valueOf(items, `${key}.${key}_max`),
      validPixels: valueOf(items, `${key}.${key}_valid_pixel_count`),
      quality: qualityFrom(quality.find((entry) => entry.index === key)),
    });
  }
  return readings;
}

function sarReading(result: AgentResult): SarReading | null {
  const items = result.evidence.items;
  const vv = valueOf(items, "sar_backscatter.vv_mean_db");
  const vh = valueOf(items, "sar_backscatter.vh_mean_db");
  if (vv === null && vh === null) return null;
  return {
    vv,
    vh,
    difference: valueOf(items, "sar_backscatter.vv_minus_vh_mean_db"),
    validPixels:
      valueOf(items, "sar_backscatter.vv_valid_pixel_count") ??
      valueOf(items, "sar_backscatter.vh_valid_pixel_count"),
    polarizations: (result.evidence.analysis?.sar_backscatter?.polarizations ?? []).map(
      (entry) => String(entry.polarization).toUpperCase(),
    ),
  };
}

export function resultSummary(result: AgentResult): ResultBody {
  const temporal = temporalFrom(result);
  if (temporal) return { kind: "temporal", reading: temporal };

  const scene = sceneFrom(shownWindow(result.evidence.execution));
  const readings = indexReadings(result);
  if (readings.length > 0) return { kind: "index", readings, scene };

  const sar = sarReading(result);
  if (sar) {
    const backscatter = result.evidence.analysis?.sar_backscatter;
    return {
      kind: "sar",
      reading: sar,
      scene: backscatter
        ? {
            id: backscatter.scene_id,
            acquired: acquisitionDate(backscatter.acquired_at),
            platform: scene?.platform ?? null,
          }
        : scene,
    };
  }

  if (outcomeOf(result).kind === "success") return { kind: "imagery", scene };
  return { kind: "none" };
}

// --------------------------------------------------------------------------- //
// Validation records, wherever the analysis published them
// --------------------------------------------------------------------------- //

export interface ValidationRecords {
  quality: PixelQuality[];
  radiometry: RadiometricState[];
  grids: GridState[];
}

/**
 * Every validation record the analysis carries. A single-scene run publishes
 * them at the top level; a comparison publishes them PER OBSERVATION, plus the
 * pair's own grid check - so looking only at the top level reported a
 * comparison as if its imagery had not been validated.
 */
export function validationRecords(analysis: AnalysisResult | null | undefined): ValidationRecords {
  if (!analysis) return { quality: [], radiometry: [], grids: [] };
  const comparison = analysis.temporal_comparison ?? null;
  const observations = comparison ? [comparison.first, comparison.second] : [];
  const present = <T>(value: T | null | undefined): value is T => value != null;
  return {
    quality: [
      ...(analysis.pixel_quality ?? []),
      ...observations.map((observation) => observation.pixel_quality).filter(present),
    ],
    radiometry: [
      ...(analysis.radiometry ?? []),
      ...observations.map((observation) => observation.radiometry).filter(present),
    ],
    grids: [
      ...(analysis.grids ?? []),
      ...observations.map((observation) => observation.grid).filter(present),
      ...[comparison?.pair_grid].filter(present),
    ],
  };
}

// --------------------------------------------------------------------------- //
// The run, stage by stage - after the fact
// --------------------------------------------------------------------------- //

export type StageState = "done" | "failed" | "attention";

export interface RunStage {
  name: string;
  state: StageState;
  /** What the response says happened at this stage, in its own values. */
  detail: string | null;
}

/**
 * The stages the run ACTUALLY went through, reconstructed from the response.
 *
 * The API is one request with no progress events, so nothing here is shown
 * while it runs; afterwards each stage appears only if the response carries
 * evidence of it, and stops at the first one that did not complete.
 */
export function runStages(result: AgentResult): RunStage[] {
  const stages: RunStage[] = [];
  const outcome = outcomeOf(result);
  const context = resultContext(result);

  // 1. Understanding the question.
  if (result.trace.plan === null) {
    if (result.status === "needs_clarification") {
      const kind = outcome.kind;
      stages.push({
        name: "Understand question",
        state: "attention",
        detail:
          kind === "unsupported"
            ? "request not supported"
            : kind === "clarification"
              ? "needs one more detail"
              : null,
      });
      if (kind === "clarification" || kind === "unsupported") return stages;
    } else {
      stages.push({ name: "Understand question", state: "failed", detail: "no plan" });
      return stages;
    }
  } else {
    stages.push({ name: "Understand question", state: "done", detail: null });
  }

  // 2. Resolving the place.
  const execution = result.evidence.execution;
  if (outcome.kind === "location_unavailable") {
    stages.push({ name: "Resolve location", state: "failed", detail: "location service unavailable" });
    return stages;
  }
  if (outcome.kind === "location_not_found") {
    stages.push({ name: "Resolve location", state: "failed", detail: "not found" });
    return stages;
  }
  if (outcome.kind === "location_is_point") {
    stages.push({ name: "Resolve location", state: "attention", detail: "a single point, not an area" });
    return stages;
  }
  if (outcome.kind === "area_too_large") {
    stages.push({ name: "Resolve location", state: "attention", detail: "area too large to measure" });
    return stages;
  }
  if (execution === null) {
    stages.push({ name: "Resolve location", state: "failed", detail: null });
    return stages;
  }
  stages.push({ name: "Resolve location", state: "done", detail: context.location });

  // 3. Finding scenes.
  const windows = execution.windows;
  const found = windows.reduce((total, window) => total + window.scene_count, 0);
  const selected = windows.filter((window) => window.selected_scene_id).length;
  stages.push({
    name: "Find satellite scenes",
    state: selected > 0 ? "done" : "attention",
    detail:
      selected > 0
        ? `${found} found · ${selected} selected`
        : "no scene matched",
  });
  if (selected === 0) return stages;

  // 4. Validating the imagery (M4 radiometry, M5 geometry) - when it ran.
  const analysis = result.evidence.analysis;
  const { radiometry, grids } = validationRecords(analysis);
  if (radiometry.length > 0 || grids.length > 0) {
    const refused =
      radiometry.some(
        (state) => state.status === "incompatible" || state.status === "undetermined",
      ) || grids.some((grid) => grid.status === "refused");
    stages.push({
      name: "Validate imagery",
      state: refused ? "attention" : "done",
      detail: refused ? "a check refused the data" : "radiometry & geometry checked",
    });
  }

  // 5. Measuring.
  const outcomes = analysis?.analysis_outcomes ?? [];
  if (outcomes.length > 0) {
    const failed = outcomes.filter((entry) => entry.status === "unavailable");
    stages.push({
      name: "Run analysis",
      state: failed.length === 0 ? "done" : failed.length === outcomes.length ? "failed" : "attention",
      detail: outcomes
        .map((entry) => `${entry.name.replace(/_/g, " ").toUpperCase()} ${entry.status === "completed" ? "✓" : "✕"}`)
        .join(" · "),
    });
  }

  // 6. The answer, checked against the evidence.
  const validation = result.trace.answer_validation;
  if (validation !== null) {
    const checks = [
      validation.numeric_grounding,
      validation.evidence_refs,
      validation.forbidden_terms,
    ].filter((check) => check !== "not_run");
    stages.push({
      name: "Check answer",
      state: checks.every((check) => check === "pass") ? "done" : "failed",
      detail: checks.every((check) => check === "pass") ? "grounded in the evidence" : "withheld",
    });
  }
  return stages;
}

// --------------------------------------------------------------------------- //
// Threshold shares
// --------------------------------------------------------------------------- //

/**
 * A threshold share, labelled as what it is: a count of pixels whose INDEX
 * exceeds a stated value - never "water" or any class. The value is read from
 * the measurement's own name (`ndwi_percent_above_index_threshold_0.3`); a bare
 * "above threshold" left the reader to guess both the threshold and what it
 * was a threshold on.
 */
export function thresholdNote(measurement: Measurement): string {
  const share = `${measurement.value.toFixed(1)}%`;
  const parsed = /^([a-z]+)_percent_above_index_threshold_(-?\d+(?:\.\d+)?)$/i.exec(
    measurement.name,
  );
  return parsed
    ? `${share} of valid pixels with ${parsed[1].toUpperCase()} > ${parsed[2]}`
    : `${share} of valid pixels above the index threshold`;
}
