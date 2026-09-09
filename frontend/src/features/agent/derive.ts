/**
 * Read-only derivations over an agent result.
 *
 * Every function here selects from what the backend returned. Nothing is
 * computed, defaulted or inferred: a value the response does not carry is
 * absent, never filled in.
 */

import type {
  AgentEvidence,
  AgentResult,
  AgentToolStep,
  EvidenceItem,
  ExecuteQueryParams,
  ExecutedWindow,
  Measurement,
  QueryExecutionResult,
  SatelliteScene,
  SpectralIndexKey,
} from "../../api/types";
import type { RunContext } from "../query/ConfigSummary";

/** The validated plan's discovery step, when the plan carried one. */
export function executeStep(
  result: { trace: { plan: { steps: { tool: string }[] } | null } } | null,
): ExecuteQueryParams | undefined {
  return result?.trace.plan?.steps.find(
    (candidate): candidate is ExecuteQueryParams =>
      candidate.tool === "execute_query",
  );
}

/**
 * Whether this run ASKED the server for a display image.
 *
 * Read from `include_imagery` on the validated plan - the only place that
 * answers it. `undefined` when no plan exists, because a run that never
 * planned neither requested imagery nor declined to: it got no further.
 *
 * The distinction is the whole point. A plan that computes an index reads the
 * raster bands directly and does not need a PNG, so `include_imagery` is
 * routinely `false` on a completely successful run. Treating a missing picture
 * as a failed retrieval reports that success as a fault.
 */
export function imageryRequested(
  result: { trace: { plan: { steps: { tool: string }[] } | null } } | null,
): boolean | undefined {
  const step = executeStep(result);
  return step === undefined ? undefined : step.include_imagery;
}

/**
 * What became of the visual (vision-language) step.
 *
 * Four outcomes, and they are NOT interchangeable. The panel used to render
 * one sentence for all of them - "No visual interpretation was produced" -
 * which reads as "the model looked and saw nothing". In the common case the
 * model never looked: the planner did not select the tool, because the
 * question asked for a number.
 *
 * - `no-plan`        nothing was planned, so nothing was requested
 * - `not-requested`  a plan was made and did not include the visual tool
 * - `failed`         the tool was selected but its step did not succeed
 * - `produced`       an observation exists (rendered from the evidence)
 */
export type VisualStepState =
  | { kind: "no-plan" }
  | { kind: "not-requested" }
  | { kind: "failed"; step: AgentToolStep }
  | { kind: "produced" };

export function visualStepState(result: AgentResult | null): VisualStepState {
  if (result === null || result.trace.plan === null) return { kind: "no-plan" };

  const planned = result.trace.plan.steps.some(
    (step) => step.tool === "rs_model_analysis",
  );
  if (!planned) return { kind: "not-requested" };

  const produced = result.evidence.items.some(
    (item) => item.source === "model" && item.visual !== null,
  );
  if (produced) return { kind: "produced" };

  const step = result.trace.steps.find(
    (candidate) => candidate.parameters.tool === "rs_model_analysis",
  );
  // Selected, and no observation came back. Reported as an incomplete step
  // rather than as an empty reading: the model was asked and did not answer.
  return step === undefined ? { kind: "not-requested" } : { kind: "failed", step };
}

/**
 * Which spectral indices this run actually computed.
 *
 * Read from the plan's own steps. An index absent from this set was NOT
 * computed for this query - which is a fact about the question asked, not
 * about whether the system can compute it. The rail must say those two things
 * differently.
 */
export function indicesComputed(
  result: { trace: { plan: { steps: { tool: string }[] } | null } } | null,
): SpectralIndexKey[] {
  const steps = result?.trace.plan?.steps ?? [];
  const used = new Set<SpectralIndexKey>();
  for (const step of steps) {
    if (step.tool === "ndwi_statistics" || step.tool === "temporal_ndwi_statistics") {
      used.add("ndwi");
    }
    if (step.tool === "spectral_indices") {
      for (const key of (step as { indices?: SpectralIndexKey[] }).indices ?? []) {
        used.add(key);
      }
    }
  }
  return ["ndvi", "ndwi", "ndbi"].filter((key): key is SpectralIndexKey =>
    used.has(key as SpectralIndexKey),
  );
}

/**
 * The window whose scene the workspace is showing.
 *
 * Imagery wins, because that is the raster on the map; otherwise the first
 * window that selected a scene at all. Deterministic, and read from the
 * execution result rather than assumed to be the first window.
 */
export function shownWindow(
  execution: QueryExecutionResult | null,
): ExecutedWindow | null {
  if (execution === null) return null;
  return (
    execution.windows.find((window) => window.imagery !== null) ??
    execution.windows.find((window) => window.selected_scene_id !== null) ??
    null
  );
}

/**
 * The STAC record for the scene that was actually selected.
 *
 * Matched by `selected_scene_id` - never `scenes[0]`. Discovery returns every
 * candidate in the window and the deterministic selection often lands on a
 * later one, so taking the first would attribute another scene's acquisition
 * time and cloud cover to the imagery on screen.
 */
export function shownScene(window: ExecutedWindow | null): SatelliteScene | null {
  if (window === null || window.selected_scene_id === null) return null;
  return (
    window.scenes.find((scene) => scene.id === window.selected_scene_id) ?? null
  );
}

/** Measurements the backend computed, in the order it returned them. */
export function measurementsFrom(evidence: AgentEvidence): Measurement[] {
  return evidence.items
    .map((item) => item.measurement)
    .filter((measurement): measurement is Measurement => measurement !== null);
}

/** The first measurement carrying a given unit, or `undefined`. */
export function measurementByUnit(
  evidence: AgentEvidence | null,
  unit: string,
): Measurement | undefined {
  if (evidence === null) return undefined;
  return measurementsFrom(evidence).find(
    (measurement) => measurement.unit === unit,
  );
}

/** One evidence item as a single readable line: a measurement, or its text. */
export function evidenceValue(item: EvidenceItem): string {
  if (item.measurement) {
    return `${item.measurement.name} = ${item.measurement.value} ${item.measurement.unit}`;
  }
  return item.text ?? "";
}

/**
 * The configuration an agent run actually used, for the workspace rail.
 *
 * Read entirely from the validated plan and the execution result - the intent
 * the server accepted, the extent it resolved, the scenes discovery returned
 * and the one selection chose. Returns `null` before any of that exists rather
 * than a shape full of blanks.
 */
export function runContextFrom(
  result: {
    trace: { plan: { steps: { tool: string }[] } | null };
    evidence: AgentEvidence;
  } | null,
): RunContext | null {
  if (result === null) return null;
  const step = executeStep(result);
  if (step === undefined) return null;

  const execution = result.evidence.execution;
  const bbox = execution?.plan.bbox ?? null;
  const window = shownWindow(execution);
  const indices = indicesComputed(result);

  const windows = step.intent.time_windows;
  const ranges = Array.isArray(windows)
    ? windows
    : [windows.baseline, windows.target];
  const label = ranges
    .map((range) =>
      range.start_date === range.end_date
        ? range.start_date
        : `${range.start_date} → ${range.end_date}`,
    )
    .join("  vs  ");

  return {
    location: step.intent.location_query,
    centre:
      bbox === null
        ? null
        : {
            lat: (bbox.south + bbox.north) / 2,
            lon: (bbox.west + bbox.east) / 2,
          },
    window: label,
    cloudRule:
      step.max_cloud_cover === null
        ? null
        : `Cloud cover ≤ ${step.max_cloud_cover}%`,
    modalities: step.intent.modalities,
    task: step.intent.task,
    ndwi: indices.includes("ndwi"),
    indices,
    scenes: window?.scenes ?? [],
    selectedSceneId: window?.selected_scene_id ?? null,
  };
}
