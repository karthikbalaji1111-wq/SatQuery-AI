import { Fragment, useEffect, useState, type ReactNode } from "react";

import { SarBackscatterPanel } from "./SarBackscatterPanel";
import type {
  AgentClarification,
  AgentResult,
  PixelQuality,
  RadiometricState,
  TemporalIndexComparison,
  AgentStatus,
  AgentToolName,
  AgentToolStep,
  AgentEvidence as Evidence,
  Measurement,
  ImageryResponse,
  Modality,
  QueryTask,
  SatelliteScene,
  SarBackscatterResult,
  SatQueryIntent,
  SpectralIndexKey,
  TimeRange,
} from "../../api/types";
import {
  evidenceValue,
  executeStep,
  measurementsFrom,
  shownScene,
  shownWindow,
  visualStepState,
} from "./derive";
import { describeImageryError } from "../query/imageryError";
import {
  outcomeOf,
  resultContext,
  resultSummary,
  matchFrom,
  runStages,
  thresholdNote,
  validationRecords,
  type OutcomeKind,
  type Quality,
  type ResultBody,
  type RunStage,
} from "./resultModel";
import { decibels, pixelCount as pixels, signedIndex } from "./format";
import {
  formatDay,
  interpretResult,
  technicalDetails,
  type Interpretation,
  type TechnicalRow,
} from "./interpretation";
import { STANDARD_INTERPRETER, useAgentRun } from "./agentRun";
import type { AgentRun, AgentRunHandlers } from "./agentRun";

/**
 * Starting points, one per capability the agent actually has. Clicking one
 * RUNS it - through the same request as a typed question; nothing is canned -
 * and leaves it in the box, where it can be edited and run again.
 */
const EXAMPLE_QUESTIONS = [
  {
    label: "Water index",
    question: "What is the NDWI of Marina Beach, Chennai in January 2025?",
  },
  // Whole cities (Bengaluru, Hyderabad) are larger than one native-resolution
  // read and are refused by the analysis gate; these are real places inside
  // them, each verified to return a measurement.
  {
    label: "Vegetation",
    question: "Show vegetation around Cubbon Park, Bengaluru in December 2024.",
  },
  {
    label: "Built-up area",
    question: "Show built-up area around Ameerpet, Hyderabad in January 2025.",
  },
  {
    label: "Radar backscatter",
    question:
      "Analyse SAR backscatter around Marina Beach, Chennai in January 2025.",
  },
  {
    label: "Temporal comparison",
    question:
      "Compare water at Marina Beach, Chennai between January 2024 and January 2025.",
  },
];

/**
 * Readable labels for closed enums the API already defines. These rename
 * nothing: each is a one-to-one display form of a value the backend returned,
 * and a value outside the set falls through to itself rather than being guessed.
 */
const MODALITY_LABELS: Record<Modality, string> = {
  "sentinel-2-optical": "Sentinel-2 L2A",
  "sentinel-1-sar": "Sentinel-1 SAR",
};

const TASK_LABELS: Record<QueryTask, string> = {
  visualize: "visualize",
  change_detection: "change detection",
  object_identification: "object identification",
};

/** The pipeline strip names the server-side operation, not the tool label. */
const TOOL_STAGES: Record<AgentToolName, string> = {
  execute_query: "stac_search",
  spectral_indices: "index_compute",
  ndwi_statistics: "ndwi_compute",
  sar_backscatter_statistics: "sar_backscatter",
  temporal_ndwi_statistics: "ndwi_temporal",
  rs_model_analysis: "vlm_observe",
};

const TOOL_KINDS: Record<AgentToolName, string> = {
  execute_query: "geo",
  spectral_indices: "index",
  ndwi_statistics: "index",
  sar_backscatter_statistics: "index",
  temporal_ndwi_statistics: "index",
  rs_model_analysis: "model",
};

/**
 * What to say when there is no answer.
 *
 * These three statuses are NOT the same event and must never read as though
 * they were. The one that matters most is `synthesis_unavailable`: the
 * satellite work SUCCEEDED there. Scenes were discovered, a scene was selected,
 * its raster was read and the index was computed - and the only thing that
 * failed was the language model writing an English sentence about it. Wording
 * that implies the analysis fell short blames the measurements for a failure
 * that happened after them, in a component that computes nothing.
 *
 * So each notice states, separately: what failed, what still holds, and
 * whether retrying is the right response.
 *
 * `detail` is deliberately a separate field rather than a longer sentence: it
 * is the slot a more specific server-supplied reason drops into (a rate limit
 * with a retry-after, say) without the headline having to change.
 */
interface StatusNotice {
  /** What failed, in one sentence. */
  summary: string;
  /** What is still true, so the reader can calibrate. */
  detail: string;
  /** Whether the same question, run again, could succeed. */
  retryable: boolean;
}

const STATUS_NOTICES: Record<Exclude<AgentStatus, "ok">, StatusNotice> = {
  planner_unavailable: {
    summary:
      "No analysis ran. The language model that selects which analysis to run did not return a usable plan, so the pipeline was never started.",
    detail:
      "Nothing was measured and no evidence was collected - not because the question could not be answered, but because the run stopped before any satellite work began.",
    retryable: true,
  },
  synthesis_unavailable: {
    summary:
      "The written summary is missing: the language model did not return a usable answer.",
    detail:
      "Any scenes, measurements and evidence collected before this step remain available below. Check the pipeline for individual tool outcomes.",
    retryable: true,
  },
  answer_withheld: {
    summary:
      "An answer was generated but failed validation against the evidence, so it was withheld rather than shown.",
    detail:
      "The measurements are unaffected and remain valid - the withheld text is the model's prose, not the analysis. The failed checks are listed below.",
    retryable: true,
  },
  location_unavailable: {
    summary: "Location service temporarily unavailable.",
    detail:
      "The place could not be looked up, so no scene was searched and nothing was measured. The question itself is fine - this is the location service, not the satellite data.",
    retryable: true,
  },
  needs_clarification: {
    summary: "No analysis ran: the question needs one more detail.",
    detail:
      "Nothing was measured - SatQuery asks rather than guessing a place, a period or an analysis. The question is shown under the query box.",
    retryable: false,
  },
};

/**
 * How a provider names itself, for attribution.
 *
 * Display casing only - the value comes from the response, and an unrecognised
 * provider is shown as the server reported it rather than being relabelled.
 */
const PROVIDER_LABELS: Record<string, string> = {
  gemini: "Gemini",
  nvidia: "NVIDIA",
  anthropic: "Claude",
  local: "Local",
  mock: "Mock",
  [STANDARD_INTERPRETER]: "the standard workflow · local intent model, no external AI",
};

function providerLabel(provider: string): string {
  return PROVIDER_LABELS[provider] ?? provider;
}

/** Presentation-layer formatting. API values are never rounded; only this is. */
function formatMeasurement(value: number, unit: string): string {
  if (!Number.isFinite(value)) return String(value);
  switch (unit) {
    case "index":
      return signedIndex(value);
    case "%":
      return value.toFixed(1);
    case "pixels":
    case "count":
      return pixels(value);
    default:
      return String(value);
  }
}

/**
 * An ISO instant as the design writes it. Sliced, never parsed into a `Date`:
 * Earth Search returns Z-suffixed UTC and re-rendering it through the viewer's
 * local zone would silently move an acquisition time.
 */
function formatInstant(iso: string): string {
  return /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/.test(iso)
    ? `${iso.slice(0, 10)} ${iso.slice(11, 19)} UTC`
    : iso;
}

function formatRange(range: TimeRange): string {
  if (range.start_date === range.end_date) return range.start_date;
  const sameYear = range.start_date.slice(0, 4) === range.end_date.slice(0, 4);
  const end = sameYear ? range.end_date.slice(5) : range.end_date;
  return `${range.start_date} → ${end}`;
}

/** Both arms of the union are rendered; neither is assumed. */
function formatWindows(windows: SatQueryIntent["time_windows"]): string {
  if (Array.isArray(windows)) return windows.map(formatRange).join(", ");
  return `${formatRange(windows.baseline)} vs ${formatRange(windows.target)}`;
}

/**
 * The spectral indices the backend names its measurements after.
 *
 * The backend prefixes every measurement with the index that produced it
 * (`ndwi_mean`, `ndvi_valid_pixel_count`, `ndbi_min`), which is the ONLY
 * reliable way to tell them apart: they all share the unit `index`, so unit
 * alone cannot say which index a number belongs to.
 *
 * Each caveat names the classification this index is NOT. One shared sentence
 * would have to be either wrong or vague, and an NDVI mean captioned "not a
 * validated water classification" is a statement about the wrong index.
 */
const INDEX_FAMILIES: { key: SpectralIndexKey; label: string; caveat: string }[] =
  [
    {
      key: "ndvi",
      label: "NDVI",
      caveat:
        "Spectral index only — not a validated vegetation or land-cover classification.",
    },
    {
      key: "ndwi",
      label: "NDWI",
      caveat: "Spectral index only — not a validated water classification.",
    },
    {
      key: "ndbi",
      label: "NDBI",
      caveat:
        "Spectral index only — not a validated built-up or bare-ground classification.",
    },
  ];

/**
 * One index's own measurements, kept together.
 *
 * Grouping by name prefix rather than by unit is the whole point. Selecting
 * "the first measurement whose unit is `index`" and "the first whose unit is
 * `%`" picks from whatever the backend happened to return first: with three
 * indices computed that paired an NDVI mean with an NDWI threshold percentage
 * and printed them as one reading. Every field here comes from the same index
 * or is absent.
 */
interface IndexGroup {
  key: SpectralIndexKey;
  label: string;
  caveat: string;
  mean: Measurement;
  percent: Measurement | undefined;
  validPixels: Measurement | undefined;
}

function belongsTo(measurement: Measurement, key: SpectralIndexKey): boolean {
  return measurement.name.toLowerCase().startsWith(`${key}_`);
}

/**
 * Every index that reported a mean, in a fixed order.
 *
 * An index the run did not compute is absent rather than zeroed, and a
 * measurement whose name matches no known index is left for the caller: this
 * never invents a family to put a stray number in.
 */
function indexGroups(measurements: Measurement[]): IndexGroup[] {
  const groups: IndexGroup[] = [];
  for (const family of INDEX_FAMILIES) {
    const own = measurements.filter((measurement) =>
      belongsTo(measurement, family.key),
    );
    const mean =
      own.find(
        (measurement) =>
          measurement.unit === "index" &&
          measurement.name.toLowerCase().endsWith("_mean"),
      ) ?? own.find((measurement) => measurement.unit === "index");
    if (mean === undefined) continue;
    // Matched by NAME, not unit alone: pixel quality reports its own "%" and
    // "pixels" under the same prefix (`ndvi_quality_valid_percent`), and taking
    // the first "%" set the valid-pixel share under "above threshold" for an
    // index that has no threshold at all.
    groups.push({
      ...family,
      mean,
      percent: own.find(
        (measurement) =>
          measurement.unit === "%" &&
          measurement.name.toLowerCase().includes("_percent_above_"),
      ),
      validPixels:
        own.find(
          (measurement) =>
            measurement.unit === "pixels" &&
            measurement.name.toLowerCase().endsWith("_valid_pixel_count"),
        ) ?? own.find((measurement) => measurement.unit === "pixels"),
    });
  }
  return groups;
}

/* ======================================================= query card (centre) */

/**
 * The natural-language query: the primary input of the workspace.
 *
 * The field starts empty and stays the user's. Nothing is prefilled, no
 * location is assumed, and the placeholder is a prompt rather than a
 * fully-formed question so an empty field can never read as an entered one.
 */
export function AgentQueryCard({ run }: { run: AgentRun }) {
  const { question, setQuestion, askState, result, busy, canAsk } = run;
  const step = executeStep(result);

  return (
    <section className="panel query-card" aria-labelledby="agent-heading">
      <h2 id="agent-heading" className="eyebrow">
        Natural-language query
      </h2>

      <form onSubmit={run.handleAsk} className="agent-form">
        <label className="sr-only" htmlFor="agent-question">
          Question
        </label>
        <div className="query-row">
          <textarea
            id="agent-question"
            className="query-input"
            name="question"
            rows={2}
            value={question}
            disabled={busy}
            placeholder="Ask about any place — a city, district, landmark or coordinates…"
            aria-describedby="agent-capabilities"
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={(event) => {
              // Enter runs the question; Shift+Enter keeps a line break. An
              // IME composition's Enter confirms the composition, not the run.
              if (
                event.key === "Enter" &&
                !event.shiftKey &&
                !event.nativeEvent.isComposing
              ) {
                void run.handleAsk(event);
              }
            }}
          />
          <div className="query-actions">
            <button
              type="button"
              className="link-button"
              onClick={run.clear}
              disabled={busy || (question === "" && result === null)}
            >
              Clear
            </button>
            <button
              type="submit"
              className="btn-primary"
              disabled={!canAsk}
              title={
                question.trim() === ""
                  ? "Enter a question to run an analysis"
                  : undefined
              }
            >
              {busy ? "Running…" : "Run analysis"}
            </button>
          </div>
        </div>
      </form>

      <p id="agent-capabilities" className="query-hint">
        Ask for vegetation (NDVI), water (NDWI), built-up area (NDBI), radar
        backscatter, or water change between two periods — for a named place
        and month. Enter to run.
      </p>

      {step === undefined ? (
        <div className="query-sub">
          <span className="agent-examples-label">Try</span>
          {EXAMPLE_QUESTIONS.map((example, index) => (
            <Fragment key={example.label}>
              {index > 0 && <span className="example-sep">·</span>}
              <button
                type="button"
                className="example-link"
                disabled={busy}
                title={example.question}
                onClick={() => void run.ask(example.question)}
              >
                {example.label}
              </button>
            </Fragment>
          ))}
        </div>
      ) : (
        // What the server understood the question to mean, read back from the
        // plan it validated - never parsed from the question text here.
        <div className="query-sub query-resolved">
          <span className="resolved-place">{step.intent.location_query}</span>
          <span className="example-sep">·</span>
          <span>{formatWindows(step.intent.time_windows)}</span>
          <span className="example-sep">·</span>
          <span>
            {step.intent.modalities
              .map((modality) => MODALITY_LABELS[modality] ?? modality)
              .join(" · ")}
          </span>
          <span className="example-sep">·</span>
          <span>{TASK_LABELS[step.intent.task] ?? step.intent.task}</span>
        </div>
      )}

      {result?.status === "needs_clarification" && result.clarification && (
        <ClarificationPrompt
          clarification={result.clarification}
          disabled={busy}
          onChoose={(choice) => void run.ask(choice)}
        />
      )}

      {askState.status === "error" && (
        <p className="result-error" role="alert">
          {askState.message}
        </p>
      )}
    </section>
  );
}

/**
 * The question the server put back, placed where the user will answer it.
 *
 * Everything shown is the server's: its message, the choices it named as
 * supported, and what it had already understood. Nothing is inferred here and
 * nothing is guessed. A choice the server phrased as a complete question
 * CONTINUES the workflow: it is put in the query box and run, exactly as if it
 * had been typed - and if it still lacks something (a month, say) the server
 * asks for that next, rather than this panel inventing it.
 */
function ClarificationPrompt({
  clarification,
  disabled,
  onChoose,
}: {
  clarification: AgentClarification;
  disabled: boolean;
  onChoose: (question: string) => void;
}) {
  const questions = clarification.option_questions ?? [];
  const choosable = questions.length === clarification.options.length;
  const understood = [
    ...clarification.understood_analyses,
    ...(clarification.understood_location
      ? [clarification.understood_location]
      : []),
    ...clarification.understood_periods.map(
      (period) => `${period.start_date} → ${period.end_date}`,
    ),
  ];
  return (
    <div
      className="clarification"
      role="status"
      data-reason={clarification.reason}
    >
      <p className="clarification-message">{clarification.message}</p>
      {clarification.options.length > 0 && (
        <ul className="clarification-options" aria-label="Supported choices">
          {clarification.options.map((option, index) => (
            <li key={option}>
              {choosable ? (
                <button
                  type="button"
                  className="clarification-option"
                  disabled={disabled}
                  title={questions[index]}
                  onClick={() => onChoose(questions[index])}
                >
                  {capitalise(option)}
                </button>
              ) : (
                capitalise(option)
              )}
            </li>
          ))}
        </ul>
      )}
      {understood.length > 0 && (
        <p className="clarification-understood">
          Understood so far: {understood.join(" · ")}
        </p>
      )}
    </div>
  );
}

/** Display form only: "vegetation (NDVI)" -> "Vegetation (NDVI)". */
function capitalise(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

/* ========================================================= pipeline (centre) */

/**
 * What actually ran, as one strip.
 *
 * The stages are the real executed steps and the run state is derived from
 * them. The only duration shown is the round trip this client measured: the
 * backend reports per-step status and no timing at all, so no per-stage figure
 * is invented to fill the row.
 */
export function AgentPipeline({ run }: { run: AgentRun }) {
  const { result, askState, busy } = run;
  const steps = result?.trace.steps ?? [];
  const elapsedMs = askState.status === "done" ? askState.elapsedMs : null;

  const failed = steps.some((step) => step.status === "failed");
  const incomplete = steps.some(
    (step) => step.status === "rejected" || step.status === "skipped",
  );
  // An empty step list is not a completed run: when planning failed nothing
  // executed, and "complete" over an empty trace would contradict the run.
  const state =
    result === null
      ? busy
        ? "running"
        : "idle"
      : result.status === "needs_clarification"
        ? "clarify"
        : result.status === "location_unavailable"
        ? "unavailable"
        : result.trace.plan === null || steps.length === 0
        ? "not-run"
        : failed
          ? "failed"
          : incomplete
            ? "partial"
            : "complete";

  const stateLabel: Record<string, string> = {
    idle: "Awaiting query",
    running: "Running",
    "not-run": "Not run",
    clarify: "Needs clarification",
    unavailable: "Location unavailable",
    failed: "Failed",
    partial: "Partial",
    complete: "Complete",
  };

  const liveSeconds = useRunningSeconds(busy);

  return (
    <section
      className="pipeline-strip"
      aria-labelledby="pipeline-heading"
      data-busy={busy || undefined}
    >
      <h2 id="pipeline-heading" className="eyebrow">
        Pipeline
      </h2>

      {busy ? (
        // One request, no progress events: what is running is said, and no
        // stage is claimed to be active before the response says it ran.
        <p className="run-running" role="status">
          Understanding the question, resolving the location, finding
          satellite scenes and measuring — each stage is reported when the run
          returns.
        </p>
      ) : result !== null ? (
        <RunStages stages={runStages(result)} />
      ) : null}

      <ol className="stage-flow" aria-label="Technical trace">
        {steps.length === 0 ? (
          <li className="stage-idle">
            {busy ? "" : "no stages executed yet"}
          </li>
        ) : (
          steps.map((step, index) => (
            <StageItem
              key={`${step.parameters.tool}:${index}`}
              step={step}
              last={index === steps.length - 1}
            />
          ))
        )}
      </ol>

      <div className="pipeline-tail">
        {busy && (
          <span className="pipeline-elapsed" title="Time since the question was sent">
            {liveSeconds} s
          </span>
        )}
        {elapsedMs !== null && (
          <span className="pipeline-elapsed" title="Client-measured round trip">
            {(elapsedMs / 1000).toFixed(2)} s
          </span>
        )}
        <span className="pipeline-state" data-state={state}>
          {stateLabel[state]}
        </span>
      </div>
    </section>
  );
}

/** Whole seconds since `active` became true; 0 while inactive. Display only. */
function useRunningSeconds(active: boolean): number {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    if (!active) {
      setSeconds(0);
      return;
    }
    const started = Date.now();
    const timer = setInterval(
      () => setSeconds(Math.floor((Date.now() - started) / 1000)),
      500,
    );
    return () => clearInterval(timer);
  }, [active]);
  return seconds;
}

const STAGE_GLYPHS: Record<RunStage["state"], string> = {
  done: "✓",
  failed: "✕",
  attention: "!",
};

/** The stages the finished run went through, as the response records them. */
function RunStages({ stages }: { stages: RunStage[] }) {
  if (stages.length === 0) return null;
  return (
    <ol className="run-stages" aria-label="What happened">
      {stages.map((stage) => (
        <li key={stage.name} data-state={stage.state}>
          <span className="run-stage-glyph" aria-hidden="true">
            {STAGE_GLYPHS[stage.state]}
          </span>
          <span className="run-stage-name">{stage.name}</span>
          {stage.detail && (
            <span className="run-stage-detail">{stage.detail}</span>
          )}
        </li>
      ))}
    </ol>
  );
}

/**
 * A retrieval that did not produce a picture, said plainly.
 *
 * The server's own words are kept in the disclosure rather than dropped: a
 * documented boundary should be inspectable, not hidden. But the headline is
 * written for a reader, and a known limitation is styled as one - showing it
 * in alarm red teaches a reader to distrust the panel it sits in.
 */
export function ImageryErrorNotice({ raw }: { raw: string }) {
  const notice = describeImageryError(raw);
  return (
    <div
      className={`imagery-notice imagery-notice-${notice.kind}`}
      role={notice.kind === "failure" ? "alert" : "status"}
    >
      <p className="imagery-notice-summary">{notice.summary}</p>
      <p className="imagery-notice-reassurance">{notice.reassurance}</p>
      <details className="imagery-notice-detail">
        <summary>Technical cause</summary>
        <p>{notice.detail}</p>
      </details>
    </div>
  );
}

function StageItem({ step, last }: { step: AgentToolStep; last: boolean }) {
  const glyph =
    step.status === "ok" ? "✓" : step.status === "failed" ? "✕" : "!";
  // Why a step did not succeed. This used to live only in a `title`, which is
  // a mouse-only channel: on a non-focusable <li> a keyboard or screen-reader
  // user had no way to reach it at all, so the one diagnostic that explains a
  // failed run was unreachable to exactly the people who cannot see the
  // colour change. It is rendered as text below.
  const reason = step.error_message ?? step.rejection_reason ?? null;
  return (
    <li
      data-status={step.status}
      data-kind={TOOL_KINDS[step.parameters.tool] ?? "geo"}
    >
      <span className="stage-glyph" aria-hidden="true">
        {glyph}
      </span>
      <span className="stage-name">
        {TOOL_STAGES[step.parameters.tool] ?? step.parameters.tool}
      </span>
      {/* The status is carried by colour and glyph for a sighted reader; this
          says it in words for everyone else. */}
      <span className="sr-only">{` — ${step.status}`}</span>
      {reason !== null && <span className="stage-reason">{reason}</span>}
      {!last && (
        <span className="stage-arrow" aria-hidden="true">
          →
        </span>
      )}
    </li>
  );
}

/* ============================================== deterministic evidence (centre) */

/**
 * Computed from the raster and the STAC record - no model inference.
 *
 * Every row is read out of the execution result the agent already returned. A
 * row whose value the backend did not provide is omitted rather than filled in,
 * and no field is renamed into a stronger claim than the backend makes.
 */
export function AgentEvidencePanel({
  evidence,
  manual = null,
  bbox: manualBbox = null,
  busy = false,
  stale = false,
}: {
  evidence: Evidence | null;
  /** A run is in flight. */
  busy?: boolean;
  /** `evidence` belongs to the previous question; a new one is running. */
  stale?: boolean;
  /** What a manual retrieval established, when no agent run produced it. */
  manual?: {
    scene: SatelliteScene | null;
    imagery: ImageryResponse | null;
    measurements: Measurement[];
    sar_backscatter?: SarBackscatterResult | null;
  } | null;
  bbox?: { west: number; south: number; east: number; north: number } | null;
}) {
  const window = shownWindow(evidence?.execution ?? null);
  // Agent evidence wins when it exists - it describes a run the server
  // validated. The manual path fills the same fields from the same services
  // when no agent run has happened.
  const scene = shownScene(window) ?? manual?.scene ?? null;
  const imagery = window?.imagery ?? manual?.imagery ?? null;
  const bbox = evidence?.execution?.plan.bbox ?? manualBbox;
  // Fall back per-value, exactly as `scene`, `imagery` and `bbox` above do.
  //
  // Selecting the source on `evidence === null` instead meant that an agent
  // result carrying NO measurements - the ordinary `planner_unavailable` case,
  // where the provider was rate limited and nothing ran - suppressed the
  // measurements the manual path had genuinely computed. The scene fields
  // still fell back and rendered, so the panel showed a real scene with its
  // real numbers stripped out: the deterministic evidence panel reporting no
  // evidence while the evidence sat in state.
  const agentMeasurements =
    evidence === null ? [] : measurementsFrom(evidence);
  const measurements =
    agentMeasurements.length > 0
      ? agentMeasurements
      : (manual?.measurements ?? []);

  // A window can discover and select a scene and still fail to retrieve its
  // picture - the documented Sentinel-1 case. The agent path used to drop
  // this entirely, so that run looked identical to one that asked for no
  // imagery at all.
  const imageryError = window?.imagery_error ?? null;

  // One readout per index that reported a mean. Each carries its own
  // threshold percentage and its own valid-pixel count, so no number is ever
  // shown under another index's name.
  // A comparison's two observations share metric names ("ndwi_mean"), so
  // grouping them would set the EARLIER mean alone under "NDWI mean". They are
  // shown by role in "Observations compared" instead.
  const isComparison = (evidence?.items ?? []).some((item) =>
    item.id.startsWith("temporal_ndwi."),
  );
  const groups = isComparison ? [] : indexGroups(measurements);
  // A measurement whose name matches no known index still deserves rendering,
  // but only when grouping found nothing - otherwise it would duplicate a
  // number a group already shows.
  const looseIndex =
    groups.length === 0 && !isComparison
      ? measurements.find((m) => m.unit === "index")
      : undefined;
  const loosePercent =
    groups.length === 0 && !isComparison
      ? measurements.find((m) => m.unit === "%")
      : undefined;
  // The shared pixel count is a property of the window, so it belongs in the
  // field grid - but only while every index agrees on it. When they differ
  // (NDBI resolves through a 20 m band) each group states its own instead of
  // one arbitrary count standing for all of them.
  const pixelCounts = measurements.filter((m) => m.unit === "pixels");
  const sharedPixels =
    pixelCounts.length > 0 &&
    pixelCounts.every((m) => m.value === pixelCounts[0].value)
      ? pixelCounts[0]
      : undefined;
  const validPixels = isComparison
    ? undefined
    : groups.length === 0
      ? pixelCounts[0]
      : sharedPixels;

  const fields: { label: string; value: string; wide?: boolean }[] = [];
  const sceneId = imagery?.scene_id ?? window?.selected_scene_id ?? null;
  if (sceneId) fields.push({ label: "Scene ID", value: sceneId });
  if (scene?.datetime) {
    fields.push({ label: "Acquisition", value: formatInstant(scene.datetime) });
  }
  if (scene) {
    const sensor = [scene.platform, scene.processing_level]
      .filter((part): part is string => Boolean(part))
      .join(" · ");
    if (sensor) fields.push({ label: "Sensor", value: sensor });
  }
  if (scene?.sar_polarizations?.length) {
    fields.push({ label: "SAR polarizations", value: scene.sar_polarizations.join(" / ") });
  }
  if (scene?.sar_instrument_mode) {
    fields.push({ label: "SAR instrument mode", value: scene.sar_instrument_mode });
  }
  if (scene?.orbit_state) {
    fields.push({ label: "Orbit direction", value: scene.orbit_state });
  }
  if (imagery && ["vv", "vh"].includes(imagery.asset)) {
    fields.push({ label: "SAR visualization", value: imagery.normalization, wide: true });
  }
  if (imagery) {
    fields.push({
      label: "Imagery dimensions",
      value: `${imagery.width} × ${imagery.height} px · ${imagery.bands?.length ?? 0} bands`,
    });
    fields.push({
      label: "Ground sample distance",
      value: `${imagery.resolution} m/px`,
    });
  }
  const plan = evidence?.execution?.plan ?? null;
  const match = matchFrom(plan?.matched_name, plan?.matched_class, plan?.matched_type);
  if (match) {
    fields.push({
      label: "Geocoder match",
      value: match.kind ? `${match.full} · ${match.kind}` : match.full,
      wide: true,
    });
  }
  const catalog = window?.catalog ?? evidence?.execution?.catalog ?? null;
  if (catalog) fields.push({ label: "Source catalog", value: catalogName(catalog) });
  if (window && window.scene_count > 0) {
    fields.push({
      label: "Scenes matched",
      // The server's deterministic selection rule, per sensor.
      value: `${window.scene_count} · ${
        window.modality === "sentinel-1-sar"
          ? "earliest acquisition selected"
          : "lowest cloud cover selected"
      }`,
    });
  }
  if (scene?.cloud_cover !== null && scene?.cloud_cover !== undefined) {
    fields.push({
      label: "Cloud cover",
      value: `${scene.cloud_cover.toFixed(1)}% scene`,
    });
  }
  if (imagery?.crs) fields.push({ label: "CRS", value: imagery.crs });
  if (bbox) {
    fields.push({
      label: "Bounds (WGS 84)",
      value:
        `${bbox.south.toFixed(3)} – ${bbox.north.toFixed(3)} N · ` +
        `${bbox.west.toFixed(3)} – ${bbox.east.toFixed(3)} E`,
      wide: true,
    });
  }
  if (validPixels) {
    fields.push({
      label: "Valid pixels",
      value: `${formatMeasurement(validPixels.value, validPixels.unit)} px`,
    });
  }

  const analysis = evidence?.analysis ?? null;
  const comparison = analysis?.temporal_comparison ?? null;

  return (
    <section
      className="panel evidence-panel"
      aria-labelledby="evidence-heading"
      data-stale={stale || undefined}
    >
      <header className="panel-head">
        <span className="mark mark-blue" aria-hidden="true" />
        <h2 id="evidence-heading" className="eyebrow">
          Deterministic evidence
        </h2>
        <span className="panel-note">machine-computed · no model inference</span>
        <span className="panel-source">source: STAC + raster stats</span>
      </header>

      {imageryError !== null && <ImageryErrorNotice raw={imageryError} />}

      {fields.length === 0 && groups.length === 0 && looseIndex === undefined ? (
        <p className="hint" role="status">
          {busy
            ? "Reading scene metadata and computing measurements…"
            : evidence === null
              ? "Scene metadata and computed measurements appear here after a run."
              : window !== null && window.scene_count === 0
                ? "No scenes matched this area and date range, so nothing was measured."
                : "No evidence was collected."}
        </p>
      ) : (
        // A measurement can arrive without scene metadata and vice versa, so
        // each half renders on its own evidence rather than on the other's.
        <div className="evidence-body" data-single={fields.length === 0}>
          <dl className="evidence-grid">
            {fields.map((field) => (
              <div key={field.label} className={field.wide ? "wide" : undefined}>
                <dt>{field.label}</dt>
                <dd>{field.value}</dd>
              </div>
            ))}
          </dl>
          {groups.length > 0 && (
            <div className="index-readouts">
              {groups.map((group) => (
                <IndexReadout
                  key={group.key}
                  index={group.mean}
                  aboveThreshold={group.percent}
                  caveat={group.caveat}
                  validPixels={
                    sharedPixels === undefined ? group.validPixels : undefined
                  }
                />
              ))}
            </div>
          )}
          {looseIndex && (
            <IndexReadout index={looseIndex} aboveThreshold={loosePercent} />
          )}
        </div>
      )}

      <SarBackscatterPanel result={evidence?.analysis?.sar_backscatter ?? manual?.sar_backscatter ?? null} />

      {comparison && evidence && (
        <ObservationPair comparison={comparison} items={evidence.items} />
      )}

      {analysis && <ValidationSummary analysis={analysis} />}

      {evidence !== null && evidence.items.length > 0 && (
        // The citation keys the grounding check resolves against. Technical on
        // purpose: a figure quoted in the answer can be traced to the exact
        // evidence id that produced it. Folded, not removed - every id and raw
        // value stays in the page for anyone auditing the run.
        <details className="evidence-technical">
          <summary>Technical evidence · {evidence.items.length} items</summary>
          <dl className="citation-list">
            {evidence.items.map((item) => (
              <div key={item.id}>
                <dt>{item.id}</dt>
                <dd>{evidenceValue(item)}</dd>
              </div>
            ))}
          </dl>
        </details>
      )}
    </section>
  );
}

/** A catalog URL by the service's host name - the URL itself stays in the export. */
function catalogName(url: string): string {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}

const RADIOMETRY_LABELS: Record<RadiometricState["status"], string> = {
  verified: "verified",
  verified_with_unknown_metadata: "verified · some metadata not published",
  incompatible: "refused · incompatible",
  undetermined: "refused · undetermined",
};

/**
 * The checks the scientific core ran before any number existed: how many
 * pixels were usable, whether the values were on the representation the
 * formula assumes, and whether the rasters combined share a verified grid.
 * Stated from the backend's own records, in its own terms.
 */
function ValidationSummary({ analysis }: { analysis: NonNullable<Evidence["analysis"]> }) {
  const records = validationRecords(analysis);
  const quality: PixelQuality[] = records.quality;
  const radiometry = records.radiometry;
  const grids = records.grids.filter(
    (grid) => grid.stage === "post_read" || grid.status === "refused",
  );
  if (quality.length === 0 && radiometry.length === 0 && grids.length === 0) return null;

  return (
    <section className="validation-summary" aria-label="Quality and validation">
      <h3 className="field-label">Quality &amp; validation</h3>
      <ul>
        {quality.map((entry) => (
          <li key={`pq:${entry.index}:${entry.window_label}:${entry.scene_id}`}>
            <span className="check-name">
              Pixel quality · {entry.index.toUpperCase()}
              {entry.window_label !== "single" ? ` · ${entry.window_label}` : ""}
            </span>
            <span className="check-value">
              {entry.valid_fraction === null
                ? `${formatMeasurement(entry.valid_pixels, "pixels")} of ${formatMeasurement(entry.total_pixels, "pixels")} pixels usable`
                : `${(entry.valid_fraction * 100).toFixed(1)}% usable — ${formatMeasurement(entry.valid_pixels, "pixels")} of ${formatMeasurement(entry.total_pixels, "pixels")} pixels`}
              {entry.masked_pixels > 0 &&
                `; ${formatMeasurement(entry.masked_pixels, "pixels")} masked by the Scene Classification Layer (${formatMeasurement(entry.cloud_pixels, "pixels")} cloud, ${formatMeasurement(entry.cloud_shadow_pixels, "pixels")} shadow)`}
            </span>
          </li>
        ))}
        {radiometry.map((state) => (
          <li key={`rad:${state.scene_id}:${state.modality}`} data-refused={state.status === "incompatible" || state.status === "undetermined" || undefined}>
            <span className="check-name">Radiometry · {state.scene_id}</span>
            <span className="check-value">
              {RADIOMETRY_LABELS[state.status] ?? state.status}
              {state.processing_baseline ? ` · baseline ${state.processing_baseline}` : ""}
            </span>
          </li>
        ))}
        {grids.map((grid, index) => (
          <li key={`grid:${grid.analysis}:${grid.scene_id}:${index}`} data-refused={grid.status === "refused" || undefined}>
            <span className="check-name">Geometry · {grid.analysis.replace(/_/g, " ")}</span>
            <span className="check-value">
              {grid.status === "valid"
                ? ["grid verified", grid.crs, grid.resolution_x !== null ? `${grid.resolution_x} m` : null,
                   grid.width !== null && grid.height !== null ? `${grid.width} × ${grid.height} px` : null]
                    .filter((part): part is string => Boolean(part))
                    .join(" · ")
                : `refused${grid.refusal ? ` — ${grid.refusal}` : ""}`}
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}

/** Both observations of a comparison: which scene answered each period. */
function ObservationPair({
  comparison,
  items,
}: {
  comparison: TemporalIndexComparison;
  items: Evidence["items"];
}) {
  const value = (id: string) =>
    items.find((item) => item.id === id)?.measurement ?? null;
  return (
    <section className="observation-pair" aria-label="Observations compared">
      <h3 className="field-label">Observations compared</h3>
      <dl className="evidence-grid">
        {([["Earlier", "first", comparison.first], ["Later", "second", comparison.second]] as const).map(
          ([role, key, observation]) => {
            const mean = value(`temporal_ndwi.${key}.ndwi_mean`);
            const valid = value(`temporal_ndwi.${key}.ndwi_valid_pixel_count`);
            return (
              <div key={role} className="wide">
                <dt>{role} · {observation.window_label}</dt>
                <dd>
                  {observation.scene_id}
                  {observation.acquired_at ? ` · ${formatInstant(observation.acquired_at)}` : ""}
                  {observation.cloud_cover !== null ? ` · ${observation.cloud_cover.toFixed(1)}% cloud` : ""}
                  {mean !== null && ` · NDWI mean ${formatMeasurement(mean.value, mean.unit)}`}
                  {valid !== null && ` over ${formatMeasurement(valid.value, valid.unit)} valid pixels`}
                </dd>
              </div>
            );
          },
        )}
      </dl>
    </section>
  );
}

/**
 * An index measurement against the scale it lives on.
 *
 * The tick is positioned from the measured value. It is labelled as an index
 * and NEVER as a water classification: the backend computes a spectral index,
 * and calling a threshold "water" would claim a validation this system has not
 * performed.
 */
function IndexReadout({
  index,
  aboveThreshold,
  caveat = "Spectral index only — not a validated land-cover classification.",
  validPixels,
}: {
  index: Measurement;
  /**
   * A threshold percentage over THIS index's pixels, or `undefined`. The
   * caller must never pass another index's percentage: the two sit on one
   * line and read as one measurement.
   */
  aboveThreshold: Measurement | undefined;
  /** What this index is not a classification of. Defaults to the general form. */
  caveat?: string;
  /** This index's own valid-pixel count, when it differs from its siblings'. */
  validPixels?: Measurement | undefined;
}) {
  const clamped = Math.max(-1, Math.min(1, index.value));
  const position = ((clamped + 1) / 2) * 100;

  return (
    <div className="index-readout">
      <span className="field-label">{index.name.replace(/_/g, " ")}</span>
      <div className="index-value-row">
        <span className="index-value">
          {formatMeasurement(index.value, index.unit)}
        </span>
        {aboveThreshold && (
          <span className="index-note">{thresholdNote(aboveThreshold)}</span>
        )}
      </div>
      <div className="index-ramp" />
      <div className="index-axis">
        <span className="axis-min">−1.0</span>
        <span className="axis-zero" style={{ left: "50%" }}>
          0
        </span>
        <span className="axis-max">+1.0</span>
        <span className="index-tick" style={{ left: `${position}%` }} />
      </div>
      {validPixels && (
        <p className="index-pixels">
          {formatMeasurement(validPixels.value, validPixels.unit)} valid pixels
        </p>
      )}
      <p className="index-caveat">{caveat}</p>
    </div>
  );
}

/* ========================================================== answer (right rail) */

/**
 * The grounded answer, or an honest statement of its absence.
 *
 * When the backend withheld the answer nothing is substituted for it: the
 * evidence panel stands on its own, which is the point of returning it.
 */
export function AgentAnswerPanel({
  result,
  asked = null,
  manualComplete = false,
  busy = false,
  stale = false,
}: {
  result: AgentResult | null;
  /**
   * What THIS result was actually asked, frozen at submission.
   *
   * Never the live selector. Switching provider while a request is in flight
   * used to leave the returning result - or its failure - sitting under the
   * newly selected provider's name, blaming a service that was never called.
   */
  asked?: { question: string; provider: string | null; model: string | null } | null;
  /**
   * A manual analysis finished and produced evidence.
   *
   * The manual path produces no written answer - there is no question to
   * answer - so this does not fabricate one. It replaces an instruction to run
   * something with a statement that something ran.
   */
  manualComplete?: boolean;
  /** A run is in flight: say so rather than showing the resting invitation. */
  busy?: boolean;
  /**
   * `result` belongs to the PREVIOUS question: a new one is running. It stays
   * readable, marked as previous, until the new result replaces it.
   */
  stale?: boolean;
}) {
  if (result === null) {
    return (
      <section className="panel answer-panel" aria-labelledby="answer-heading">
        <h2 id="answer-heading" className="eyebrow">
          Analysis result
        </h2>
        <p className="answer-absent" role="status">
          {busy
            ? "Analysing — the answer appears once the evidence is grounded."
            : manualComplete
              ? "Analysis complete. This run was configured manually, so no written answer was generated — the measurements and scene provenance are in the deterministic evidence panel, and can be exported."
              : "Run an analysis to produce an answer grounded in the deterministic evidence."}
        </p>
      </section>
    );
  }

  const outcome = outcomeOf(result);
  const body = resultSummary(result);
  const context = resultContext(result);
  // A measured result reads plain English first; its specialist record - the
  // grounded sentence and its checks included - folds into Technical details.
  const interpretation = outcome.kind === "success" ? interpretResult(result) : null;
  const technical = outcome.kind === "success" ? technicalDetails(result) : null;

  return (
    <section
      className="panel answer-panel"
      aria-labelledby="answer-heading"
      data-outcome={outcome.kind}
      data-stale={stale || undefined}
    >
      <div className="answer-head">
        <h2 id="answer-heading" className="eyebrow">
          Analysis result
        </h2>
        <span className="outcome-chip" data-kind={outcome.kind}>
          {outcome.label}
        </span>
      </div>

      {stale && (
        <p className="stale-note">
          Previous result — the new question is still running.
        </p>
      )}

      {result.status === "ok" && result.answer !== null ? (
        <>
          {outcome.kind === "success" ? (
            <>
              <ResultCard body={body} change={interpretation?.change} />
              <InterpretationBlock interpretation={interpretation} />
            </>
          ) : (
            <MeasurementAbsent
              kind={outcome.kind}
              reason={outcome.reason}
              sceneCount={context.sceneCount}
            />
          )}
          <ResultContextList context={context} body={body} />
          {technical !== null ? (
            <TechnicalDetails rows={technical}>
              <p
                className="agent-answer"
                data-length={answerLength(result.answer)}
                data-role="summary"
              >
                {result.answer}
              </p>
              <ValidationRow validation={result.trace.answer_validation} />
            </TechnicalDetails>
          ) : (
            <p
              className="agent-answer"
              data-length={answerLength(result.answer)}
              data-role={outcome.kind === "success" ? "summary" : "verdict"}
            >
              {result.answer}
            </p>
          )}
        </>
      ) : (
        <StatusNoticeBlock
          status={result.status}
          outcome={outcome.kind}
          measured={measurementsFrom(result.evidence).length}
          failure={result.failure ?? null}
        />
      )}

      <RunAttribution asked={asked} />

      {technical === null && <ValidationRow validation={result.trace.answer_validation} />}

      {/* What a provider failure left behind. A question back, a refusal and a
          location outage ran nothing, so counting their empty evidence would
          only add zeros. */}
      {result.answer === null && outcome.kind === "provider_failure" && (
        <CollectedSummary evidence={result.evidence} />
      )}
    </section>
  );
}

/* ---------------------------------------------------- the result, by operation */

/** "sentinel-2b" -> "Sentinel-2B". Display casing of the catalog's own value. */
function platformLabel(platform: string | null): string | null {
  if (!platform) return null;
  return platform.replace(
    /^sentinel-(\d)([a-z]?)$/i,
    (_, number: string, unit: string) => `Sentinel-${number}${unit.toUpperCase()}`,
  );
}

function qualityText(quality: Quality): string {
  const share =
    quality.validFraction === null
      ? null
      : `${(quality.validFraction * 100).toFixed(1)}%`;
  return share === null
    ? `${pixels(quality.valid)} of ${pixels(quality.total)}`
    : `${share} · ${pixels(quality.valid)} of ${pixels(quality.total)}`;
}

/**
 * The measured value, set as the headline - by operation, never generically.
 *
 * An index is a mean on a -1..+1 scale; backscatter is decibels per
 * polarization; a comparison is two periods and the change between them. Each
 * is laid out as what it is, from the backend's own values.
 */
function ResultCard({ body, change }: { body: ResultBody; change?: string }) {
  switch (body.kind) {
    case "index":
      return (
        <div className="result-card" data-kind="index" data-count={body.readings.length}>
          {body.readings.map((reading) => (
            <div key={reading.key} className="result-reading">
              <p className="result-op">
                {reading.title} <span className="result-op-code">· {reading.label}</span>
              </p>
              <p className="result-value">{signedIndex(reading.mean)}</p>
              {/* The range and the exact value live in Technical details. */}
              {reading.validPixels !== null && (
                <p className="result-sub">
                  Average over {pixels(reading.validPixels)} usable pixels
                </p>
              )}
            </div>
          ))}
        </div>
      );
    case "sar": {
      // Plain names for the two radar measurements; their technical names
      // (VV, VH) and the product (γ⁰, RTC) are in Technical details.
      const both = body.reading.vv !== null && body.reading.vh !== null;
      return (
        <div className="result-card" data-kind="sar">
          <p className="result-op">
            Radar measurements <span className="result-op-code">· Sentinel-1 radar</span>
          </p>
          <dl className="result-sar">
            {body.reading.vv !== null && (
              <div>
                <dt>{both ? "First measurement" : "Measurement"}</dt>
                <dd>{decibels(body.reading.vv)}</dd>
              </div>
            )}
            {body.reading.vh !== null && (
              <div>
                <dt>{both ? "Second measurement" : "Measurement"}</dt>
                <dd>{decibels(body.reading.vh)}</dd>
              </div>
            )}
            {both && body.reading.difference !== null && (
              <div>
                <dt>Difference</dt>
                <dd>{decibels(body.reading.difference)}</dd>
              </div>
            )}
          </dl>
          {body.reading.validPixels !== null && (
            <p className="result-sub">
              Averages over {pixels(body.reading.validPixels)} usable pixels
            </p>
          )}
        </div>
      );
    }
    case "temporal": {
      const { earlier, later, difference, pairedChange } = body.reading;
      return (
        <div className="result-card" data-kind="temporal">
          <p className="result-op">
            Water change <span className="result-op-code">· two dates</span>
          </p>
          <p className="result-question">What changed</p>
          {change && <p className="result-change">{change}</p>}
          <ol className="result-periods" aria-label="What changed">
            {[earlier, later].map((side) => (
              <li key={side.role} data-role={side.role.toLowerCase()}>
                <span className="period-role">{side.role}</span>
                <span className="period-when">{side.period ?? "—"}</span>
                <span className="period-value">
                  {side.mean === null ? "—" : signedIndex(side.mean)}
                </span>
                {side.acquired && (
                  <span className="period-acquired" title={side.acquired}>
                    image taken {formatDay(side.acquired)}
                  </span>
                )}
              </li>
            ))}
          </ol>
          {/* One change for a reader; the per-pixel change is in How we know
              and Technical details. It stands in only when the overall
              difference was withheld. */}
          {(difference !== null || pairedChange !== null) && (
            <dl className="metric-pair">
              <div>
                <dt>{difference !== null ? "Change" : "Change per pixel"}</dt>
                <dd>{signedIndex(difference ?? (pairedChange as number))}</dd>
              </div>
            </dl>
          )}
          <p className="result-sub">
            {difference === null && pairedChange === null
              ? "The difference was withheld by the analysis; see the evidence for why."
              : "Later minus earlier. A measured change - no cause is inferred."}
          </p>
        </div>
      );
    }
    case "imagery":
      return (
        <div className="result-card" data-kind="imagery">
          <p className="result-op">True-colour image</p>
          <p className="result-sub">
            {body.scene
              ? `Scene ${body.scene.id}${body.scene.acquired ? `, acquired ${body.scene.acquired}` : ""}.`
              : "The scene is on the map."}{" "}
            Imagery only - nothing was measured.
          </p>
        </div>
      );
    case "none":
      return null;
  }
}

/**
 * "What this means": the measured result in plain English.
 *
 * Built by `interpretResult` from the same values the card shows - no model
 * writes it - and set BELOW the number, smaller than it: the measurement is
 * the result, this is how to read it.
 */
function InterpretationBlock({
  interpretation,
}: {
  interpretation: Interpretation | null;
}) {
  if (interpretation === null) return null;
  return (
    <>
      <section className="interpretation" aria-labelledby="interpretation-heading">
        <h3 id="interpretation-heading" className="interpretation-label">
          What this means
        </h3>
        <p className="interpretation-headline">{interpretation.headline}</p>
        {interpretation.sampleWarnings?.map((warning) => (
          <p key={warning} className="interpretation-warning" role="note">
            <span aria-hidden="true">⚠ </span>
            {warning}
          </p>
        ))}
        <p className="interpretation-text">{interpretation.explanation}</p>
        {interpretation.caveat && (
          <p className="interpretation-caveat">{interpretation.caveat}</p>
        )}
      </section>
      <section className="how-we-know" aria-labelledby="how-we-know-heading">
        <h3 id="how-we-know-heading" className="interpretation-label">
          How we know
        </h3>
        <p className="interpretation-text">{interpretation.howWeKnow}</p>
      </section>
    </>
  );
}

/**
 * Layer three: the specialist record, folded away until asked for. Index
 * names, formulas, exact values, sensor, scene and the validation checks -
 * every term the plain layers above deliberately leave out - plus the grounded
 * answer sentence and its checks, passed in as children.
 */
function TechnicalDetails({
  rows,
  children,
}: {
  rows: TechnicalRow[];
  children?: ReactNode;
}) {
  return (
    <details className="technical-details">
      <summary>Technical details</summary>
      <dl className="technical-list">
        {rows.map((row, position) => (
          <div key={`${row.label}:${position}`}>
            <dt>{row.label}</dt>
            <dd>{row.value}</dd>
          </div>
        ))}
      </dl>
      {children}
    </details>
  );
}

/** Where and when the result applies - each row only when the run carries it. */
function ResultContextList({
  context,
  body,
}: {
  context: ReturnType<typeof resultContext>;
  body: ResultBody;
}) {
  const rows: { label: string; value: string; title?: string }[] = [];
  if (context.location) rows.push({ label: "Location", value: context.location });
  if (context.matched) {
    // What was actually measured: the geocoder's own match, not the typed name.
    rows.push({
      label: "Matched",
      value: context.matched.kind
        ? `${context.matched.name} (${context.matched.kind})`
        : context.matched.name,
      title: context.matched.full,
    });
  }
  if (body.kind !== "temporal" && context.periods.length > 0) {
    rows.push({ label: "Period", value: context.periods.join(" · ") });
  }
  const scene =
    body.kind === "index" || body.kind === "sar" || body.kind === "imagery"
      ? body.scene
      : null;
  if (scene) {
    // The satellite that took the image and the day it did - the scene id
    // itself stays in Technical details and the evidence panel.
    const platform = platformLabel(scene.platform);
    rows.push({
      label: "Satellite image",
      value: platform ? `${platform}${body.kind === "sar" ? " (radar)" : ""}` : scene.id,
      title: scene.id,
    });
    const day = formatDay(scene.acquired);
    if (day) rows.push({ label: "Date", value: day, title: scene.acquired ?? undefined });
  }
  if (context.sceneCount !== null && body.kind !== "temporal") {
    rows.push({
      label: "Images found",
      value: String(context.sceneCount),
    });
  }
  if (body.kind === "index") {
    const quality = body.readings.find((reading) => reading.quality !== null)?.quality;
    if (quality) rows.push({ label: "Usable pixels", value: qualityText(quality) });
  } else if (body.kind === "temporal") {
    // Each observation has its own pixels; neither stands for both.
    const share = (quality: Quality | null) =>
      quality?.validFraction === null || quality === null
        ? null
        : `${(quality.validFraction * 100).toFixed(1)}%`;
    const earlier = share(body.reading.earlier.quality);
    const later = share(body.reading.later.quality);
    if (earlier !== null && later !== null) {
      rows.push({ label: "Usable pixels", value: `${earlier} earlier · ${later} later` });
    }
  }

  if (rows.length === 0) return null;
  return (
    <dl className="result-context">
      {rows.map((row) => (
        <div key={row.label}>
          <dt>{row.label}</dt>
          <dd title={row.title}>{row.value}</dd>
        </div>
      ))}
    </dl>
  );
}

/**
 * The run completed and the answer is a verdict, not a measurement: either no
 * scene matched, or a validation check refused to produce a number. The two
 * are said differently, and a refusal keeps the server's own reason.
 */
function MeasurementAbsent({
  kind,
  reason,
  sceneCount,
}: {
  kind: OutcomeKind;
  reason: string | null;
  sceneCount: number | null;
}) {
  return (
    <div className="answer-notice" data-kind={kind} role="status">
      {kind === "analysis_refused" ? (
        <>
          <p className="answer-notice-summary">The analysis was not computed.</p>
          <p className="answer-notice-detail">
            A validation check refused the data rather than report a number it
            could not stand behind. The scene and its provenance are in the
            evidence. Each period selects its own scene, so another month may
            be measurable.
          </p>
          {reason && (
            <details className="answer-notice-reason">
              <summary>Why</summary>
              <p>{reason}</p>
            </details>
          )}
        </>
      ) : (
        <>
          <p className="answer-notice-summary">No measurement could be made.</p>
          <p className="answer-notice-detail">
            {sceneCount === 0
              ? "No satellite scene matched this place and period. A different month may have one."
              : "The run completed, but the evidence did not support a measurement."}
          </p>
        </>
      )}
    </div>
  );
}

/**
 * An absent answer, explained.
 *
 * The count of measurements that DID survive is stated from the evidence
 * rather than described in prose, because it is the fact that separates "the
 * analysis failed" from "the sentence about the analysis failed". A reader who
 * sees five measurements and a missing paragraph can act on the five.
 */
/**
 * Which provider and model produced the result on screen.
 *
 * Read from the run's own snapshot, so it keeps naming the provider that was
 * actually used even after the selector has moved on. Rendered only when a run
 * has happened; before that there is nothing to attribute.
 */
function RunAttribution({
  asked,
}: {
  asked?: { question: string; provider: string | null; model: string | null } | null;
}) {
  if (!asked || (asked.provider === null && asked.model === null)) return null;
  if (asked.provider === STANDARD_INTERPRETER) {
    return (
      <p className="run-attribution" data-testid="run-attribution">
        Interpreted by {providerLabel(asked.provider)}
      </p>
    );
  }
  return (
    <p className="run-attribution" data-testid="run-attribution">
      Produced by{" "}
      {asked.provider !== null ? providerLabel(asked.provider) : "the default provider"}
      {asked.model !== null ? ` · ${asked.model}` : ""}
    </p>
  );
}


/**
 * A question put back is not always a missing detail. A place that does not
 * exist, an area too large to measure and a request SatQuery does not do are
 * refusals, and "the question needs one more detail" would misdescribe them.
 * The server's own words are under the query box; these say which KIND of
 * stop it was.
 */
const CLARIFICATION_NOTICES: Partial<Record<OutcomeKind, StatusNotice>> = {
  location_not_found: {
    summary: "No analysis ran: the place could not be found.",
    detail:
      "The location service found no match for it. Check the spelling, or add the city or state.",
    retryable: false,
  },
  location_is_point: {
    summary: "No analysis ran: the place is only a point on the map.",
    detail:
      "The location service has a single point for it, not an area, and a point has no surrounding area to measure. SatQuery does not draw one around it. Name the area you mean - the note under the query box says what was matched.",
    retryable: false,
  },
  area_too_large: {
    summary: "No analysis ran: the area is too large to measure.",
    detail:
      "Measurements are read at the sensor's native resolution, which limits the size of an area. Name a neighbourhood, park or landmark instead - the note under the query box says how.",
    retryable: false,
  },
  unsupported: {
    summary: "No analysis ran: SatQuery does not do this.",
    detail:
      "It measures vegetation (NDVI), water (NDWI), built-up area (NDBI), radar backscatter and water change between two periods, for a named place and period.",
    retryable: false,
  },
};

function StatusNoticeBlock({
  status,
  outcome,
  measured,
  failure,
}: {
  status: Exclude<AgentStatus, "ok"> | AgentStatus;
  /** Which kind of stop, when the status alone does not say. */
  outcome?: OutcomeKind;
  measured: number;
  failure: AgentResult["failure"];
}) {
  const notice =
    (outcome !== undefined ? CLARIFICATION_NOTICES[outcome] : undefined) ??
    STATUS_NOTICES[status as Exclude<AgentStatus, "ok">];
  if (notice === undefined) {
    return (
      <p className="answer-absent" role="status">
        No answer was produced.
      </p>
    );
  }

  return (
    <div
      className="answer-notice"
      data-kind={status}
      data-outcome={outcome}
      data-measured={measured > 0}
      role="status"
    >
      <p className="answer-notice-summary">{notice.summary}</p>
      <p className="answer-notice-detail">{notice.detail}</p>
      {measured > 0 && (
        <p className="answer-notice-standing">
          {measured} measurement{measured === 1 ? "" : "s"} from this run
          {measured === 1 ? " is" : " are"} shown below and in the
          deterministic evidence panel.
        </p>
      )}
      {failure && failure.stage !== "location" && (
        <p className="answer-notice-detail">
          {failure.stage === "planning" ? "Planning" : "Synthesis"} · {failure.code}: {failure.message}
        </p>
      )}
      {failure?.code === "geocoding_unavailable" ? (
        // The summary already says what failed; this says when to retry, and
        // names the dependency and code for anyone diagnosing it.
        <p className="answer-notice-retry" data-dependency={failure.dependency ?? undefined}>
          {failure.retry_after_seconds !== null &&
          Math.ceil(failure.retry_after_seconds) >= 1
            ? `Try again in about ${Math.ceil(failure.retry_after_seconds)} second${Math.ceil(failure.retry_after_seconds) === 1 ? "" : "s"}.`
            : "Try again shortly."}{" "}
          <span className="answer-notice-code">
            ({failure.dependency ?? "geocoder"} · {failure.code})
          </span>
        </p>
      ) : failure?.code === "rate_limited" ? (
        <p className="answer-notice-retry">
          {failure.retry_after_seconds !== null
            ? // A sub-second wait rounds to "0 seconds", which reads as broken;
              // and a one-second wait must not read "1 seconds".
              Math.ceil(failure.retry_after_seconds) < 1
              ? "The provider asked to be retried shortly."
              : `The provider requested a wait of ${Math.ceil(failure.retry_after_seconds)} second${Math.ceil(failure.retry_after_seconds) === 1 ? "" : "s"} before retrying.`
            : "The provider quota is exhausted. Wait for quota to reset or explicitly select another configured provider."}
        </p>
      ) : notice.retryable && !failure && (
        <p className="answer-notice-retry">
          Running the same question again may produce the summary.
        </p>
      )}
    </div>
  );
}

/** How far the answer must shrink to hold the rail. Presentation only. */
function answerLength(answer: string): "short" | "long" | "very-long" {
  if (answer.length > 260) return "very-long";
  if (answer.length > 150) return "long";
  return "short";
}

/**
 * The checks that actually ran.
 *
 * The reference marks the answer "verified — model observation and NDWI
 * measurement agree". Nothing in this system establishes that agreement:
 * grounding checks that the numbers trace to evidence, that citations resolve
 * and that no forbidden phrase appears - it cannot confirm a model's
 * description of a picture. So this reports the checks that did run, and
 * `visual_claims` reads "attributed", which is provenance and never a verdict.
 */
function ValidationRow({
  validation,
}: {
  validation: AgentResult["trace"]["answer_validation"];
}) {
  if (validation === null) return null;

  const checks = [
    { label: "Numbers grounded", outcome: validation.numeric_grounding },
    { label: "Citations resolve", outcome: validation.evidence_refs },
    { label: "Terminology", outcome: validation.forbidden_terms },
  ].filter((check) => check.outcome !== "not_run");

  const allPass = checks.every((check) => check.outcome === "pass");

  return (
    <div className={`validation-box${allPass ? "" : " validation-box-bad"}`}>
      <div className="validation-line">
        {checks.map((check) => (
          <span
            key={check.label}
            className={`verdict ${check.outcome === "pass" ? "" : "verdict-bad"}`}
          >
            {check.outcome === "pass" ? check.label : `${check.label} — failed`}
          </span>
        ))}
      </div>
      {validation.visual_claims === "attributed" && (
        <p className="validation-note">
          A model observation contributed to this answer. It is attributed, not
          verified.
        </p>
      )}
    </div>
  );
}

/** What a run produced when it produced no answer. Counted, never estimated. */
function CollectedSummary({ evidence }: { evidence: Evidence }) {
  const scene = shownWindow(evidence.execution)?.selected_scene_id ?? null;
  return (
    <dl className="collected-summary">
      <div>
        <dt>Evidence items</dt>
        <dd>{evidence.items.length}</dd>
      </div>
      <div>
        <dt>Measurements</dt>
        <dd>{measurementsFrom(evidence).length}</dd>
      </div>
      <div>
        <dt>Scene selected</dt>
        <dd>{scene ?? "none"}</dd>
      </div>
    </dl>
  );
}

/* =============================================== visual observation (right rail) */

/**
 * The model's reading of the picture, kept apart from the computed values.
 *
 * Everything in the evidence panel was computed from pixels or metadata; this
 * was written by a model looking at an image, and nothing can check it. That is
 * said three ways on purpose: the violet key, the header qualifier, and the
 * amber caveat.
 */
export function AgentObservationPanel({
  evidence,
  result = null,
  busy = false,
  stale = false,
  onExport,
}: {
  evidence: Evidence | null;
  /**
   * The whole run, when the caller has it. The plan is what says whether a
   * visual reading was ever ASKED FOR, and the panel cannot tell the truth
   * without it: absent evidence alone cannot distinguish a model that looked
   * and found nothing from a model that was never called.
   */
  result?: AgentResult | null;
  /** A run is in flight. */
  busy?: boolean;
  /** The evidence belongs to the previous question; a new one is running. */
  stale?: boolean;
  onExport?: () => void;
}) {
  const observations = (evidence?.items ?? []).filter(
    (item) => item.source === "model" && item.visual !== null,
  );
  const state = visualStepState(result);

  return (
    <section
      className="panel observation-panel"
      aria-labelledby="observation-heading"
      data-stale={stale || undefined}
      data-empty={observations.length === 0 || undefined}
    >
      <header className="panel-head">
        <span className="mark mark-violet" aria-hidden="true" />
        <h2 id="observation-heading" className="eyebrow">
          Visual observation
        </h2>
      </header>

      {observations.length === 0 ? (
        <p className="hint" role="status">
          {busy
            ? "A visual reading appears here only if the question asks what the scene looks like."
            : evidence === null
              ? "A vision-language reading of the scene appears here when a question calls for one."
              : // Each of these is a DIFFERENT event, and the panel used to
                // report all of them as "no visual interpretation was
                // produced" - which reads as a model that looked and saw
                // nothing, or as a failure. Usually neither happened.
                state.kind === "not-requested"
                ? "The planner did not request a visual observation for this query, so no image was sent to a model. Deterministic results, when produced, appear in the evidence panel."
                : state.kind === "failed"
                  ? `A visual observation was requested but the model step did not complete${
                      state.step.error_message ??
                      state.step.rejection_reason
                        ? `: ${state.step.error_message ?? state.step.rejection_reason}`
                        : "."
                    }`
                  : // no-plan: nothing was planned, so nothing was asked.
                    "No plan was produced for this run, so no visual observation was requested."}
        </p>
      ) : (
        observations.map((item) => (
          <div key={item.id} className="observation-body">
            <p className="agent-visual-statement">{item.visual!.statement}</p>
            <p className="model-line">
              <span className="model-name">
                Model observation · {providerLabel(item.visual!.provider)} ·{" "}
                {item.visual!.model}
              </span>
              <span className="model-meta">scene {item.visual!.scene_id}</span>
            </p>
            <p className="qualitative-note">
              <span className="mark mark-amber" aria-hidden="true" />
              Qualitative interpretation of pixels, not a measurement. Use the
              deterministic evidence panel for reportable values.
            </p>
          </div>
        ))
      )}

      {onExport && (
        <div className="rail-actions">
          <button type="button" className="btn-primary" onClick={onExport}>
            Export evidence (JSON)
          </button>
        </div>
      )}
    </section>
  );
}

/* ============================================================== composed panel */

/**
 * The whole agent surface in one component.
 *
 * The workspace places these regions in three separate columns and drives them
 * from `useAgentRun`; this composition keeps them together so the panel can
 * still be exercised on its own.
 */
export function AgentPanel(handlers: AgentRunHandlers = {}) {
  const run = useAgentRun(handlers);
  return (
    <div className="agent-root" data-has-result={run.result !== null}>
      <AgentQueryCard run={run} />
      <AgentPipeline run={run} />
      <AgentAnswerPanel result={run.displayed} stale={run.stale} busy={run.busy} />
      <AgentObservationPanel
        evidence={run.displayed?.evidence ?? null}
        result={run.displayed}
        stale={run.stale}
      />
      <AgentEvidencePanel
        evidence={run.displayed?.evidence ?? null}
        stale={run.stale}
      />
    </div>
  );
}
