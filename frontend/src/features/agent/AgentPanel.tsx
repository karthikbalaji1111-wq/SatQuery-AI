import { Fragment } from "react";

import type {
  AgentResult,
  AgentStatus,
  AgentToolName,
  AgentToolStep,
  AgentEvidence as Evidence,
  Measurement,
  ImageryResponse,
  Modality,
  QueryTask,
  SatelliteScene,
  SatQueryIntent,
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
import { useAgentRun } from "./agentRun";
import type { AgentRun, AgentRunHandlers } from "./agentRun";

/**
 * Starting points, one per capability the agent actually has. Clicking fills
 * the box rather than submitting: the question stays the user's to edit.
 */
const EXAMPLE_QUESTIONS = [
  {
    label: "Visible water",
    question:
      "Is there visible water in the Sentinel-2 image of Marina Beach, Chennai?",
  },
  {
    label: "Water index",
    question: "What is the NDWI of Marina Beach, Chennai in January 2025?",
  },
  {
    label: "Vegetation",
    question:
      "What is the vegetation condition around Bengaluru in December 2024?",
  },
  {
    label: "Built-up area",
    question: "Analyse the built-up area around Hyderabad in December 2024.",
  },
  {
    label: "Temporal comparison",
    question: "Compare NDWI at Marina Beach between January and July 2025.",
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
  temporal_ndwi_statistics: "ndwi_temporal",
  rs_model_analysis: "vlm_observe",
};

const TOOL_KINDS: Record<AgentToolName, string> = {
  execute_query: "geo",
  spectral_indices: "index",
  ndwi_statistics: "index",
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
  mock: "Mock",
};

function providerLabel(provider: string): string {
  return PROVIDER_LABELS[provider] ?? provider;
}

/** Presentation-layer formatting. API values are never rounded; only this is. */
function formatMeasurement(value: number, unit: string): string {
  if (!Number.isFinite(value)) return String(value);
  switch (unit) {
    case "index":
      return `${value >= 0 ? "+" : ""}${value.toFixed(4)}`;
    case "%":
      return value.toFixed(1);
    case "pixels":
    case "count":
      return Math.round(value).toLocaleString("en-US");
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
 * The two measurements worth setting at headline size.
 *
 * Analytical units only. A discovery count is a fact about the search, not a
 * result of the analysis, and setting one large would give the reader the wrong
 * headline. When nothing analytical was computed the block is absent.
 */
function headlineMeasurements(evidence: Evidence): Measurement[] {
  const all = measurementsFrom(evidence);
  const index = all.find((measurement) => measurement.unit === "index");
  const percent = all.find((measurement) => measurement.unit === "%");
  const paired = [index, percent].filter(
    (measurement): measurement is Measurement => measurement !== undefined,
  );
  if (paired.length === 2) return paired;
  return all
    .filter(
      (measurement) => measurement.unit === "index" || measurement.unit === "%",
    )
    .slice(0, 2);
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
            onChange={(event) => setQuestion(event.target.value)}
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
                onClick={() => setQuestion(example.question)}
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

      {askState.status === "error" && (
        <p className="result-error" role="alert">
          {askState.message}
        </p>
      )}
    </section>
  );
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
    failed: "Failed",
    partial: "Partial",
    complete: "Complete",
  };

  return (
    <section className="pipeline-strip" aria-labelledby="pipeline-heading">
      <h2 id="pipeline-heading" className="eyebrow">
        Pipeline
      </h2>

      <ol className="stage-flow">
        {steps.length === 0 ? (
          <li className="stage-idle">
            {busy ? "planning…" : "no stages executed yet"}
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
}: {
  evidence: Evidence | null;
  /** A run is in flight. */
  busy?: boolean;
  /** What a manual retrieval established, when no agent run produced it. */
  manual?: {
    scene: SatelliteScene | null;
    imagery: ImageryResponse | null;
    measurements: Measurement[];
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
  const measurements =
    evidence === null
      ? (manual?.measurements ?? [])
      : measurementsFrom(evidence);

  // A window can discover and select a scene and still fail to retrieve its
  // picture - the documented Sentinel-1 case. The agent path used to drop
  // this entirely, so that run looked identical to one that asked for no
  // imagery at all.
  const imageryError = window?.imagery_error ?? null;

  const index = measurements.find((m) => m.unit === "index");
  const aboveThreshold = measurements.find((m) => m.unit === "%");
  const validPixels = measurements.find((m) => m.unit === "pixels");

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

  return (
    <section className="panel evidence-panel" aria-labelledby="evidence-heading">
      <header className="panel-head">
        <span className="mark mark-blue" aria-hidden="true" />
        <h2 id="evidence-heading" className="eyebrow">
          Deterministic evidence
        </h2>
        <span className="panel-note">machine-computed · no model inference</span>
        <span className="panel-source">source: STAC + raster stats</span>
      </header>

      {imageryError !== null && <ImageryErrorNotice raw={imageryError} />}

      {fields.length === 0 && index === undefined ? (
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
          {index && (
            <IndexReadout index={index} aboveThreshold={aboveThreshold} />
          )}
        </div>
      )}

      {evidence !== null && evidence.items.length > 0 && (
        // The citation keys the grounding check resolves against. Technical on
        // purpose: a figure quoted in the answer can be traced to the exact
        // evidence id that produced it.
        <dl className="citation-list">
          {evidence.items.map((item) => (
            <div key={item.id}>
              <dt>{item.id}</dt>
              <dd>{evidenceValue(item)}</dd>
            </div>
          ))}
        </dl>
      )}
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
}: {
  index: Measurement;
  aboveThreshold: Measurement | undefined;
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
          <span className="index-note">
            {formatMeasurement(aboveThreshold.value, aboveThreshold.unit)}% above
            threshold
          </span>
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
      <p className="index-caveat">
        Spectral index only — not a validated water classification.
      </p>
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
  busy = false,
}: {
  result: AgentResult | null;
  /** A run is in flight: say so rather than showing the resting invitation. */
  busy?: boolean;
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
            : "Run an analysis to produce an answer grounded in the deterministic evidence."}
        </p>
      </section>
    );
  }

  const metrics = headlineMeasurements(result.evidence);

  return (
    <section className="panel answer-panel" aria-labelledby="answer-heading">
      <h2 id="answer-heading" className="eyebrow">
        Analysis result
      </h2>

      {result.status === "ok" && result.answer !== null ? (
        <p className="agent-answer" data-length={answerLength(result.answer)}>
          {result.answer}
        </p>
      ) : (
        <StatusNoticeBlock
          status={result.status}
          measured={measurementsFrom(result.evidence).length}
          failure={result.failure ?? null}
        />
      )}

      <ValidationRow validation={result.trace.answer_validation} />

      {metrics.length > 0 && (
        <dl className="metric-pair">
          {metrics.map((measurement) => (
            <div key={measurement.name}>
              <dt>{measurement.name.replace(/_/g, " ")}</dt>
              <dd>
                {formatMeasurement(measurement.value, measurement.unit)}
                {measurement.unit === "%" && (
                  <span className="metric-unit"> %</span>
                )}
              </dd>
            </div>
          ))}
        </dl>
      )}

      {result.answer === null && <CollectedSummary evidence={result.evidence} />}
    </section>
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
function StatusNoticeBlock({
  status,
  measured,
  failure,
}: {
  status: Exclude<AgentStatus, "ok"> | AgentStatus;
  measured: number;
  failure: AgentResult["failure"];
}) {
  const notice = STATUS_NOTICES[status as Exclude<AgentStatus, "ok">];
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
      {failure && (
        <p className="answer-notice-detail">
          {failure.stage === "planning" ? "Planning" : "Synthesis"} · {failure.code}: {failure.message}
        </p>
      )}
      {failure?.code === "rate_limited" ? (
        <p className="answer-notice-retry">
          {failure.retry_after_seconds !== null
            ? `The provider requested a wait of ${Math.ceil(failure.retry_after_seconds)} seconds before retrying.`
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
            ? "Waiting for the model's reading of the scene…"
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
      <AgentAnswerPanel result={run.result} />
      <AgentObservationPanel
        evidence={run.result?.evidence ?? null}
        result={run.result}
      />
      <AgentEvidencePanel evidence={run.result?.evidence ?? null} />
    </div>
  );
}
