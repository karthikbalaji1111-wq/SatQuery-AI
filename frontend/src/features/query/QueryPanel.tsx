import { type FormEvent, useEffect, useRef, useState } from "react";

import { ApiError } from "../../api/client";
import { shownScene, shownWindow } from "../agent/derive";
import { ImageryErrorNotice } from "../agent/AgentPanel";
import { resolveLocation } from "../../api/geospatial";
import {
  analyzeQuery,
  buildQueryPlan,
  executeQuery,
  parsePrompt,
} from "../../api/query";
import { fetchSceneImagery, searchScenes } from "../../api/satellite";
import type {
  AnalysisResult,
  BoundingBox,
  ExecutedWindow,
  GeoResolveResponse,
  ImageryResponse,
  Measurement,
  Modality,
  NdwiComparison,
  NdwiOverlay,
  NdwiTemporalChange,
  QueryExecutionResult,
  QueryTask,
  ResolvedQueryPlan,
  SatQueryIntent,
  SatelliteScene,
  SceneSearchResponse,
  SpatialMeasurement,
  SpectralIndexKey,
  TemporalComparison,
  TemporalIndexComparison,
  TemporalMode,
} from "../../api/types";
import type { MapAoi } from "../map/footprint";
import { ConfigSummary } from "./ConfigSummary";
import type { RunContext } from "./ConfigSummary";

type ResolveState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "done"; result: GeoResolveResponse };

type SearchState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "done"; result: SceneSearchResponse };

type ImageryState =
  | { status: "idle" }
  | { status: "loading"; sceneId: string }
  | { status: "error"; sceneId: string; message: string }
  | { status: "done"; sceneId: string; result: ImageryResponse };

type PlanState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "done"; result: ResolvedQueryPlan };

type ParseState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "done"; result: SatQueryIntent };

type ExecuteState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "done"; result: QueryExecutionResult };

type AnalyzeState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "done"; result: AnalysisResult };

function errorMessage(error: unknown): string {
  return error instanceof ApiError ? error.message : "Unexpected error";
}

function formatBbox(bbox: BoundingBox): string {
  return (
    `W ${bbox.west.toFixed(4)}, S ${bbox.south.toFixed(4)}, ` +
    `E ${bbox.east.toFixed(4)}, N ${bbox.north.toFixed(4)}`
  );
}

/**
 * Presentation-layer number formatting. The API representation and the backend
 * values are never rounded - only what the user reads is.
 *
 * Unit-driven, because the backend already labels every Measurement: an index
 * keeps 4 decimals (NDWI is meaningful to ~1e-4), a percentage 1, a pixel count
 * is an integer with thousands separators.
 */
function formatMeasurement(value: number, unit: string): string {
  if (!Number.isFinite(value)) return String(value);
  switch (unit) {
    case "index":
      return value.toFixed(4);
    case "%":
      return value.toFixed(1);
    case "pixels":
      return Math.round(value).toLocaleString("en-US");
    default:
      return String(value);
  }
}

/** Day intervals: whole-day precision is what an acquisition gap warrants. */
function formatDays(days: number): string {
  return Number.isFinite(days) ? days.toFixed(1) : String(days);
}

/** Cloud cover, matching the existing scene-list presentation. */
function formatPercent(value: number): string {
  return Number.isFinite(value) ? value.toFixed(1) : String(value);
}

/**
 * Only `visualize` has an analysis engine. The other two are valid intents the
 * backend accepts and answers with `status: "not_implemented"`, so they stay
 * selectable - but they are labelled so no one reads Temporal NDWI Statistics
 * as a change-detection result.
 */
const TASK_OPTIONS: { value: QueryTask; label: string }[] = [
  { value: "visualize", label: "Visualize" },
  { value: "change_detection", label: "Change Detection (unavailable)" },
  {
    value: "object_identification",
    label: "Object Identification (unavailable)",
  },
];

/**
 * Phase flow: natural-language text -> parsed SatQueryIntent (review/edit) ->
 * structured query plan (intent + resolved bbox); and, independently, place ->
 * resolve -> Sentinel-2 discovery -> bounded RGB imagery for a selected scene.
 * NL parsing only fills the form - it never auto-runs Build Query Plan.
 * "Run full query" sends the current intent to /query/execute, which grounds
 * the location, runs discovery per temporal window, deterministically selects a
 * scene, and (optionally) retrieves bounded imagery. The manual step-by-step
 * forms remain available.
 * No map, no AI/VLM image reasoning, no spectral controls.
 */
interface QueryPanelProps {
  /** Claim the workspace before a manual run replaces an agent run. */
  onStart?: () => void;
  /**
   * Called when a scene preview is retrieved, so a parent can place it on the
   * map. Optional: the panel works exactly as before without it, and it never
   * triggers a request of its own - this is the imagery already fetched here.
   */
  onImagery?: (imagery: ImageryResponse | null) => void;
  /**
   * Called with the NDWI overlay a query produced, or `null` when it produced
   * none. Same contract as `onImagery`: the result of a request this panel
   * already made, travelling upward - never a second fetch.
   */
  onNdwi?: (overlay: NdwiOverlay | null) => void;
  /**
   * Called with the temporal NDWI change overlay a query produced, or `null`
   * when the grids were not comparable and nothing was subtracted.
   */
  onChange?: (overlay: NdwiOverlay | null) => void;
  /**
   * The area this panel resolved, so the footprint map can show where the
   * request applies. Same contract as the others: a result already obtained
   * here travelling upward, never a fetch of its own.
   */
  onAoi?: (aoi: MapAoi | null) => void;
  /**
   * The configuration an agent run actually used. When present it is what the
   * rail reports, because it describes the run that produced what is on screen
   * - the manual form's own fields describe a request not yet made.
   */
  runContext?: RunContext | null;
  /**
   * The scene and measurements this panel produced, so the evidence panel can
   * report a manual run as fully as an agent run. Same contract as the others:
   * results already obtained here travelling upward.
   */
  onEvidence?: (evidence: ManualEvidence | null) => void;
}

/** What the manual path can establish without any model call. */
export interface ManualEvidence {
  scene: SatelliteScene | null;
  imagery: ImageryResponse | null;
  measurements: Measurement[];
}

/**
 * The indices offered, with what a HIGH value indicates.
 *
 * Worded as a spectral observation, never as a classification: a high NDBI is
 * a built-up-like reflectance signature, not a building, and the interface
 * must not promise a classifier the system does not have.
 */
const INDEX_CHOICES: { key: SpectralIndexKey; label: string; meaning: string }[] =
  [
    { key: "ndvi", label: "NDVI", meaning: "vegetation-like response" },
    { key: "ndwi", label: "NDWI", meaning: "water-like response" },
    { key: "ndbi", label: "NDBI", meaning: "built-up / bare response" },
  ];

export function QueryPanel({
  onStart,
  onImagery,
  onNdwi,
  onChange,
  onAoi,
  runContext = null,
  onEvidence,
}: QueryPanelProps = {}) {
  const [place, setPlace] = useState("");
  const [resolveState, setResolveState] = useState<ResolveState>({
    status: "idle",
  });

  // --- natural-language intent parsing ---
  const [nlText, setNlText] = useState("");
  const [parseState, setParseState] = useState<ParseState>({ status: "idle" });

  // --- structured query intent ---
  const [temporalMode, setTemporalMode] = useState<TemporalMode>("single");
  const [obsDate, setObsDate] = useState("");
  const [baselineStart, setBaselineStart] = useState("");
  const [baselineEnd, setBaselineEnd] = useState("");
  const [targetStart, setTargetStart] = useState("");
  const [targetEnd, setTargetEnd] = useState("");
  const [opticalOn, setOpticalOn] = useState(true);
  const [sarOn, setSarOn] = useState(false);
  const [task, setTask] = useState<QueryTask>("visualize");
  const [planState, setPlanState] = useState<PlanState>({ status: "idle" });
  const [includeImagery, setIncludeImagery] = useState(false);
  const [includeNdwi, setIncludeNdwi] = useState(false);
  // Which additional spectral indices to compute. Independent of the NDWI
  // flag above, which is also what produces the georeferenced overlay.
  const [indices, setIndices] = useState<SpectralIndexKey[]>([]);
  const [includeTemporalNdwi, setIncludeTemporalNdwi] = useState(false);
  const [executeState, setExecuteState] = useState<ExecuteState>({
    status: "idle",
  });
  const [analyzeState, setAnalyzeState] = useState<AnalyzeState>({
    status: "idle",
  });

  // --- STAC discovery + imagery ---
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [maxCloud, setMaxCloud] = useState("");
  const [searchState, setSearchState] = useState<SearchState>({ status: "idle" });
  const [imageryState, setImageryState] = useState<ImageryState>({
    status: "idle",
  });

  const resolved =
    resolveState.status === "done" ? resolveState.result : undefined;
  const datesInvalid =
    startDate !== "" && endDate !== "" && startDate > endDate;
  const canSearch =
    resolved !== undefined &&
    startDate !== "" &&
    endDate !== "" &&
    !datesInvalid &&
    searchState.status !== "loading";

  const modalities: Modality[] = [
    ...(opticalOn ? (["sentinel-2-optical"] as const) : []),
    ...(sarOn ? (["sentinel-1-sar"] as const) : []),
  ];
  const compareRangesInvalid =
    temporalMode === "compare" &&
    ((baselineStart !== "" && baselineEnd !== "" && baselineStart > baselineEnd) ||
      (targetStart !== "" && targetEnd !== "" && targetStart > targetEnd));
  const temporalComplete =
    temporalMode === "single"
      ? obsDate !== ""
      : baselineStart !== "" &&
        baselineEnd !== "" &&
        targetStart !== "" &&
        targetEnd !== "";
  const planReady =
    place.trim() !== "" &&
    modalities.length > 0 &&
    temporalComplete &&
    !compareRangesInvalid;
  const canBuildPlan = planReady && planState.status !== "loading";
  const canExecute = planReady && executeState.status !== "loading";

  /** Populate the editable Query Plan form from a parsed intent. */
  function applyIntent(intent: SatQueryIntent) {
    setPlace(intent.location_query);
    setOpticalOn(intent.modalities.includes("sentinel-2-optical"));
    setSarOn(intent.modalities.includes("sentinel-1-sar"));
    setTask(intent.task);

    const windows = intent.time_windows;
    if (intent.temporal_mode === "compare" && !Array.isArray(windows)) {
      setTemporalMode("compare");
      setBaselineStart(windows.baseline.start_date);
      setBaselineEnd(windows.baseline.end_date);
      setTargetStart(windows.target.start_date);
      setTargetEnd(windows.target.end_date);
    } else if (Array.isArray(windows) && windows.length > 0) {
      // single, or timeseries collapsed to its first window for the manual form
      setTemporalMode("single");
      setObsDate(windows[0].start_date);
    }
  }

  async function handleParse(event: FormEvent) {
    event.preventDefault();
    const text = nlText.trim();
    if (!text) return;

    setParseState({ status: "loading" });
    try {
      const intent = await parsePrompt(text);
      applyIntent(intent); // fills the form; user must still click Build Query Plan
      setParseState({ status: "done", result: intent });
    } catch (error) {
      setParseState({ status: "error", message: errorMessage(error) });
    }
  }

  // Guards the scene-preview race above: a monotonic ticket identifying the
  // newest request, and the in-flight controller so the superseded one stops.
  const previewTicketRef = useRef(0);
  const previewAbortRef = useRef<AbortController | null>(null);
  const runTicketRef = useRef(0);
  useEffect(() => () => {
    runTicketRef.current += 1;
    previewTicketRef.current += 1;
    previewAbortRef.current?.abort();
  }, []);

  function beginManualRun() {
    const ticket = ++runTicketRef.current;
    previewTicketRef.current += 1;
    previewAbortRef.current?.abort();
    onStart?.();
    onEvidence?.(null);
    onImagery?.(null);
    onNdwi?.(null);
    onChange?.(null);
    onAoi?.(null);
    return () => runTicketRef.current === ticket;
  }

  /**
   * Drop a displayed analysis when the selection it described changes.
   *
   * The result on screen answers the options that were set when it ran. Change
   * those options and it is answering a question nobody is asking any more -
   * an NDWI mean sitting under a picker that now reads NDVI. Clearing is the
   * honest response; re-running is the user's decision.
   */
  function invalidateAnalysis() {
    setAnalyzeState({ status: "idle" });
    onNdwi?.(null);
    onChange?.(null);
  }

  async function handleResolve(event: FormEvent) {
    event.preventDefault();
    const trimmed = place.trim();
    if (!trimmed) return;

    const current = beginManualRun();
    setResolveState({ status: "loading" });
    setSearchState({ status: "idle" });
    setImageryState({ status: "idle" });
    // The map holds the last scene handed to it, so clearing local state is not
    // enough: without this the map keeps showing a scene from the previous
    // location while the panel has already moved on.
    onImagery?.(null);
    try {
      const result = await resolveLocation({ place: trimmed });
      if (!current()) return;
      setResolveState({ status: "done", result });
      // The resolved extent, exactly as the geospatial service returned it.
      onAoi?.({ ...result.bbox, scene_id: null });
    } catch (error) {
      if (!current()) return;
      setResolveState({ status: "error", message: errorMessage(error) });
    }
  }

  /** Assemble a SatQueryIntent from the current Query Plan form state. */
  function currentIntent(): SatQueryIntent {
    const timeWindows: SatQueryIntent["time_windows"] =
      temporalMode === "single"
        ? [{ start_date: obsDate, end_date: obsDate }]
        : ({
            baseline: { start_date: baselineStart, end_date: baselineEnd },
            target: { start_date: targetStart, end_date: targetEnd },
          } satisfies TemporalComparison);

    return {
      location_query: place.trim(),
      temporal_mode: temporalMode,
      time_windows: timeWindows,
      modalities,
      task,
    };
  }

  async function handleBuildPlan(event: FormEvent) {
    event.preventDefault();
    if (!canBuildPlan) return;

    setPlanState({ status: "loading" });
    try {
      const result = await buildQueryPlan(currentIntent());
      setPlanState({ status: "done", result });
    } catch (error) {
      setPlanState({ status: "error", message: errorMessage(error) });
    }
  }

  async function handleExecute() {
    if (!canExecute) return;

    const current = beginManualRun();
    setExecuteState({ status: "loading" });
    // A new query invalidates whatever the map is showing, before any
    // new result exists to replace it.
    onNdwi?.(null);
    onChange?.(null);
    setAnalyzeState({ status: "idle" });

    let result: QueryExecutionResult;
    try {
      result = await executeQuery({
        intent: currentIntent(),
        include_imagery: includeImagery,
      });
      if (!current()) return;
      setExecuteState({ status: "done", result });
      const window = shownWindow(result);
      const imagery = window?.imagery ?? null;
      onImagery?.(imagery);
      onAoi?.({
        ...result.plan.bbox,
        scene_id: window?.selected_scene_id ?? null,
        imagery_requested: includeImagery,
        imagery_error: window?.imagery_error ?? null,
      });
      onEvidence?.({ scene: shownScene(window), imagery, measurements: [] });
    } catch (error) {
      if (!current()) return;
      setExecuteState({ status: "error", message: errorMessage(error) });
      return;
    }

    // Analysis is a separate boundary: a failure here must never discard the
    // execution result that is already rendered.
    setAnalyzeState({ status: "loading" });
    try {
      const analysis = await analyzeQuery({
        execution: result,
        // Omitted when off, so a non-NDWI request is byte-identical
        // to the pre-NDWI behaviour.
        ...(includeNdwi
          ? { include_ndwi: true, include_ndwi_overlay: true }
          : {}),
        ...(includeTemporalNdwi ? { include_temporal_ndwi: true } : {}),
        // Omitted when empty, so a request that asks for no extra index is
        // byte-identical to the previous contract.
        ...(indices.length > 0 ? { indices } : {}),
      });
      if (!current()) return;
      setAnalyzeState({ status: "done", result: analysis });
      // Measurements the analysis computed, reported alongside whatever scene
      // is already on screen.
      onEvidence?.({
        scene: null,
        imagery: null,
        measurements: analysis.measurements,
      });
      // null when the analysis produced no overlay, so the map never keeps one
      // from an earlier query.
      onNdwi?.(analysis.ndwi_overlay ?? null);
      // null when the grids were not comparable, so a stale change overlay can
      // never outlive the comparison that produced it.
      onChange?.(analysis.temporal_comparison?.change?.overlay ?? null);
    } catch (error) {
      if (!current()) return;
      setAnalyzeState({ status: "error", message: errorMessage(error) });
      onNdwi?.(null);
      onChange?.(null);
    }
  }

  async function handleSearch(event: FormEvent) {
    event.preventDefault();
    if (!resolved || !canSearch) return;

    const current = beginManualRun();
    onAoi?.({ ...resolved.bbox, scene_id: null });
    const cloud = maxCloud.trim() === "" ? undefined : Number(maxCloud);
    setSearchState({ status: "loading" });
    setImageryState({ status: "idle" });
    onImagery?.(null);
    try {
      const result = await searchScenes({
        bbox: resolved.bbox,
        start_date: startDate,
        end_date: endDate,
        ...(cloud !== undefined && !Number.isNaN(cloud)
          ? { max_cloud_cover: cloud }
          : {}),
      });
      if (!current()) return;
      setSearchState({ status: "done", result });
    } catch (error) {
      if (!current()) return;
      setSearchState({ status: "error", message: errorMessage(error) });
    }
  }

  async function handlePreview(scene: SatelliteScene) {
    if (!resolved) return;

    // Each candidate keeps its own button, so a reader can click scene B while
    // scene A is still loading - which is reasonable, and used to be wrong.
    // Both requests would run, and whichever RESPONSE arrived last won, so a
    // slow first click could silently replace the scene the reader chose
    // second. The ticket makes the last CLICK win instead: only the newest
    // request may commit, and the previous one is aborted rather than left to
    // finish into a result nobody is waiting for.
    const activeRun = beginManualRun();
    onAoi?.({ ...resolved.bbox, scene_id: scene.id, imagery_requested: true });
    const ticket = ++previewTicketRef.current;
    previewAbortRef.current?.abort();
    const controller = new AbortController();
    previewAbortRef.current = controller;
    const current = () => activeRun() && previewTicketRef.current === ticket;

    setImageryState({ status: "loading", sceneId: scene.id });
    // Drop the old overlay before the new one is in flight, so the map never
    // shows a scene the user has already replaced.
    onImagery?.(null);
    try {
      const result = await fetchSceneImagery(
        { scene_id: scene.id, bbox: resolved.bbox },
        controller.signal,
      );
      if (!current()) return;
      setImageryState({ status: "done", sceneId: scene.id, result });
      onImagery?.(result);
      // The scene actually rendered, with its STAC record - everything the
      // evidence panel needs from a manual retrieval.
      onEvidence?.({ scene, imagery: result, measurements: [] });
    } catch (error) {
      // An abort is this code superseding itself, not a failure to report.
      if (!current() || controller.signal.aborted) return;
      setImageryState({
        status: "error",
        sceneId: scene.id,
        message: errorMessage(error),
      });
      onImagery?.(null);
    }
  }

  // The configuration this panel currently describes. Read straight off its own
  // controls - it reports what a run WOULD use, until a real run reports what
  // it did use.
  const manualContext: RunContext = {
    location: resolved?.display_name ?? null,
    centre: resolved?.center ?? null,
    window:
      temporalMode === "compare" && baselineStart && targetStart
        ? `${baselineStart} → ${targetEnd || targetStart}`
        : obsDate ||
          (startDate && endDate ? `${startDate} → ${endDate}` : null),
    cloudRule: maxCloud === "" ? null : `Cloud cover ≤ ${maxCloud}%`,
    modalities: [
      ...(opticalOn ? (["sentinel-2-optical"] as const) : []),
      ...(sarOn ? (["sentinel-1-sar"] as const) : []),
    ],
    task,
    ndwi: includeNdwi || includeTemporalNdwi,
    scenes: searchState.status === "done" ? searchState.result.scenes : [],
    selectedSceneId:
      executeState.status === "done"
        ? (executeState.result.windows.find(
            (executed) => executed.selected_scene_id !== null,
          )?.selected_scene_id ?? null)
        : null,
  };

  return (
    <section className="panel" aria-labelledby="query-heading">
      <h2 id="query-heading">Query configuration</h2>

      <ConfigSummary context={runContext ?? manualContext} />

      {/* The controls that produce that configuration. Below the read-out on
          purpose: the workspace is an instrument first and a form second. */}
      {/* The controls that produce the configuration above. Collapsed by
          default: the read-out already states what a run will use, so the
          form is the secondary path and should not repeat it at full
          height. A native <details> keeps every control mounted and
          keyboard-reachable rather than unmounting them. */}
      <details className="config-form">
        <summary>
          <span>Configure</span>
          <span className="config-form-hint">edit parameters</span>
        </summary>
        <div className="config-form-body">

      {/* Named bands, so the form reads as a configuration workspace rather
          than one long column of inputs. Headings only - no markup was
          restructured and no control moved. */}
      <h3>Location</h3>

      <form onSubmit={handleResolve} className="query-form">
        <label htmlFor="place-input">Place name</label>
        <input
          id="place-input"
          name="place"
          value={place}
          autoComplete="off"
          placeholder="Any city, district, landmark or lat, lon"
          onChange={(event) => setPlace(event.target.value)}
        />
        <button
          type="submit"
          disabled={resolveState.status === "loading" || place.trim() === ""}
        >
          {resolveState.status === "loading" ? "Resolving…" : "Resolve location"}
        </button>
      </form>

      {resolveState.status === "error" && (
        <p className="result-error" role="alert">
          {resolveState.message}
        </p>
      )}

      <form onSubmit={handleBuildPlan} className="query-form plan-form">
        <h3>Date / time window</h3>

        <fieldset className="temporal-mode">
          <legend>Temporal mode</legend>
          <label>
            <input
              type="radio"
              name="temporal_mode"
              value="single"
              checked={temporalMode === "single"}
              onChange={() => setTemporalMode("single")}
            />
            Single date
          </label>
          <label>
            <input
              type="radio"
              name="temporal_mode"
              value="compare"
              checked={temporalMode === "compare"}
              onChange={() => setTemporalMode("compare")}
            />
            Compare dates
          </label>
        </fieldset>

        {temporalMode === "single" ? (
          <div>
            <label htmlFor="obs-date">Observation date</label>
            <input
              id="obs-date"
              name="obs_date"
              type="date"
              value={obsDate}
              onChange={(event) => setObsDate(event.target.value)}
            />
          </div>
        ) : (
          <div className="field-row">
            <div>
              <label htmlFor="baseline-start">Baseline start</label>
              <input
                id="baseline-start"
                type="date"
                value={baselineStart}
                onChange={(event) => setBaselineStart(event.target.value)}
              />
            </div>
            <div>
              <label htmlFor="baseline-end">Baseline end</label>
              <input
                id="baseline-end"
                type="date"
                value={baselineEnd}
                onChange={(event) => setBaselineEnd(event.target.value)}
              />
            </div>
            <div>
              <label htmlFor="target-start">Target start</label>
              <input
                id="target-start"
                type="date"
                value={targetStart}
                onChange={(event) => setTargetStart(event.target.value)}
              />
            </div>
            <div>
              <label htmlFor="target-end">Target end</label>
              <input
                id="target-end"
                type="date"
                value={targetEnd}
                onChange={(event) => setTargetEnd(event.target.value)}
              />
            </div>
          </div>
        )}

        <h3>Satellite / sensor</h3>

        <fieldset className="modalities">
          <legend>Modalities</legend>
          <label>
            <input
              type="checkbox"
              name="sentinel-2-optical"
              checked={opticalOn}
              onChange={(event) => setOpticalOn(event.target.checked)}
            />
            Sentinel-2 Optical
          </label>
          <label>
            <input
              type="checkbox"
              name="sentinel-1-sar"
              checked={sarOn}
              onChange={(event) => setSarOn(event.target.checked)}
            />
            Sentinel-1 SAR
          </label>
        </fieldset>
        <h3>Analysis type</h3>


        <div>
          <label htmlFor="task-select">Task</label>
          <select
            id="task-select"
            name="task"
            value={task}
            onChange={(event) => setTask(event.target.value as QueryTask)}
          >
            {TASK_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>

        {modalities.length === 0 && (
          <p className="result-error">Select at least one modality.</p>
        )}
        {compareRangesInvalid && (
          <p className="result-error">
            Each window&apos;s start must be on or before its end.
          </p>
        )}

        <button type="submit" disabled={!canBuildPlan}>
          {planState.status === "loading" ? "Building…" : "Build Query Plan"}
        </button>

        <label className="include-imagery">
          <input
            type="checkbox"
            name="include_imagery"
            checked={includeImagery}
            onChange={(event) => setIncludeImagery(event.target.checked)}
          />
          Include bounded imagery preview
        </label>
        <label className="include-ndwi">
          <input
            type="checkbox"
            name="include_ndwi"
            checked={includeNdwi}
            onChange={(event) => {
              setIncludeNdwi(event.target.checked);
              invalidateAnalysis();
            }}
          />
          Compute NDWI index statistics (Sentinel-2)
        </label>
        <fieldset className="index-picker">
          <legend>Spectral indices</legend>
          {INDEX_CHOICES.map((choice) => (
            <label key={choice.key}>
              <input
                type="checkbox"
                name={`index_${choice.key}`}
                checked={indices.includes(choice.key)}
                onChange={(event) => {
                  setIndices((current) =>
                    event.target.checked
                      ? [...current, choice.key]
                      : current.filter((key) => key !== choice.key),
                  );
                  invalidateAnalysis();
                }}
              />
              <span className="index-name">{choice.label}</span>
              <span className="index-meaning">{choice.meaning}</span>
            </label>
          ))}
        </fieldset>
        <label className="include-temporal-ndwi">
          <input
            type="checkbox"
            name="include_temporal_ndwi"
            checked={includeTemporalNdwi}
            onChange={(event) => {
              setIncludeTemporalNdwi(event.target.checked);
              invalidateAnalysis();
            }}
          />
          Compute temporal NDWI statistics (two Sentinel-2 dates)
        </label>
        <h3>Execution</h3>
        <button type="button" onClick={handleExecute} disabled={!canExecute}>
          {executeState.status === "loading" ? "Running…" : "Run full query"}
        </button>
      </form>

      <h3>Parse from text</h3>
      <p className="hint">
        Optional. Parsing pre-fills the fields above for you to review.
      </p>
      <form onSubmit={handleParse} className="query-form nl-form">
        <label htmlFor="nl-input">Natural Language Request</label>
        <textarea
          id="nl-input"
          name="nl_prompt"
          rows={3}
          value={nlText}
          placeholder="e.g. Show optical imagery of Lake Victoria this summer"
          onChange={(event) => setNlText(event.target.value)}
        />
        <button
          type="submit"
          disabled={parseState.status === "loading" || nlText.trim() === ""}
        >
          {parseState.status === "loading" ? "Parsing…" : "Parse Request"}
        </button>
      </form>

      {parseState.status === "error" && (
        <p className="result-error" role="alert">
          {parseState.message}
        </p>
      )}

      {parseState.status === "done" && (
        <p className="hint" role="status">
          Parsed intent: {parseState.result.temporal_mode} ·{" "}
          {parseState.result.modalities.join(", ")} · {parseState.result.task}.
          The Query Plan form below is pre-filled — review or edit it, then click
          Build Query Plan.
          {parseState.result.temporal_mode === "timeseries" &&
            " (Time-series windows collapsed to the first window in the manual form.)"}
        </p>
      )}

      {planState.status === "error" && (
        <p className="result-error" role="alert">
          {planState.message}
        </p>
      )}

      {planState.status === "done" && <PlanView plan={planState.result} />}

      {executeState.status === "error" && (
        <p className="result-error" role="alert">
          {executeState.message}
        </p>
      )}

      {executeState.status === "done" && (
        <ExecutionView result={executeState.result} />
      )}

      {analyzeState.status === "error" && (
        <p className="result-error" role="alert">
          Analysis failed: {analyzeState.message}
        </p>
      )}

      {analyzeState.status === "done" && (
        <AnalysisView result={analyzeState.result} />
      )}

      {resolved && (
        <>
          <dl className="result">
            <div>
              <dt>Center</dt>
              <dd>
                {resolved.center.lat.toFixed(5)},{" "}
                {resolved.center.lon.toFixed(5)}
              </dd>
            </div>
            <div>
              <dt>Bounding box</dt>
              <dd>{formatBbox(resolved.bbox)}</dd>
            </div>
          </dl>

          <form onSubmit={handleSearch} className="query-form scene-search-form">
            <div className="field-row">
              <div>
                <label htmlFor="start-date">Start date</label>
                <input
                  id="start-date"
                  name="start_date"
                  type="date"
                  value={startDate}
                  onChange={(event) => setStartDate(event.target.value)}
                />
              </div>
              <div>
                <label htmlFor="end-date">End date</label>
                <input
                  id="end-date"
                  name="end_date"
                  type="date"
                  value={endDate}
                  onChange={(event) => setEndDate(event.target.value)}
                />
              </div>
              <div>
                <label htmlFor="max-cloud">Max cloud %</label>
                <input
                  id="max-cloud"
                  name="max_cloud_cover"
                  type="number"
                  min={0}
                  max={100}
                  placeholder="any"
                  value={maxCloud}
                  onChange={(event) => setMaxCloud(event.target.value)}
                />
              </div>
            </div>

            {datesInvalid && (
              <p className="result-error">Start date must be on or before end date.</p>
            )}

            <button type="submit" disabled={!canSearch}>
              {searchState.status === "loading"
                ? "Searching…"
                : "Search Sentinel-2 scenes"}
            </button>
          </form>

          {searchState.status === "error" && (
            <p className="result-error" role="alert">
              {searchState.message}
            </p>
          )}

          {searchState.status === "done" && (
            <SceneResults
              result={searchState.result}
              imageryState={imageryState}
              onPreview={handlePreview}
            />
          )}
        </>
      )}
        </div>
      </details>
    </section>
  );
}

function PlanView({ plan }: { plan: ResolvedQueryPlan }) {
  const { intent, bbox } = plan;
  const windows = intent.time_windows;
  return (
    <dl className="result plan-result">
      <div>
        <dt>Location</dt>
        <dd>{intent.location_query}</dd>
      </div>
      <div>
        <dt>Temporal mode</dt>
        <dd>{intent.temporal_mode}</dd>
      </div>
      <div>
        <dt>Time windows</dt>
        <dd>
          {Array.isArray(windows) ? (
            <ul>
              {windows.map((w, index) => (
                <li key={index}>
                  {w.start_date} → {w.end_date}
                </li>
              ))}
            </ul>
          ) : (
            <ul>
              <li>
                baseline: {windows.baseline.start_date} →{" "}
                {windows.baseline.end_date}
              </li>
              <li>
                target: {windows.target.start_date} → {windows.target.end_date}
              </li>
            </ul>
          )}
        </dd>
      </div>
      <div>
        <dt>Modalities</dt>
        <dd>{intent.modalities.join(", ")}</dd>
      </div>
      <div>
        <dt>Task</dt>
        <dd>{intent.task}</dd>
      </div>
      <div>
        <dt>Resolved bounding box</dt>
        <dd>{formatBbox(bbox)}</dd>
      </div>
    </dl>
  );
}

function ExecutionView({ result }: { result: QueryExecutionResult }) {
  return (
    <div className="result execution-result">
      <p className="hint" role="status">
        Executed: {result.executed_modalities.join(", ") || "none"} · source:{" "}
        {result.catalog}
      </p>
      {result.skipped_modalities.map((skipped) => (
        <p key={skipped.modality} className="hint hint-limitation">
          Skipped {skipped.modality}: {skipped.reason}
        </p>
      ))}
      {result.windows.length === 0 ? (
        <p className="hint" role="status">
          No windows executed for the requested modalities.
        </p>
      ) : (
        <ul className="execution-windows">
          {result.windows.map((win) => (
            <ExecutionWindowView
              key={`${win.modality}:${win.label}`}
              win={win}
            />
          ))}
        </ul>
      )}
    </div>
  );
}

function ExecutionWindowView({ win }: { win: ExecutedWindow }) {
  return (
    <li className="execution-window">
      <h4>
        {win.modality} · {win.label}
      </h4>
      <dl>
        <div>
          <dt>Window</dt>
          <dd>
            {win.time_range.start_date} → {win.time_range.end_date}
          </dd>
        </div>
        <div>
          <dt>Scenes found</dt>
          <dd>{win.scene_count}</dd>
        </div>
        <div>
          <dt>Selected scene</dt>
          <dd>{win.selected_scene_id ?? "— none —"}</dd>
        </div>
      </dl>
            {win.imagery_error && <ImageryErrorNotice raw={win.imagery_error} />}
      {win.imagery && (
        <figure className="scene-image">
          <img
            src={`data:${win.imagery.media_type};base64,${win.imagery.image_base64}`}
            alt={`Bounded RGB window for scene ${win.imagery.scene_id}`}
            width={win.imagery.width}
            height={win.imagery.height}
          />
        </figure>
      )}
    </li>
  );
}

/**
 * Renders the analysis boundary result: status, answer, per-window traceability
 * and warnings. Text only - no maps, overlays, detections or charts.
 */
/** How each operator reads, so the comparison is unambiguous on screen. */
const COMPARISON_SYMBOLS: Record<NdwiComparison, string> = {
  gt: ">",
  gte: "≥",
  lt: "<",
  lte: "≤",
};

/**
 * The threshold count, rendered from the structured measurement only.
 *
 * Every figure here is the backend's deterministic count over real pixels; the
 * answer prose is never parsed for numbers. The denominator is stated
 * explicitly - "of valid pixels" - because it is not the raster's pixel count,
 * and a reader who assumed otherwise would misread the percentage.
 */
function SpatialMeasurementView({
  measurement,
}: {
  measurement: SpatialMeasurement;
}) {
  const acquired = measurement.acquired_at?.slice(0, 10);
  return (
    <div className="analysis-measurement">
      <p className="measurement-headline">
        NDWI {COMPARISON_SYMBOLS[measurement.operator]}{" "}
        {measurement.threshold.toFixed(2)}
      </p>
      <p className="measurement-value">
        {measurement.percentage.toFixed(2)}% of valid pixels
      </p>
      <p className="hint">
        {measurement.matching_pixel_count.toLocaleString("en-US")} /{" "}
        {measurement.valid_pixel_count.toLocaleString("en-US")} pixels
      </p>
      <p className="hint">
        {measurement.scene_id}
        {acquired ? ` · ${acquired}` : ""}
        {measurement.crs ? ` · ${measurement.crs}` : ""}
      </p>
    </div>
  );
}


/** A signed change, so a rise and a fall are never confused. */
function signed(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;
}

/**
 * The paired-pixel change, rendered only when the backend produced one.
 *
 * Absent means the two observations were not on the same grid and nothing was
 * subtracted - the reason is in the comparison's warnings. It is deliberately
 * called NDWI change, not water gained or lost: the index moved, and this
 * system has not classified water.
 */
function TemporalChangeView({ change }: { change: NdwiTemporalChange }) {
  const earlier = change.first_acquired_at?.slice(0, 10);
  const later = change.second_acquired_at?.slice(0, 10);
  return (
    <div className="analysis-measurement">
      <p className="measurement-headline">NDWI Change (later − earlier)</p>
      <p className="hint">
        Earlier: {earlier ?? "date unknown"} · {change.first_scene_id}
      </p>
      <p className="hint">
        Later: {later ?? "date unknown"} · {change.second_scene_id}
      </p>
      <p className="measurement-value">
        Mean change: {signed(change.change_mean)}
      </p>
      <p className="hint">
        Min: {signed(change.change_min)} · Max: {signed(change.change_max)}
      </p>
      <p className="hint">
        Paired pixels:{" "}
        {change.paired_valid_pixel_count.toLocaleString("en-US")} · {change.crs}
      </p>
    </div>
  );
}


function AnalysisView({ result }: { result: AnalysisResult }) {
  return (
    <div className="result analysis-result">
      <h3>Analysis</h3>
      <p className="hint" role="status">
        {result.task} · {result.status}
      </p>
      <p className="analysis-answer">{result.answer}</p>

      {result.windows_considered.length > 0 && (
        <ul className="analysis-windows">
          {result.windows_considered.map((ref) => (
            <li key={`${ref.modality}:${ref.label}`}>
              {ref.modality} · {ref.label} ({ref.time_range.start_date} →{" "}
              {ref.time_range.end_date}) · {ref.selected_scene_id ?? "— none —"}
            </li>
          ))}
        </ul>
      )}

      {result.spatial_measurement && (
        <SpatialMeasurementView measurement={result.spatial_measurement} />
      )}

      {result.measurements.length > 0 && (
        <dl className="analysis-measurements">
          {result.measurements.map((measurement) => (
            <div key={measurement.name}>
              <dt>{measurement.name}</dt>
              <dd>
                {formatMeasurement(measurement.value, measurement.unit)}{" "}
                {measurement.unit}
              </dd>
            </div>
          ))}
        </dl>
      )}

      {result.temporal_comparison?.change && (
        <TemporalChangeView change={result.temporal_comparison.change} />
      )}

      {result.temporal_comparison && (
        <TemporalComparisonView comparison={result.temporal_comparison} />
      )}

      {result.warnings.map((warning) => (
        <p key={warning} className="hint hint-limitation">
          Warning: {warning}
        </p>
      ))}
    </div>
  );
}

/**
 * Temporal NDWI Statistics: two Sentinel-2 observations, each indexed on its
 * own pixels, shown side by side. The Mean NDWI Difference is a difference
 * between two aggregate statistics - no pixels were compared with one another,
 * nothing was co-registered, and the backend suppresses the value entirely when
 * that framing would mislead. Text only: no map, overlay or mask.
 */
function TemporalComparisonView({
  comparison,
}: {
  comparison: TemporalIndexComparison;
}) {
  const { first, second, compatibility, differences, warnings } = comparison;

  return (
    <div className="temporal-comparison">
      <h4>Temporal NDWI Statistics</h4>

      <ul className="temporal-observations">
        {[first, second].map((observation) => (
          <li key={`${observation.window_label}:${observation.scene_id}`}>
            <h5>{observation.window_label}</h5>
            <dl>
              <div>
                <dt>Scene</dt>
                <dd>{observation.scene_id}</dd>
              </div>
              <div>
                <dt>Acquired</dt>
                <dd>{observation.acquired_at ?? "— unknown —"}</dd>
              </div>
              <div>
                <dt>Cloud cover</dt>
                <dd>
                  {observation.cloud_cover === null
                    ? "— unknown —"
                    : `${formatPercent(observation.cloud_cover)}%`}
                </dd>
              </div>
              {observation.measurements.map((measurement) => (
                <div key={measurement.name}>
                  <dt>{measurement.name}</dt>
                  <dd>
                    {formatMeasurement(measurement.value, measurement.unit)}{" "}
                    {measurement.unit}
                  </dd>
                </div>
              ))}
            </dl>
          </li>
        ))}
      </ul>

      {differences.length > 0 && (
        <dl className="temporal-differences">
          {differences.map((difference) => (
            <div key={difference.name}>
              <dt>Mean NDWI Difference</dt>
              <dd>
                {formatMeasurement(difference.value, difference.unit)}{" "}
                {difference.unit}
              </dd>
            </div>
          ))}
        </dl>
      )}

      <p className="hint">
        Compatibility: {compatibility.co_registration_status} · footprint
        overlap {compatibility.bbox_overlap} · CRS {compatibility.crs_match} ·
        resolution {compatibility.resolution_match}
        {compatibility.temporal_separation_days !== null &&
          ` · ${formatDays(compatibility.temporal_separation_days)} days apart`}
      </p>

      {[...warnings, ...compatibility.limitations].map((note) => (
        <p key={note} className="hint hint-limitation">
          Note: {note}
        </p>
      ))}
    </div>
  );
}

function SceneResults({
  result,
  imageryState,
  onPreview,
}: {
  result: SceneSearchResponse;
  imageryState: ImageryState;
  onPreview: (scene: SatelliteScene) => void;
}) {
  if (result.scene_count === 0) {
    return (
      <p className="hint" role="status">
        No Sentinel-2 scenes found for this area and date range.
      </p>
    );
  }

  return (
    <div className="scene-results">
      <p className="hint" role="status">
        {result.scene_count} scene{result.scene_count === 1 ? "" : "s"} · source:{" "}
        {result.catalog}
      </p>
      <ul className="scene-list">
        {result.scenes.map((scene) => (
          <li key={scene.id} className="scene">
            <code className="scene-id">{scene.id}</code>
            <dl>
              <div>
                <dt>Acquired</dt>
                <dd>{scene.datetime ?? "—"}</dd>
              </div>
              <div>
                <dt>Cloud cover</dt>
                <dd>
                  {scene.cloud_cover === null
                    ? "—"
                    : `${scene.cloud_cover.toFixed(1)}%`}
                </dd>
              </div>
              <div>
                <dt>Platform</dt>
                <dd>{scene.platform ?? "—"}</dd>
              </div>
              <div>
                <dt>Processing level</dt>
                <dd>{scene.processing_level ?? "—"}</dd>
              </div>
              <div>
                <dt>Bounding box</dt>
                <dd>{scene.bbox ? formatBbox(scene.bbox) : "—"}</dd>
              </div>
              <div>
                <dt>Thumbnail URL</dt>
                <dd>
                  {scene.thumbnail_url ? (
                    <a
                      href={scene.thumbnail_url}
                      target="_blank"
                      rel="noreferrer"
                    >
                      {scene.thumbnail_url}
                    </a>
                  ) : (
                    "—"
                  )}
                </dd>
              </div>
            </dl>

            <button
              type="button"
              className="preview-button"
              onClick={() => onPreview(scene)}
              disabled={
                imageryState.status === "loading" &&
                imageryState.sceneId === scene.id
              }
            >
              {imageryState.status === "loading" &&
              imageryState.sceneId === scene.id
                ? "Loading image…"
                : "Load image"}
            </button>

            <SceneImage sceneId={scene.id} imageryState={imageryState} />
          </li>
        ))}
      </ul>
    </div>
  );
}

function SceneImage({
  sceneId,
  imageryState,
}: {
  sceneId: string;
  imageryState: ImageryState;
}) {
  if (imageryState.status === "idle" || imageryState.sceneId !== sceneId) {
    return null;
  }
  if (imageryState.status === "loading") {
    return (
      <p className="hint" role="status">
        Retrieving bounded RGB window…
      </p>
    );
  }
  if (imageryState.status === "error") {
    return (
      <p className="result-error" role="alert">
        {imageryState.message}
      </p>
    );
  }

  const img = imageryState.result;
  return (
    <figure className="scene-image">
      <img
        src={`data:${img.media_type};base64,${img.image_base64}`}
        alt={`Bounded RGB window for scene ${img.scene_id}`}
        width={img.width}
        height={img.height}
      />
      <figcaption className="scene-image-meta">
        {img.asset} · {img.width}×{img.height}px · {img.crs ?? "unknown CRS"} ·{" "}
        {img.resolution ? `${img.resolution} m/px native` : "resolution n/a"} ·
        window {img.window.width}×{img.window.height} of {img.source_shape[1]}×
        {img.source_shape[0]} · norm: {img.normalization}
      </figcaption>
    </figure>
  );
}
