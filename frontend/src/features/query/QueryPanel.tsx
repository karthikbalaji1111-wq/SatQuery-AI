import { type FormEvent, useState } from "react";

import { ApiError } from "../../api/client";
import { resolveLocation } from "../../api/geospatial";
import { buildQueryPlan } from "../../api/query";
import { fetchSceneImagery, searchScenes } from "../../api/satellite";
import type {
  BoundingBox,
  GeoResolveResponse,
  ImageryResponse,
  Modality,
  QueryTask,
  ResolvedQueryPlan,
  SatelliteScene,
  SatQueryIntent,
  SceneSearchResponse,
  TemporalComparison,
  TemporalMode,
} from "../../api/types";

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

function errorMessage(error: unknown): string {
  return error instanceof ApiError ? error.message : "Unexpected error";
}

function formatBbox(bbox: BoundingBox): string {
  return (
    `W ${bbox.west.toFixed(4)}, S ${bbox.south.toFixed(4)}, ` +
    `E ${bbox.east.toFixed(4)}, N ${bbox.north.toFixed(4)}`
  );
}

const TASK_OPTIONS: { value: QueryTask; label: string }[] = [
  { value: "visualize", label: "Visualize" },
  { value: "change_detection", label: "Change Detection" },
  { value: "object_identification", label: "Object Identification" },
];

/**
 * Phase flow: place -> structured intent -> query plan (intent + resolved
 * bbox); and, independently, place -> resolve -> Sentinel-2 discovery ->
 * bounded RGB imagery for a selected scene.
 * No map, no AI/VLM, no NL parsing, no spectral controls.
 */
export function QueryPanel() {
  const [place, setPlace] = useState("");
  const [resolveState, setResolveState] = useState<ResolveState>({
    status: "idle",
  });

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
  const canBuildPlan =
    place.trim() !== "" &&
    modalities.length > 0 &&
    temporalComplete &&
    !compareRangesInvalid &&
    planState.status !== "loading";

  async function handleResolve(event: FormEvent) {
    event.preventDefault();
    const trimmed = place.trim();
    if (!trimmed) return;

    setResolveState({ status: "loading" });
    setSearchState({ status: "idle" });
    setImageryState({ status: "idle" });
    try {
      const result = await resolveLocation({ place: trimmed });
      setResolveState({ status: "done", result });
    } catch (error) {
      setResolveState({ status: "error", message: errorMessage(error) });
    }
  }

  async function handleBuildPlan(event: FormEvent) {
    event.preventDefault();
    if (!canBuildPlan) return;

    const timeWindows: SatQueryIntent["time_windows"] =
      temporalMode === "single"
        ? [{ start_date: obsDate, end_date: obsDate }]
        : ({
            baseline: { start_date: baselineStart, end_date: baselineEnd },
            target: { start_date: targetStart, end_date: targetEnd },
          } satisfies TemporalComparison);

    const intent: SatQueryIntent = {
      location_query: place.trim(),
      temporal_mode: temporalMode,
      time_windows: timeWindows,
      modalities,
      task,
    };

    setPlanState({ status: "loading" });
    try {
      const result = await buildQueryPlan(intent);
      setPlanState({ status: "done", result });
    } catch (error) {
      setPlanState({ status: "error", message: errorMessage(error) });
    }
  }

  async function handleSearch(event: FormEvent) {
    event.preventDefault();
    if (!resolved || !canSearch) return;

    const cloud = maxCloud.trim() === "" ? undefined : Number(maxCloud);
    setSearchState({ status: "loading" });
    setImageryState({ status: "idle" });
    try {
      const result = await searchScenes({
        bbox: resolved.bbox,
        start_date: startDate,
        end_date: endDate,
        ...(cloud !== undefined && !Number.isNaN(cloud)
          ? { max_cloud_cover: cloud }
          : {}),
      });
      setSearchState({ status: "done", result });
    } catch (error) {
      setSearchState({ status: "error", message: errorMessage(error) });
    }
  }

  async function handlePreview(scene: SatelliteScene) {
    if (!resolved) return;
    setImageryState({ status: "loading", sceneId: scene.id });
    try {
      const result = await fetchSceneImagery({
        scene_id: scene.id,
        bbox: resolved.bbox,
      });
      setImageryState({ status: "done", sceneId: scene.id, result });
    } catch (error) {
      setImageryState({
        status: "error",
        sceneId: scene.id,
        message: errorMessage(error),
      });
    }
  }

  return (
    <section className="panel" aria-labelledby="query-heading">
      <h2 id="query-heading">Ask</h2>
      <p className="hint">
        Structured intent → resolved bounding box (query plan); and place →
        Sentinel-2 discovery → bounded RGB imagery.
      </p>

      <form onSubmit={handleResolve} className="query-form">
        <label htmlFor="place-input">Place name</label>
        <input
          id="place-input"
          name="place"
          value={place}
          autoComplete="off"
          placeholder="e.g. Chennai"
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
        <h3>Query plan</h3>

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
      </form>

      {planState.status === "error" && (
        <p className="result-error" role="alert">
          {planState.message}
        </p>
      )}

      {planState.status === "done" && <PlanView plan={planState.result} />}

      {resolved && (
        <>
          <dl className="result">
            {resolved.display_name && (
              <div>
                <dt>Match</dt>
                <dd>{resolved.display_name}</dd>
              </div>
            )}
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
