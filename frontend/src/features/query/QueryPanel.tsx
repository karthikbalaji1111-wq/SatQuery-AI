import { type FormEvent, useState } from "react";

import { ApiError } from "../../api/client";
import { resolveLocation } from "../../api/geospatial";
import { searchScenes } from "../../api/satellite";
import type { GeoResolveResponse, SceneSearchResponse } from "../../api/types";

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

function errorMessage(error: unknown): string {
  return error instanceof ApiError ? error.message : "Unexpected error";
}

function formatBbox(bbox: {
  west: number;
  south: number;
  east: number;
  north: number;
}): string {
  return (
    `W ${bbox.west.toFixed(4)}, S ${bbox.south.toFixed(4)}, ` +
    `E ${bbox.east.toFixed(4)}, N ${bbox.north.toFixed(4)}`
  );
}

/**
 * Phase flow: place name -> geospatial resolve -> validated bbox ->
 * Sentinel-2 STAC scene discovery. No map, no imagery, no NL parsing.
 */
export function QueryPanel() {
  const [place, setPlace] = useState("");
  const [resolveState, setResolveState] = useState<ResolveState>({
    status: "idle",
  });

  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [maxCloud, setMaxCloud] = useState("");
  const [searchState, setSearchState] = useState<SearchState>({ status: "idle" });

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

  async function handleResolve(event: FormEvent) {
    event.preventDefault();
    const trimmed = place.trim();
    if (!trimmed) return;

    setResolveState({ status: "loading" });
    setSearchState({ status: "idle" });
    try {
      const result = await resolveLocation({ place: trimmed });
      setResolveState({ status: "done", result });
    } catch (error) {
      setResolveState({ status: "error", message: errorMessage(error) });
    }
  }

  async function handleSearch(event: FormEvent) {
    event.preventDefault();
    if (!resolved || !canSearch) return;

    const cloud = maxCloud.trim() === "" ? undefined : Number(maxCloud);
    setSearchState({ status: "loading" });
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

  return (
    <section className="panel" aria-labelledby="query-heading">
      <h2 id="query-heading">Ask</h2>
      <p className="hint">
        Place name → resolved bounding box → Sentinel-2 scene discovery
        (metadata only).
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
            <SceneResults result={searchState.result} />
          )}
        </>
      )}
    </section>
  );
}

function SceneResults({ result }: { result: SceneSearchResponse }) {
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
          </li>
        ))}
      </ul>
    </div>
  );
}
