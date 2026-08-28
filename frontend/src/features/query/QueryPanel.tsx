import { type FormEvent, useState } from "react";

import { ApiError } from "../../api/client";
import { resolveLocation } from "../../api/geospatial";
import type { GeoResolveResponse } from "../../api/types";

type State =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "done"; result: GeoResolveResponse }
  | { status: "error"; message: string };

/**
 * Geospatial grounding only: resolve a place name to coordinates + bounding box.
 * Query understanding, imagery, and the map arrive in later phases.
 */
export function QueryPanel() {
  const [place, setPlace] = useState("");
  const [state, setState] = useState<State>({ status: "idle" });

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const trimmed = place.trim();
    if (!trimmed) return;

    setState({ status: "loading" });
    try {
      const result = await resolveLocation({ place: trimmed });
      setState({ status: "done", result });
    } catch (error) {
      const message =
        error instanceof ApiError ? error.message : "Unexpected error";
      setState({ status: "error", message });
    }
  }

  return (
    <section className="panel" aria-labelledby="query-heading">
      <h2 id="query-heading">Ask</h2>
      <p className="hint">
        Geospatial grounding only — resolve a place name to coordinates.
      </p>

      <form onSubmit={handleSubmit} className="query-form">
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
          disabled={state.status === "loading" || place.trim() === ""}
        >
          {state.status === "loading" ? "Resolving…" : "Resolve location"}
        </button>
      </form>

      {state.status === "error" && (
        <p className="result-error" role="alert">
          {state.message}
        </p>
      )}

      {state.status === "done" && (
        <dl className="result">
          {state.result.display_name && (
            <div>
              <dt>Match</dt>
              <dd>{state.result.display_name}</dd>
            </div>
          )}
          <div>
            <dt>Center</dt>
            <dd>
              {state.result.center.lat.toFixed(5)},{" "}
              {state.result.center.lon.toFixed(5)}
            </dd>
          </div>
          <div>
            <dt>Bounding box</dt>
            <dd>
              W {state.result.bbox.west.toFixed(4)}, S{" "}
              {state.result.bbox.south.toFixed(4)}, E{" "}
              {state.result.bbox.east.toFixed(4)}, N{" "}
              {state.result.bbox.north.toFixed(4)}
            </dd>
          </div>
        </dl>
      )}
    </section>
  );
}
