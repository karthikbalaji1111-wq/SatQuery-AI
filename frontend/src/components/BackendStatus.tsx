import { useEffect, useState } from "react";

import { ApiError } from "../api/client";
import { getHealth, getReadiness } from "../api/health";
import type { HealthResponse, ReadinessResponse } from "../api/types";

type Status =
  | { state: "loading" }
  | {
      state: "ok";
      data: HealthResponse;
      /** `null` when readiness could not be determined - see below. */
      readiness: ReadinessResponse | null;
    }
  | { state: "error"; message: string };

/**
 * What the backend is, and whether it can actually work.
 *
 * This read `/health` alone and printed "Operational". `/health` answers only
 * "is the process alive?", so a deployment holding no credential for its
 * selected provider - or a local model that was never installed - displayed
 * "Operational" beside an AI path that could not answer a single query. The
 * one state an operator most needs to see was the one the header could not
 * show.
 *
 * Liveness still decides whether the backend is reachable at all; readiness
 * qualifies it. When readiness cannot be fetched the component says exactly
 * what it did before rather than inventing a verdict: an unknown capability is
 * not a failed one.
 */
export function BackendStatus() {
  const [status, setStatus] = useState<Status>({ state: "loading" });

  useEffect(() => {
    const controller = new AbortController();

    getHealth(controller.signal)
      .then(async (data) => {
        // Readiness is additive: a failure to obtain it must not turn a
        // reachable backend into an unreachable one.
        const readiness = await getReadiness(controller.signal).catch(() => null);
        if (controller.signal.aborted) return;
        setStatus({ state: "ok", data, readiness });
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        const message =
          error instanceof ApiError ? error.message : "Unknown error";
        setStatus({ state: "error", message });
      });

    return () => controller.abort();
  }, []);

  const unready =
    status.state === "ok"
      ? // An optional capability (the AI provider) is reported by the server
        // but never named as the reason the deployment is degraded.
        (status.readiness?.capabilities ?? []).filter(
          (c) => !c.ready && c.required !== false,
        )
      : [];
  const degraded = status.state === "ok" && status.readiness?.ready === false;

  return (
    <div className="backend-status" role="status">
      {status.state === "loading" && <span>Checking backend…</span>}
      {status.state === "ok" && !degraded && (
        // The detail moves to a tooltip: a header is for state at a glance, and
        // the build environment in particular should not be the widest element
        // on screen.
        <span
          className="ok"
          title={`${status.data.service} v${status.data.version} · ${status.data.environment}`}
        >
          Operational
        </span>
      )}
      {status.state === "ok" && degraded && (
        // Named, not merely coloured: "Degraded" alone would leave the reader
        // to guess, and the server already said which capability is missing.
        <span
          className="degraded"
          title={unready.map((c) => c.detail).join(" ")}
        >
          Degraded — {unready.map((c) => c.name.replace(/_/g, " ")).join(", ")}
        </span>
      )}
      {status.state === "error" && (
        <span className="error">Backend unreachable — {status.message}</span>
      )}
    </div>
  );
}
