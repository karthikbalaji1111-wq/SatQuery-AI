import { useEffect, useState } from "react";

import { ApiError } from "../api/client";
import { getHealth } from "../api/health";
import type { HealthResponse } from "../api/types";

type Status =
  | { state: "loading" }
  | { state: "ok"; data: HealthResponse }
  | { state: "error"; message: string };

/** Live indicator that verifies the frontend can reach the backend `/health`. */
export function BackendStatus() {
  const [status, setStatus] = useState<Status>({ state: "loading" });

  useEffect(() => {
    const controller = new AbortController();

    getHealth(controller.signal)
      .then((data) => setStatus({ state: "ok", data }))
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        const message =
          error instanceof ApiError ? error.message : "Unknown error";
        setStatus({ state: "error", message });
      });

    return () => controller.abort();
  }, []);

  return (
    <div className="backend-status" role="status">
      {status.state === "loading" && <span>Checking backend…</span>}
      {status.state === "ok" && (
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
      {status.state === "error" && (
        <span className="error">Backend unreachable — {status.message}</span>
      )}
    </div>
  );
}
