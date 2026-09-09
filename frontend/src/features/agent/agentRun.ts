import { type FormEvent, useRef, useState } from "react";

import { askAgent } from "../../api/agent";
import { ApiError } from "../../api/client";
import type {
  AgentResult,
  ImageryResponse,
  NdwiOverlay,
  QueryExecutionResult,
} from "../../api/types";
import type { AiProvider } from "../../api/types";
import type { MapAoi } from "../map/footprint";
import { imageryRequested, shownWindow } from "./derive";

export type AskState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "done"; result: AgentResult; elapsedMs: number };

function errorMessage(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  return error instanceof Error ? error.message : "Unexpected error";
}

/** The scene preview the agent's own execution already retrieved. */
export function executedImagery(
  execution: QueryExecutionResult | null,
): ImageryResponse | null {
  return shownWindow(execution)?.imagery ?? null;
}

export interface AgentRunHandlers {
  /** Retire other workspace runs before this request starts. */
  onStart?: () => void;
  /**
   * Which inference backend answers this run's visual step. Read at request
   * time, so changing it never disturbs the scene, evidence, map or query
   * already on screen - only the next run's provider.
   */
  provider?: AiProvider | null;
  /** Which model that provider uses. `null` uses the server's default. */
  model?: string | null;
  onImagery?: (imagery: ImageryResponse | null) => void;
  onNdwi?: (overlay: NdwiOverlay | null) => void;
  onChange?: (overlay: NdwiOverlay | null) => void;
  onAoi?: (aoi: MapAoi | null) => void;
}

export interface AgentRun {
  question: string;
  setQuestion: (question: string) => void;
  askState: AskState;
  result: AgentResult | null;
  busy: boolean;
  canAsk: boolean;
  handleAsk: (event: FormEvent) => Promise<void>;
  clear: () => void;
}

/**
 * One agent run: the question, the request, and the result.
 *
 * Extracted from the panel so the workspace can place the query, the pipeline,
 * the evidence, the answer and the observation in three different columns while
 * they all read from a single piece of state. It sends the question, hands the
 * result upward, and interprets nothing: no tool is chosen here, no index is
 * computed, no answer is validated, and no evidence is re-derived.
 */
export function useAgentRun({
  onStart,
  provider = null,
  model = null,
  onImagery,
  onNdwi,
  onChange,
  onAoi,
}: AgentRunHandlers = {}): AgentRun {
  const [question, setQuestion] = useState("");
  const [askState, setAskState] = useState<AskState>({ status: "idle" });

  /**
   * Which run owns the screen.
   *
   * The submit button is disabled while a run is in flight, but that alone
   * does NOT make a superseded response impossible: `clear()` returns the
   * state to idle, which re-enables submit while the previous request is still
   * open. The older response then lands last and overwrites the newer run's
   * answer, evidence, imagery and map - a result rendered under a question
   * that is no longer on screen.
   *
   * So every run takes a ticket, and only the newest ticket may write. The
   * superseded request is also aborted, so it stops occupying a connection and
   * a provider slot rather than merely having its answer discarded.
   */
  const ticketRef = useRef(0);
  const abortRef = useRef<AbortController | null>(null);

  const busy = askState.status === "loading";
  const canAsk = question.trim() !== "" && !busy;
  const result = askState.status === "done" ? askState.result : null;

  function reset() {
    onImagery?.(null);
    onNdwi?.(null);
    onChange?.(null);
    onAoi?.(null);
  }

  async function handleAsk(event: FormEvent) {
    event.preventDefault();
    if (!canAsk) return;

    const ticket = ++ticketRef.current;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    onStart?.();
    setAskState({ status: "loading" });
    // A new question invalidates whatever is on the map. Cleared before the
    // request rather than after it, so a slow run never leaves the previous
    // scene on screen beside a newer question.
    reset();

    // Round-trip time, measured here. The backend reports no timing of its
    // own, so this is labelled as what it is - the request - and never split
    // into per-stage figures the server did not produce.
    const startedAt = Date.now();
    try {
      const answer = await askAgent(question.trim(), {
        provider,
        model,
        signal: controller.signal,
      });
      // Superseded: a newer run owns the screen. Drop this result entirely
      // rather than painting it over the newer one.
      if (ticket !== ticketRef.current) return;
      setAskState({
        status: "done",
        result: answer,
        elapsedMs: Date.now() - startedAt,
      });

      const { execution, analysis } = answer.evidence;
      onImagery?.(executedImagery(execution));
      onNdwi?.(analysis?.ndwi_overlay ?? null);
      onChange?.(analysis?.temporal_comparison?.change?.overlay ?? null);

      // The resolved extent, read straight from the plan the server validated.
      const bbox = execution?.plan.bbox ?? null;
      const window = shownWindow(execution);
      onAoi?.(
        bbox === null
          ? null
          : {
              ...bbox,
              scene_id: window?.selected_scene_id ?? null,
              // Whether a picture was ASKED for, read from the validated plan
              // rather than inferred from its absence. Without this the map
              // cannot tell a failed retrieval from one nobody requested, and
              // it reported every run as a failure.
              imagery_requested: imageryRequested(answer),
              imagery_error: window?.imagery_error ?? null,
            },
      );
    } catch (error) {
      // A superseded or cleared run reports nothing: its abort is an intended
      // outcome, not a fault the reader needs to see.
      if (ticket !== ticketRef.current) return;
      setAskState({ status: "error", message: errorMessage(error) });
    }
  }

  function clear() {
    // Retire the in-flight run before returning to idle. Without this, idle
    // re-enables submit while the old request is still open, and whichever
    // response lands last wins.
    ticketRef.current += 1;
    abortRef.current?.abort();
    abortRef.current = null;
    setQuestion("");
    setAskState({ status: "idle" });
    reset();
  }

  return {
    question,
    setQuestion,
    askState,
    result,
    busy,
    canAsk,
    handleAsk,
    clear,
  };
}
