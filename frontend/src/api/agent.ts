import { apiRequest } from "./client";
import type { AgentResult } from "./types";

/**
 * Ask the agent a free-form question.
 *
 * A language model chooses which of the existing deterministic analyses to run;
 * the server validates that choice against a closed tool set, executes it, and
 * validates the generated answer against the evidence before returning.
 *
 * Every agent outcome is a 200 - including `planner_unavailable`,
 * `synthesis_unavailable` and `answer_withheld`, which carry the deterministic
 * evidence with no answer. Only transport and genuine faults reject.
 */
export function askAgent(
  question: string,
  signal?: AbortSignal,
): Promise<AgentResult> {
  return apiRequest<AgentResult>("/api/v1/query/agent", {
    method: "POST",
    body: { question },
    signal,
  });
}
