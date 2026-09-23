import { apiRequest } from "./client";
import { asAgentResult } from "./validate";
import type { AgentQuestionRequest, AgentResult, AiProvider } from "./types";

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
export async function askAgent(
  question: string,
  options: {
    provider?: AiProvider | null;
    model?: string | null;
    signal?: AbortSignal;
  } = {},
): Promise<AgentResult> {
  const { provider = null, model = null, signal } = options;
  // `provider` and `model` select the inference backend for this run only.
  // Each is omitted when unset, so a default request stays byte-identical to
  // the previous contract. Neither ever carries a credential - keys live
  // server-side and no key is exposed to the browser.
  const body: AgentQuestionRequest = { question };
  if (provider !== null) body.provider = provider;
  if (model !== null) body.model = model;

  // Every panel reads `trace.steps` and `evidence.items` directly, so a
  // response that lacks them is refused here rather than failing later inside a
  // component with no idea where the bad value came from.
  return asAgentResult(
    await apiRequest<unknown>("/api/v1/query/agent", {
      method: "POST",
      body,
      signal,
    }),
  );
}
