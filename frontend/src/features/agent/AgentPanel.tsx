import { type FormEvent, useState } from "react";

import { askAgent } from "../../api/agent";
import { ApiError } from "../../api/client";
import type {
  AgentResult,
  AgentStatus,
  AgentToolName,
  AgentToolStep,
  AgentEvidence as Evidence,
  EvidenceItem,
} from "../../api/types";

type AskState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "done"; result: AgentResult };

/** Readable labels for the closed tool set. Nothing else is ever displayed. */
const TOOL_LABELS: Record<AgentToolName, string> = {
  execute_query: "Execute query",
  ndwi_statistics: "NDWI statistics",
  temporal_ndwi_statistics: "Temporal NDWI statistics",
};

/**
 * What to say when there is no answer. Each message names the stage that
 * stopped, and none of them stands in for the answer: an absent answer is
 * reported as absent, never softened into prose the backend did not produce.
 */
const STATUS_MESSAGES: Record<Exclude<AgentStatus, "ok">, string> = {
  planner_unavailable:
    "The request could not be planned, so nothing was run. No answer and no evidence were produced.",
  synthesis_unavailable:
    "The evidence below was collected, but an answer could not be generated for it.",
  answer_withheld:
    "An answer was generated but did not pass validation against the evidence, so it was withheld. The evidence below is unaffected.",
};

function errorMessage(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  return error instanceof Error ? error.message : "Unexpected error";
}

function toolLabel(tool: AgentToolName): string {
  return TOOL_LABELS[tool] ?? tool;
}

/**
 * The agentic entry point: one question, one request, one result.
 *
 * A deliberately thin surface. It sends the question, renders what the backend
 * established, and interprets nothing: no tool is chosen here, no index is
 * computed, no answer is validated, and no evidence is re-derived. Sections
 * appear in a fixed order - Plan, Tools selected, Execution, Evidence, Answer -
 * so the reader sees what was intended and what actually happened before they
 * see the prose.
 *
 * There is no reasoning view, and there is nothing to build one from: the
 * backend contract has no field for it.
 */
export function AgentPanel() {
  const [question, setQuestion] = useState("");
  const [askState, setAskState] = useState<AskState>({ status: "idle" });

  const busy = askState.status === "loading";
  const canAsk = question.trim() !== "" && !busy;

  async function handleAsk(event: FormEvent) {
    event.preventDefault();
    if (!canAsk) return;

    setAskState({ status: "loading" });
    try {
      const result = await askAgent(question.trim());
      setAskState({ status: "done", result });
    } catch (error) {
      setAskState({ status: "error", message: errorMessage(error) });
    }
  }

  return (
    <section className="panel" aria-labelledby="agent-heading">
      <h2 id="agent-heading">Ask the agent</h2>
      <p className="hint">
        Ask a question in plain language. The server chooses which of the
        existing deterministic analyses to run, executes them, and checks the
        answer against the evidence before showing it.
      </p>

      <form onSubmit={handleAsk} className="query-form agent-form">
        <label htmlFor="agent-question">Question</label>
        <textarea
          id="agent-question"
          name="question"
          rows={3}
          value={question}
          disabled={busy}
          placeholder="e.g. What is the NDWI of Chennai in January 2024?"
          onChange={(event) => setQuestion(event.target.value)}
        />
        <button type="submit" disabled={!canAsk}>
          {busy ? "Asking…" : "Ask"}
        </button>
      </form>

      {askState.status === "error" && (
        <p className="result-error" role="alert">
          {askState.message}
        </p>
      )}

      {askState.status === "done" && <AgentResultView result={askState.result} />}
    </section>
  );
}

/** Plan → Tools selected → Execution → Evidence → Answer, in that order. */
function AgentResultView({ result }: { result: AgentResult }) {
  const { trace, evidence } = result;

  return (
    <div className="result agent-result">
      <PlanView plan={trace.plan} />
      <ToolsView plan={trace.plan} />
      <ExecutionStepsView steps={trace.steps} />
      <EvidenceView evidence={evidence} />
      <AnswerView result={result} />
    </div>
  );
}

/** What the server was asked to do, after validation. */
function PlanView({ plan }: { plan: AgentResult["trace"]["plan"] }) {
  return (
    <div className="agent-section">
      <h3>Plan</h3>
      {plan === null ? (
        <p className="hint">No plan was produced.</p>
      ) : (
        <ol className="agent-plan">
          {plan.steps.map((step, index) => (
            <li key={`${step.tool}:${index}`}>
              {toolLabel(step.tool)}
              {step.tool === "execute_query" && (
                <span className="hint">
                  {" "}
                  · {step.intent.location_query} · {step.intent.temporal_mode} ·{" "}
                  {step.intent.modalities.join(", ")}
                  {step.include_imagery && " · with imagery"}
                  {step.max_cloud_cover !== null &&
                    ` · max cloud ${step.max_cloud_cover}%`}
                </span>
              )}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

/** The tools the plan actually names - never a fixed list. */
function ToolsView({ plan }: { plan: AgentResult["trace"]["plan"] }) {
  return (
    <div className="agent-section">
      <h3>Tools selected</h3>
      {plan === null || plan.steps.length === 0 ? (
        <p className="hint">No tools were selected.</p>
      ) : (
        <ul className="agent-tools">
          {plan.steps.map((step, index) => (
            <li key={`${step.tool}:${index}`}>{toolLabel(step.tool)}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

/** Observable per-step outcome. Not a reasoning trace. */
function ExecutionStepsView({ steps }: { steps: AgentToolStep[] }) {
  return (
    <div className="agent-section">
      <h3>Execution</h3>
      {steps.length === 0 ? (
        <p className="hint">Nothing was executed.</p>
      ) : (
        <ul className="agent-steps">
          {steps.map((step, index) => (
            <li key={`${step.parameters.tool}:${index}`}>
              {toolLabel(step.parameters.tool)} — {step.status}
              {step.error_message && (
                <span className="result-error"> {step.error_message}</span>
              )}
              {step.rejection_reason && (
                <span className="hint"> {step.rejection_reason}</span>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/** The deterministic findings, kept visually distinct from the prose. */
function EvidenceView({ evidence }: { evidence: Evidence }) {
  return (
    <div className="agent-section">
      <h3>Evidence</h3>
      {evidence.items.length === 0 ? (
        <p className="hint">No evidence was collected.</p>
      ) : (
        <dl className="agent-evidence">
          {evidence.items.map((item) => (
            <div key={item.id}>
              <dt>{item.id}</dt>
              <dd>{evidenceValue(item)}</dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
}

function evidenceValue(item: EvidenceItem): string {
  if (item.measurement) {
    return `${item.measurement.name} = ${item.measurement.value} ${item.measurement.unit}`;
  }
  return item.text ?? "";
}

/**
 * The generated prose - or an honest statement of its absence.
 *
 * When the backend withheld the answer, nothing is substituted for it. The
 * evidence above stands on its own; that is the point of returning it.
 */
function AnswerView({ result }: { result: AgentResult }) {
  return (
    <div className="agent-section">
      <h3>Answer</h3>
      {result.status === "ok" && result.answer !== null ? (
        <p className="agent-answer">{result.answer}</p>
      ) : (
        <p className="hint" role="status">
          {STATUS_MESSAGES[result.status as Exclude<AgentStatus, "ok">] ??
            "No answer was produced."}
        </p>
      )}
    </div>
  );
}
