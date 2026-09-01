import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AgentPanel } from "./AgentPanel";

afterEach(() => {
  vi.restoreAllMocks();
});

const QUESTION = "What is the NDWI of Chennai in January 2024?";

const INTENT = {
  location_query: "Chennai",
  temporal_mode: "single",
  time_windows: [{ start_date: "2024-01-01", end_date: "2024-01-31" }],
  modalities: ["sentinel-2-optical"],
  task: "visualize",
};

const EXECUTE_STEP = {
  tool: "execute_query",
  intent: INTENT,
  include_imagery: false,
  max_cloud_cover: null,
};

const EVIDENCE = {
  items: [
    {
      id: "ndwi.ndwi_mean",
      source: "ndwi",
      measurement: { name: "ndwi_mean", value: 0.2777, unit: "index" },
      text: null,
      produced_by: "analysis.engines.compute_ndwi_measurements",
    },
    {
      id: "compatibility.limitation.0",
      source: "compatibility",
      measurement: null,
      text: "Equal AOI coverage is NOT established.",
      produced_by: "query.compatibility.compute_compatibility",
    },
  ],
  execution: null,
  analysis: null,
};

function agentResult(overrides: Record<string, unknown> = {}) {
  return {
    status: "ok",
    answer: "The mean NDWI was 0.2777 index.",
    trace: {
      plan: { steps: [EXECUTE_STEP, { tool: "ndwi_statistics" }] },
      steps: [
        {
          status: "ok",
          parameters: EXECUTE_STEP,
          rejection_reason: null,
          error_message: null,
        },
        {
          status: "ok",
          parameters: { tool: "ndwi_statistics" },
          rejection_reason: null,
          error_message: null,
        },
      ],
      evidence_refs: ["ndwi.ndwi_mean"],
      answer_validation: {
        numeric_grounding: "pass",
        forbidden_terms: "pass",
        evidence_refs: "pass",
      },
    },
    evidence: EVIDENCE,
    ...overrides,
  };
}

/** Mirrors the fetch stub convention used by QueryPanel.test.tsx. */
function stubAgent(
  route:
    | { body: unknown; ok?: boolean; status?: number }
    | (() => Promise<Response>),
) {
  const fn = vi.fn().mockImplementation((url: string) => {
    if (!String(url).includes("/query/agent")) {
      return Promise.reject(new Error(`unexpected call: ${url}`));
    }
    if (typeof route === "function") return route();
    return Promise.resolve({
      ok: route.ok ?? true,
      status: route.status ?? 200,
      text: () => Promise.resolve(JSON.stringify(route.body)),
    } as Response);
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

function askQuestion(question: string = QUESTION) {
  fireEvent.change(screen.getByLabelText(/question/i), {
    target: { value: question },
  });
  fireEvent.click(screen.getByRole("button", { name: /^ask$/i }));
}

async function askAndWait(question: string = QUESTION) {
  askQuestion(question);
  await waitFor(() =>
    expect(screen.getByRole("heading", { name: "Answer" })).toBeInTheDocument(),
  );
}

// ===========================================================================
// A. The form
// ===========================================================================

describe("AgentPanel - the Ask form", () => {
  it("renders a question field and an Ask button", () => {
    render(<AgentPanel />);

    expect(screen.getByLabelText(/question/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^ask$/i })).toBeInTheDocument();
  });

  it("disables Ask until a question is entered", () => {
    render(<AgentPanel />);
    const button = screen.getByRole("button", { name: /^ask$/i });

    expect(button).toBeDisabled();

    fireEvent.change(screen.getByLabelText(/question/i), {
      target: { value: QUESTION },
    });
    expect(button).toBeEnabled();
  });

  it("rejects a whitespace-only question without calling the API", () => {
    const fetchMock = stubAgent({ body: agentResult() });
    render(<AgentPanel />);

    fireEvent.change(screen.getByLabelText(/question/i), {
      target: { value: "   " },
    });

    expect(screen.getByRole("button", { name: /^ask$/i })).toBeDisabled();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("posts the question to /api/v1/query/agent", async () => {
    const fetchMock = stubAgent({ body: agentResult() });
    render(<AgentPanel />);

    await askAndWait();

    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toContain("/api/v1/query/agent");
    expect((init as RequestInit).method).toBe("POST");
    expect(JSON.parse((init as RequestInit).body as string)).toEqual({
      question: QUESTION,
    });
  });

  it("sends only the question - no imagery flag or provider config", async () => {
    const fetchMock = stubAgent({ body: agentResult() });
    render(<AgentPanel />);

    await askAndWait();

    const body = JSON.parse(
      (fetchMock.mock.calls[0][1] as RequestInit).body as string,
    );
    expect(Object.keys(body)).toEqual(["question"]);
    for (const banned of ["include_imagery", "api_key", "apiKey", "model"]) {
      expect(banned in body).toBe(false);
    }
  });

  it("disables the controls while the request is in flight", async () => {
    let release: (value: Response) => void = () => {};
    stubAgent(
      () =>
        new Promise<Response>((resolve) => {
          release = resolve;
        }),
    );
    render(<AgentPanel />);

    askQuestion();

    await waitFor(() =>
      expect(screen.getByRole("button", { name: /asking/i })).toBeDisabled(),
    );
    expect(screen.getByLabelText(/question/i)).toBeDisabled();

    release({
      ok: true,
      status: 200,
      text: () => Promise.resolve(JSON.stringify(agentResult())),
    } as Response);

    await waitFor(() =>
      expect(screen.getByRole("button", { name: /^ask$/i })).toBeEnabled(),
    );
  });
});

// ===========================================================================
// B. Successful result rendering, in the required order
// ===========================================================================

describe("AgentPanel - successful result", () => {
  it("renders Plan, Tools selected, Execution, Evidence and Answer in order", async () => {
    stubAgent({ body: agentResult() });
    render(<AgentPanel />);

    await askAndWait();

    const headings = screen
      .getAllByRole("heading")
      .map((node) => node.textContent);
    const order = ["Plan", "Tools selected", "Execution", "Evidence", "Answer"];
    const positions = order.map((label) => headings.indexOf(label));

    expect(positions.every((index) => index >= 0)).toBe(true);
    expect([...positions]).toEqual([...positions].sort((a, b) => a - b));
  });

  it("renders the answer text returned by the backend", async () => {
    stubAgent({ body: agentResult() });
    render(<AgentPanel />);

    await askAndWait();

    expect(
      screen.getByText("The mean NDWI was 0.2777 index."),
    ).toBeInTheDocument();
  });

  it("renders readable tool labels taken from the response", async () => {
    stubAgent({ body: agentResult() });
    render(<AgentPanel />);

    await askAndWait();

    const tools = screen.getByRole("heading", { name: "Tools selected" })
      .parentElement as HTMLElement;
    expect(within(tools).getByText("Execute query")).toBeInTheDocument();
    expect(within(tools).getByText("NDWI statistics")).toBeInTheDocument();
    // A tool the response does not contain must not appear.
    expect(
      within(tools).queryByText("Temporal NDWI statistics"),
    ).not.toBeInTheDocument();
  });

  it("renders only the tools present in the response", async () => {
    stubAgent({
      body: agentResult({
        trace: {
          ...agentResult().trace,
          plan: { steps: [EXECUTE_STEP, { tool: "temporal_ndwi_statistics" }] },
          steps: [
            {
              status: "ok",
              parameters: EXECUTE_STEP,
              rejection_reason: null,
              error_message: null,
            },
            {
              status: "ok",
              parameters: { tool: "temporal_ndwi_statistics" },
              rejection_reason: null,
              error_message: null,
            },
          ],
        },
      }),
    });
    render(<AgentPanel />);

    await askAndWait();

    const tools = screen.getByRole("heading", { name: "Tools selected" })
      .parentElement as HTMLElement;
    expect(
      within(tools).getByText("Temporal NDWI statistics"),
    ).toBeInTheDocument();
    expect(within(tools).queryByText("NDWI statistics")).not.toBeInTheDocument();
  });

  it("renders the observable execution status of each step", async () => {
    stubAgent({
      body: agentResult({
        trace: {
          ...agentResult().trace,
          steps: [
            {
              status: "failed",
              parameters: EXECUTE_STEP,
              rejection_reason: null,
              error_message: "The satellite catalog is unavailable.",
            },
            {
              status: "skipped",
              parameters: { tool: "ndwi_statistics" },
              rejection_reason: "nothing to analyse",
              error_message: null,
            },
          ],
        },
      }),
    });
    render(<AgentPanel />);

    await askAndWait();

    const execution = screen.getByRole("heading", { name: "Execution" })
      .parentElement as HTMLElement;
    expect(within(execution).getByText(/failed/i)).toBeInTheDocument();
    expect(within(execution).getByText(/skipped/i)).toBeInTheDocument();
    expect(
      within(execution).getByText(/The satellite catalog is unavailable./),
    ).toBeInTheDocument();
  });

  it("renders the evidence returned by the backend", async () => {
    stubAgent({ body: agentResult() });
    render(<AgentPanel />);

    await askAndWait();

    const evidence = screen.getByRole("heading", { name: "Evidence" })
      .parentElement as HTMLElement;
    expect(within(evidence).getByText("ndwi.ndwi_mean")).toBeInTheDocument();
    expect(within(evidence).getByText(/0\.2777/)).toBeInTheDocument();
    expect(
      within(evidence).getByText("Equal AOI coverage is NOT established."),
    ).toBeInTheDocument();
  });

  it("shows a compact empty state when there is no evidence", async () => {
    stubAgent({
      body: agentResult({
        evidence: { items: [], execution: null, analysis: null },
      }),
    });
    render(<AgentPanel />);

    await askAndWait();

    const evidence = screen.getByRole("heading", { name: "Evidence" })
      .parentElement as HTMLElement;
    expect(within(evidence).getByText(/no evidence/i)).toBeInTheDocument();
  });
});

// ===========================================================================
// C. Failure statuses - no fabricated prose
// ===========================================================================

describe("AgentPanel - withheld and unavailable answers", () => {
  it("shows no answer text when the answer is null", async () => {
    stubAgent({
      body: agentResult({ status: "answer_withheld", answer: null }),
    });
    render(<AgentPanel />);

    await askAndWait();

    const answer = screen.getByRole("heading", { name: "Answer" })
      .parentElement as HTMLElement;
    expect(
      within(answer).queryByText("The mean NDWI was 0.2777 index."),
    ).not.toBeInTheDocument();
    expect(within(answer).getByText(/withheld/i)).toBeInTheDocument();
  });

  it("preserves the evidence when the answer is withheld", async () => {
    stubAgent({
      body: agentResult({ status: "answer_withheld", answer: null }),
    });
    render(<AgentPanel />);

    await askAndWait();

    const evidence = screen.getByRole("heading", { name: "Evidence" })
      .parentElement as HTMLElement;
    expect(within(evidence).getByText("ndwi.ndwi_mean")).toBeInTheDocument();
  });

  it("preserves the evidence when synthesis is unavailable", async () => {
    stubAgent({
      body: agentResult({ status: "synthesis_unavailable", answer: null }),
    });
    render(<AgentPanel />);

    await askAndWait();

    const evidence = screen.getByRole("heading", { name: "Evidence" })
      .parentElement as HTMLElement;
    expect(within(evidence).getByText("ndwi.ndwi_mean")).toBeInTheDocument();
    const answer = screen.getByRole("heading", { name: "Answer" })
      .parentElement as HTMLElement;
    expect(within(answer).getByText(/could not be generated/i)).toBeInTheDocument();
  });

  it("renders planner_unavailable safely with nothing invented", async () => {
    stubAgent({
      body: {
        status: "planner_unavailable",
        answer: null,
        trace: {
          plan: null,
          steps: [],
          evidence_refs: [],
          answer_validation: null,
        },
        evidence: { items: [], execution: null, analysis: null },
      },
    });
    render(<AgentPanel />);

    await askAndWait();

    expect(screen.getByText(/could not be planned/i)).toBeInTheDocument();
    const plan = screen.getByRole("heading", { name: "Plan" })
      .parentElement as HTMLElement;
    expect(within(plan).getByText(/no plan/i)).toBeInTheDocument();
  });

  it("never implies a withheld answer is trustworthy", async () => {
    stubAgent({
      body: agentResult({ status: "answer_withheld", answer: null }),
    });
    render(<AgentPanel />);

    await askAndWait();

    for (const claim of [/verified/i, /confirmed/i, /trustworthy/i, /accurate/i]) {
      expect(screen.queryByText(claim)).not.toBeInTheDocument();
    }
  });
});

// ===========================================================================
// D. Errors
// ===========================================================================

describe("AgentPanel - request failures", () => {
  it("renders a backend error without crashing", async () => {
    stubAgent({
      ok: false,
      status: 502,
      body: { error: { code: "upstream_error", message: "provider down" } },
    });
    render(<AgentPanel />);

    askQuestion();

    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent("provider down"),
    );
    expect(
      screen.queryByRole("heading", { name: "Answer" }),
    ).not.toBeInTheDocument();
  });

  it("renders a network failure as an error", async () => {
    stubAgent(() => Promise.reject(new Error("offline")));
    render(<AgentPanel />);

    askQuestion();

    await waitFor(() => expect(screen.getByRole("alert")).toBeInTheDocument());
    expect(screen.getByRole("button", { name: /^ask$/i })).toBeEnabled();
  });
});

// ===========================================================================
// E. Forbidden surfaces
// ===========================================================================

describe("AgentPanel - what it must never show", () => {
  it("displays no reasoning, thoughts or chain-of-thought labels", async () => {
    stubAgent({ body: agentResult() });
    const { container } = render(<AgentPanel />);

    await askAndWait();

    const rendered = container.textContent?.toLowerCase() ?? "";
    for (const banned of [
      "thought",
      "reasoning",
      "thinking",
      "chain of thought",
      "rationale",
      "internal",
    ]) {
      expect(rendered).not.toContain(banned);
    }
  });

  it("shows no provider or API-key configuration", async () => {
    stubAgent({ body: agentResult() });
    const { container } = render(<AgentPanel />);

    await askAndWait();

    const rendered = container.textContent?.toLowerCase() ?? "";
    for (const banned of ["api_key", "apikey", "gemini", "system_instruction"]) {
      expect(rendered).not.toContain(banned);
    }
  });
});
