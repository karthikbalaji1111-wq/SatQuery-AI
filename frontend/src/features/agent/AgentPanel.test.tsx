import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AgentAnswerPanel, AgentEvidencePanel, AgentPanel } from "./AgentPanel";

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
  fireEvent.click(screen.getByRole("button", { name: /^run analysis$/i }));
}

async function askAndWait(question: string = QUESTION) {
  askQuestion(question);
  await waitFor(() =>
    expect(
      screen.getByRole("heading", { name: "Pipeline" }).closest("section"),
    ).not.toHaveTextContent(/No run yet/),
  );
}

// ===========================================================================
// A. The form
// ===========================================================================

describe("AgentPanel - the Ask form", () => {
  it("renders a question field and an Ask button", () => {
    render(<AgentPanel />);

    expect(screen.getByLabelText(/question/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^run analysis$/i })).toBeInTheDocument();
  });

  it("disables Ask until a question is entered", () => {
    render(<AgentPanel />);
    const button = screen.getByRole("button", { name: /^run analysis$/i });

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

    expect(screen.getByRole("button", { name: /^run analysis$/i })).toBeDisabled();
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
      expect(screen.getByRole("button", { name: /running/i })).toBeDisabled(),
    );
    expect(screen.getByLabelText(/question/i)).toBeDisabled();

    release({
      ok: true,
      status: 200,
      text: () => Promise.resolve(JSON.stringify(agentResult())),
    } as Response);

    await waitFor(() =>
      expect(screen.getByRole("button", { name: /^run analysis$/i })).toBeEnabled(),
    );
  });
});

// ===========================================================================
// B. Successful result rendering, in the required order
// ===========================================================================

describe("AgentPanel - successful result", () => {
  it("renders query, execution, answer and evidence in band order", async () => {
    stubAgent({ body: agentResult() });
    render(<AgentPanel />);

    await askAndWait();

    const headings = screen
      .getAllByRole("heading")
      .map((node) => node.textContent);
    // Direction B band order: the question, what ran, the answer, then the
    // observation and evidence split. (This fixture carries no model
    // observation; its heading is covered by the visual-observation suite.)
    const order = [
      "Natural-language query",
      "Pipeline",
      "Analysis result",
      "Deterministic evidence",
    ];
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

    // Which tools ran is the execution stage list now; there is no separate
    // "Tools selected" section in this composition.
    const tools = screen.getByRole("heading", { name: "Pipeline" })
      .closest("section") as HTMLElement;
    expect(within(tools).getByText("stac_search")).toBeInTheDocument();
    expect(within(tools).getByText("ndwi_compute")).toBeInTheDocument();
    // A tool the response does not contain must not appear.
    expect(
      within(tools).queryByText("ndwi_temporal"),
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

    // Which tools ran is the execution stage list now; there is no separate
    // "Tools selected" section in this composition.
    const tools = screen.getByRole("heading", { name: "Pipeline" })
      .closest("section") as HTMLElement;
    expect(
      within(tools).getByText("ndwi_temporal"),
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

    const execution = screen.getByRole("heading", { name: "Pipeline" })
      .closest("section") as HTMLElement;
    expect(within(execution).getByText("stac_search")).toBeInTheDocument();
    expect(within(execution).getByText("ndwi_compute")).toBeInTheDocument();
    // The run state summarises the same outcome, separately from the rows.
    expect(within(execution).getByText(/^Failed$/)).toBeInTheDocument();
    // Visible text, not a `title`: on a non-focusable <li> a tooltip is a
    // mouse-only channel, so the one diagnostic explaining a failed run was
    // unreachable to keyboard and screen-reader users.
    expect(
      within(execution).getByText(/The satellite catalog is unavailable\./),
    ).toBeInTheDocument();
    expect(
      within(execution).getByText(/nothing to analyse/),
    ).toBeInTheDocument();
  });

  it("renders the evidence returned by the backend", async () => {
    stubAgent({ body: agentResult() });
    render(<AgentPanel />);

    await askAndWait();

    const evidence = screen.getByRole("heading", { name: "Deterministic evidence" })
      .closest("section") as HTMLElement;
    expect(within(evidence).getByText("ndwi.ndwi_mean")).toBeInTheDocument();
    // Twice by design: the index readout headlines it, the citation row below
    // carries the id and raw value that grounding resolves against.
    expect(
      within(evidence).getByText("ndwi_mean = 0.2777 index"),
    ).toBeInTheDocument();
    expect(within(evidence).getByText("+0.2777")).toBeInTheDocument();
    expect(
      within(evidence).getByText("Equal AOI coverage is NOT established."),
    ).toBeInTheDocument();
  });

  it("headlines a temporal comparison by the comparison, not one observation", async () => {
    // Observed live: both observations' means are named "ndwi_mean", so the
    // headline set the EARLIER one large as "ndwi mean" beside an answer about
    // change. A comparison is headlined by the measurements that name it.
    const item = (id: string, name: string, value: number, unit = "index") => ({
      id,
      source: "temporal_ndwi",
      measurement: { name, value, unit },
      text: null,
      produced_by: "analysis.engines.compare_ndwi_observations",
    });
    stubAgent({
      body: agentResult({
        answer: "The earlier mean NDWI was 0.02665 index.",
        evidence: {
          items: [
            item("temporal_ndwi.first.ndwi_mean", "ndwi_mean", 0.02665),
            item(
              "temporal_ndwi.first.ndwi_percent_above_index_threshold_0.3",
              "ndwi_percent_above_index_threshold_0.3",
              14.07,
              "%",
            ),
            item("temporal_ndwi.second.ndwi_mean", "ndwi_mean", 0.1464),
            item(
              "temporal_ndwi.difference.mean_ndwi_difference",
              "mean_ndwi_difference",
              0.1197,
            ),
            item("temporal_ndwi.change.ndwi_change_mean", "ndwi_change_mean", 0.1197),
          ],
          execution: null,
          analysis: null,
        },
      }),
    });
    render(<AgentPanel />);

    await askAndWait();

    const answer = screen.getByRole("heading", { name: "Analysis result" })
      .closest("section") as HTMLElement;
    const labels = [...answer.querySelectorAll(".metric-pair dt")].map(
      (node) => node.textContent,
    );
    expect(labels).toEqual(["mean ndwi difference", "ndwi change mean"]);
    expect(labels).not.toContain("ndwi mean");
  });

  it("shows no headline when a comparison's difference was suppressed", async () => {
    const mean = (id: string, value: number) => ({
      id,
      source: "temporal_ndwi",
      measurement: { name: "ndwi_mean", value, unit: "index" },
      text: null,
      produced_by: "analysis.engines.compare_ndwi_observations",
    });
    stubAgent({
      body: agentResult({
        answer: "The earlier mean NDWI was 0.02665 index.",
        evidence: {
          items: [
            mean("temporal_ndwi.first.ndwi_mean", 0.02665),
            mean("temporal_ndwi.second.ndwi_mean", 0.1464),
          ],
          execution: null,
          analysis: null,
        },
      }),
    });
    render(<AgentPanel />);

    await askAndWait();

    const answer = screen.getByRole("heading", { name: "Analysis result" })
      .closest("section") as HTMLElement;
    expect(answer.querySelector(".metric-pair")).toBeNull();
  });

  it("shows a compact empty state when there is no evidence", async () => {
    stubAgent({
      body: agentResult({
        evidence: { items: [], execution: null, analysis: null },
      }),
    });
    render(<AgentPanel />);

    await askAndWait();

    const evidence = screen.getByRole("heading", { name: "Deterministic evidence" })
      .closest("section") as HTMLElement;
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

    const answer = screen.getByRole("heading", { name: "Analysis result" })
      .closest("section") as HTMLElement;
    expect(
      within(answer).queryByText("The mean NDWI was 0.2777 index."),
    ).not.toBeInTheDocument();
    expect(within(answer).getByText(/An answer was generated but failed validation/i)).toBeInTheDocument();
  });

  it("preserves the evidence when the answer is withheld", async () => {
    stubAgent({
      body: agentResult({ status: "answer_withheld", answer: null }),
    });
    render(<AgentPanel />);

    await askAndWait();

    const evidence = screen.getByRole("heading", { name: "Deterministic evidence" })
      .closest("section") as HTMLElement;
    expect(within(evidence).getByText("ndwi.ndwi_mean")).toBeInTheDocument();
  });

  it("preserves the evidence when synthesis is unavailable", async () => {
    stubAgent({
      body: agentResult({ status: "synthesis_unavailable", answer: null }),
    });
    render(<AgentPanel />);

    await askAndWait();

    const evidence = screen.getByRole("heading", { name: "Deterministic evidence" })
      .closest("section") as HTMLElement;
    expect(within(evidence).getByText("ndwi.ndwi_mean")).toBeInTheDocument();
    const answer = screen.getByRole("heading", { name: "Analysis result" })
      .closest("section") as HTMLElement;
    expect(within(answer).getByText(/written summary is missing/i)).toBeInTheDocument();
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

    expect(screen.getByText(/did not return a usable plan/i)).toBeInTheDocument();
    // The Plan and Tools sections are gone: what the server was asked to do is
    // now the context strip, and which tools ran is the execution stage list.
    // With no plan, both must stay empty rather than echo the question back as
    // though it had been understood.
    const execution = screen.getByRole("heading", { name: "Pipeline" })
      .closest("section") as HTMLElement;
    expect(
      within(execution).getByText(/no stages executed yet/i),
    ).toBeInTheDocument();
    expect(screen.queryByText("Location")).not.toBeInTheDocument();
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
    // The Answer band is part of the composition and always present; what must
    // never happen is it presenting an answer when the request failed.
    const answer = screen.getByRole("heading", { name: "Analysis result" })
      .closest("section") as HTMLElement;
    expect(within(answer).getByText(/Run an analysis to produce/i)).toBeInTheDocument();
    expect(within(answer).queryByText(/provider down/)).not.toBeInTheDocument();
  });

  it("renders a network failure as an error", async () => {
    stubAgent(() => Promise.reject(new Error("offline")));
    render(<AgentPanel />);

    askQuestion();

    await waitFor(() => expect(screen.getByRole("alert")).toBeInTheDocument());
    expect(screen.getByRole("button", { name: /^run analysis$/i })).toBeEnabled();
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

// =========================================================================== //
// Phase 18.1 - the visual observation
// =========================================================================== //
//
// A model observation is not a measurement and must never be dressed as one.
// These tests pin the attribution: the model is named on screen, the block is
// structurally distinct from the deterministic evidence list, and nothing about
// it reads as "verified".

const VISUAL_ITEM = {
  id: "model.visual.S2B_44PMV_20250104_0_L2A",
  source: "model",
  measurement: null,
  text: null,
  visual: {
    statement: "Water is visible along the eastern shoreline.",
    provider: "gemini",
    model: "gemini-3.6-flash",
    scene_id: "S2B_44PMV_20250104_0_L2A",
  },
  produced_by: "gemini-3.6-flash",
};

function visualResult(overrides: Record<string, unknown> = {}) {
  return {
    status: "ok",
    answer: "Water is visible along the eastern shoreline.",
    trace: {
      plan: {
        steps: [
          EXECUTE_STEP,
          { tool: "rs_model_analysis", question: "Is there visible water?" },
        ],
      },
      steps: [
        { status: "ok", parameters: EXECUTE_STEP, rejection_reason: null, error_message: null },
        {
          status: "ok",
          parameters: { tool: "rs_model_analysis", question: "Is there visible water?" },
          rejection_reason: null,
          error_message: null,
        },
      ],
      evidence_refs: [VISUAL_ITEM.id],
      answer_validation: {
        numeric_grounding: "pass",
        forbidden_terms: "pass",
        evidence_refs: "pass",
        visual_claims: "attributed",
      },
    },
    evidence: { items: [VISUAL_ITEM], execution: null, analysis: null },
    ...overrides,
  };
}

async function askWith(body: unknown) {
  vi.spyOn(globalThis, "fetch").mockResolvedValue({
    ok: true,
    status: 200,
    text: () => Promise.resolve(JSON.stringify(body)),
  } as Response);
  render(<AgentPanel />);
  fireEvent.change(screen.getByLabelText("Question"), {
    target: { value: "Is there visible water?" },
  });
  fireEvent.click(screen.getByRole("button", { name: /^run analysis$/i }));
  await waitFor(() =>
    expect(
      screen.getByRole("heading", { name: "Pipeline" }).closest("section"),
    ).not.toHaveTextContent(/No run yet/),
  );
}

describe("AgentPanel - visual observation", () => {
  it("renders the model's statement", async () => {
    await askWith(visualResult());
    // Appears both as the observation and (here) as the synthesised answer -
    // scope to the observation block rather than asserting global uniqueness.
    const block = screen
      .getByRole("heading", { name: "Visual observation" })
      .closest("section") as HTMLElement;
    expect(
      within(block).getByText(/Water is visible along the eastern shoreline\./),
    ).toBeInTheDocument();
  });

  it("attributes the observation to the named model", async () => {
    await askWith(visualResult());
    expect(screen.getByText(/^Model observation · /)).toBeInTheDocument();
    expect(screen.getByText(/gemini-3\.6-flash/)).toBeInTheDocument();
  });

  it("attributes an Anthropic observation to Claude by its own model id", async () => {
    // The label is display casing only; the model id comes from the response,
    // so a run answered by Claude is as attributable as one answered by Gemini.
    await askWith(
      visualResult({
        evidence: {
          items: [
            {
              ...VISUAL_ITEM,
              visual: {
                ...VISUAL_ITEM.visual,
                provider: "anthropic",
                model: "claude-opus-5",
              },
              produced_by: "claude-opus-5",
            },
          ],
          execution: null,
          analysis: null,
        },
      }),
    );
    expect(screen.getByText(/Claude/)).toBeInTheDocument();
    expect(screen.getByText(/claude-opus-5/)).toBeInTheDocument();
  });

  it("attributes a local observation to the local model by its own id", async () => {
    // A run answered on this machine is as attributable as a cloud one: the
    // provider reads "Local" and the model id is the Ollama tag that ran.
    await askWith(
      visualResult({
        evidence: {
          items: [
            {
              ...VISUAL_ITEM,
              visual: {
                ...VISUAL_ITEM.visual,
                provider: "local",
                model: "qwen3-vl:4b-instruct",
              },
              produced_by: "qwen3-vl:4b-instruct",
            },
          ],
          execution: null,
          analysis: null,
        },
      }),
    );
    expect(screen.getByText(/Local/)).toBeInTheDocument();
    expect(screen.getByText(/qwen3-vl:4b-instruct/)).toBeInTheDocument();
  });

  it("splits the observation and the evidence, observation first", async () => {
    await askWith(visualResult());
    const headings = screen
      .getAllByRole("heading")
      .map((node) => node.textContent);
    const answer = headings.indexOf("Analysis result");
    const observation = headings.indexOf("Visual observation");
    const evidence = headings.indexOf("Deterministic evidence");

    // Direction B §1: the answer is band 3; the interpretation and the
    // computed values split band 4 beneath it, never sharing a container.
    expect(answer).toBeGreaterThanOrEqual(0);
    expect(answer).toBeLessThan(observation);
    expect(observation).toBeLessThan(evidence);
  });

  it("labels the visual tool readably in the plan", async () => {
    await askWith(visualResult());
    expect(screen.getAllByText(/Visual observation/i).length).toBeGreaterThan(0);
  });

  it("does not render a visual observation as a measurement", async () => {
    await askWith(visualResult());
    // A measurement renders as "name = value unit"; a visual claim must not.
    expect(screen.queryByText(/=\s*null/)).not.toBeInTheDocument();
    expect(screen.queryByText(/undefined/)).not.toBeInTheDocument();
  });

  it("shows nothing visual when there is no model evidence", async () => {
    await askWith({
      ...visualResult(),
      evidence: { items: [], execution: null, analysis: null },
      trace: {
        ...visualResult().trace,
        evidence_refs: [],
        answer_validation: {
          numeric_grounding: "pass",
          forbidden_terms: "pass",
          evidence_refs: "pass",
          visual_claims: "not_run",
        },
      },
    });
    expect(screen.queryByText(/Model observation/i)).not.toBeInTheDocument();
  });
});

// --------------------------------------------------------------------------- #
// The agent -> map handoff
// --------------------------------------------------------------------------- #
//
// The agent plans `execute_query` with imagery, so a georeferenced scene comes
// back inside the evidence. These tests pin that it reaches the map exactly the
// way QueryPanel's does - passed upward, never re-fetched - and that a new
// question clears the previous scene before the new one arrives.

const AGENT_IMAGERY = {
  scene_id: "S2A_44PMV_20240115_0_L2A",
  media_type: "image/png",
  image_base64: "iVBORw0KGgo=",
  crs: "EPSG:32644",
  width: 112,
  height: 300,
  corners_wgs84: [
    [80.279621, 13.066132],
    [80.289951, 13.06616],
    [80.290029, 13.039034],
    [80.2797, 13.039006],
  ],
};

const NDWI_OVERLAY = {
  scene_id: "S2A_44PMV_20240115_0_L2A",
  window_label: "single",
  media_type: "image/png",
  image_base64: "TkRXSQ==",
  corners_wgs84: AGENT_IMAGERY.corners_wgs84,
};

function withEvidence(execution: unknown, analysis: unknown) {
  return agentResult({
    evidence: { items: EVIDENCE.items, execution, analysis },
  });
}

function executionWithImagery(imagery: unknown) {
  return {
    plan: { intent: INTENT, bbox: null },
    executed_modalities: ["sentinel-2-optical"],
    skipped_modalities: [],
    windows: [
      {
        modality: "sentinel-2-optical",
        label: "single",
        time_range: INTENT.time_windows[0],
        scene_count: 6,
        scenes: [],
        selected_scene_id: AGENT_IMAGERY.scene_id,
        imagery,
        imagery_error: null,
      },
    ],
    catalog: "https://example.test/v1",
  };
}

async function askWithHandlers(body: unknown, handlers: Record<string, unknown>) {
  stubAgent({ body });
  render(<AgentPanel {...handlers} />);
  fireEvent.change(screen.getByLabelText(/question/i), {
    target: { value: QUESTION },
  });
  fireEvent.click(screen.getByRole("button", { name: /^run analysis$/i }));
}

describe("AgentPanel - map handoff", () => {
  it("passes the executed window's imagery upward", async () => {
    const onImagery = vi.fn();
    await askWithHandlers(
      withEvidence(executionWithImagery(AGENT_IMAGERY), null),
      { onImagery },
    );

    await waitFor(() =>
      expect(onImagery).toHaveBeenLastCalledWith(
        expect.objectContaining({ scene_id: AGENT_IMAGERY.scene_id }),
      ),
    );
  });

  it("clears the previous scene before the new question is answered", async () => {
    const onImagery = vi.fn();
    await askWithHandlers(
      withEvidence(executionWithImagery(AGENT_IMAGERY), null),
      { onImagery },
    );

    // The very first call happens before the response lands.
    expect(onImagery).toHaveBeenNthCalledWith(1, null);
    await waitFor(() => expect(onImagery).toHaveBeenCalledTimes(2));
  });

  it("passes null when the execution retrieved no imagery", async () => {
    const onImagery = vi.fn();
    await askWithHandlers(withEvidence(executionWithImagery(null), null), {
      onImagery,
    });

    await waitFor(() =>
      expect(screen.getByRole("heading", { name: "Analysis result" })),
    );
    expect(onImagery).toHaveBeenLastCalledWith(null);
  });

  it("passes an NDWI overlay produced by the agent's analysis", async () => {
    const onNdwi = vi.fn();
    await askWithHandlers(
      withEvidence(executionWithImagery(AGENT_IMAGERY), {
        status: "ok",
        task: "visualize",
        answer: "Analysed.",
        windows_considered: [],
        warnings: [],
        measurements: [],
        temporal_comparison: null,
        ndwi_overlay: NDWI_OVERLAY,
      }),
      { onNdwi },
    );

    await waitFor(() =>
      expect(onNdwi).toHaveBeenLastCalledWith(
        expect.objectContaining({ scene_id: NDWI_OVERLAY.scene_id }),
      ),
    );
  });

  it("works with no handlers attached", async () => {
    stubAgent({ body: withEvidence(executionWithImagery(AGENT_IMAGERY), null) });
    render(<AgentPanel />);
    fireEvent.change(screen.getByLabelText(/question/i), {
      target: { value: QUESTION },
    });
    fireEvent.click(screen.getByRole("button", { name: /^run analysis$/i }));

    await waitFor(() =>
      expect(
        screen.getByRole("heading", { name: "Analysis result" }),
      ).toBeInTheDocument(),
    );
  });
});

// A run state must never contradict the run: planning failure means nothing
// executed, so the panel cannot report a completed run over an empty trace.
describe("AgentPanel - run state honesty", () => {
  it("does not report COMPLETE when planning produced no steps", async () => {
    await askWith({
      status: "planner_unavailable",
      answer: null,
      trace: { plan: null, steps: [], evidence_refs: [], answer_validation: null },
      evidence: { items: [], execution: null, analysis: null },
    });

    const execution = screen.getByRole("heading", { name: "Pipeline" })
      .closest("section") as HTMLElement;
    expect(within(execution).getByText(/^Not run$/)).toBeInTheDocument();
    expect(within(execution).queryByText(/^Complete$/)).not.toBeInTheDocument();
  });
});

// A window spanning a year boundary must keep both years: the abbreviation is
// only ever the repeated year, never information the reader needs.
describe("AgentPanel - window formatting", () => {
  it("keeps both years when the window crosses a year boundary", async () => {
    await askWith(
      agentResult({
        trace: {
          plan: {
            steps: [
              {
                ...EXECUTE_STEP,
                intent: {
                  ...INTENT,
                  time_windows: [
                    { start_date: "2024-12-01", end_date: "2025-01-31" },
                  ],
                },
              },
            ],
          },
          steps: [
            {
              status: "ok",
              parameters: EXECUTE_STEP,
              rejection_reason: null,
              error_message: null,
            },
          ],
          evidence_refs: [],
          answer_validation: null,
        },
      }),
    );
    expect(screen.getByText("2024-12-01 → 2025-01-31")).toBeInTheDocument();
  });
});

// --------------------------------------------------------------------------- #
// The query hero is a real input
// --------------------------------------------------------------------------- #
//
// Regression: the placeholder once carried a fully-formed Marina Beach question
// styled like entered text, so the resting hero read as a hardcoded demo query.
// The field must start empty, prompt without naming a place, and run only what
// the user actually typed.

describe("AgentPanel - the query hero is a real input", () => {
  it("starts empty with a prompt that names no location", () => {
    render(<AgentPanel />);
    const input = screen.getByLabelText(/question/i) as HTMLTextAreaElement;

    expect(input.value).toBe("");
    const placeholder = input.placeholder;
    expect(placeholder).not.toMatch(/marina|chennai|sentinel-2 image of/i);
    expect(placeholder).toMatch(/ask about/i);
  });

  it("disables Run analysis until the user has typed something", () => {
    render(<AgentPanel />);
    const run = screen.getByRole("button", { name: /^run analysis$/i });
    expect(run).toBeDisabled();

    fireEvent.change(screen.getByLabelText(/question/i), {
      target: { value: "Any water near Kochi?" },
    });
    expect(run).toBeEnabled();
  });

  it("sends the user's own text, not an example", async () => {
    const fetchMock = stubAgent({ body: agentResult() });
    render(<AgentPanel />);

    const typed = "Any visible water near Kochi in March 2025?";
    fireEvent.change(screen.getByLabelText(/question/i), {
      target: { value: typed },
    });
    fireEvent.click(screen.getByRole("button", { name: /^run analysis$/i }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    const body = JSON.parse(
      (fetchMock.mock.calls[0][1] as { body: string }).body,
    );
    expect(body.question).toBe(typed);
    // The submitted text stays in the hero after the run.
    expect(
      (screen.getByLabelText(/question/i) as HTMLTextAreaElement).value,
    ).toBe(typed);
  });

  it("fills the input from an example without submitting it", () => {
    const fetchMock = stubAgent({ body: agentResult() });
    render(<AgentPanel />);

    fireEvent.click(screen.getByRole("button", { name: /radar backscatter/i }));

    const input = screen.getByLabelText(/question/i) as HTMLTextAreaElement;
    expect(input.value).toMatch(/SAR backscatter around Marina Beach, Chennai/);
    // Clicking an example is not a submission.
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe("AgentPanel - honest imagery degradation", () => {
  // A window can discover and select a scene and still fail to retrieve its
  // picture. That is the documented Sentinel-1 case: the catalog publishes the
  // measurement asset on storage this deployment holds no credentials for.
  // The agent path used to drop `imagery_error` entirely, so such a run looked
  // identical to one that never asked for imagery.
  const S1_ERROR =
    "Asset 'vv' is published as s3:// which this deployment cannot read; " +
    "only anonymous HTTPS assets are supported. Sentinel-1 GRD measurement " +
    "assets are currently published this way, so bounded Sentinel-1 " +
    "retrieval is not available - see the Sentinel-1 note in the README.";

  function withImageryError(message: string) {
    const base = executionWithImagery(null);
    const execution = {
      ...base,
      windows: [{ ...base.windows[0], imagery_error: message }],
    };
    return agentResult({
      evidence: { items: [], execution, analysis: null },
    });
  }

  it("states that Sentinel-1 imagery was not retrieved, and why", async () => {
    stubAgent({ body: withImageryError(S1_ERROR) });
    render(<AgentPanel />);
    await askAndWait();

    const evidence = screen
      .getByRole("heading", { name: "Deterministic evidence" })
      .closest("section") as HTMLElement;

    expect(
      within(evidence).getByText(/Sentinel-1 imagery is not available/i),
    ).toBeInTheDocument();
    // Discovery still stands - the reader must not read this as a total failure.
    expect(
      within(evidence).getByText(/discovery and metadata are unaffected/i),
    ).toBeInTheDocument();
  });

  it("does not put the server's operator-facing wording in the headline", async () => {
    stubAgent({ body: withImageryError(S1_ERROR) });
    render(<AgentPanel />);
    await askAndWait();

    const summary = screen.getByText(/Sentinel-1 imagery is not available/i);
    // The README pointer and the URI scheme belong in the disclosure, not in
    // the sentence a judge reads off the screen.
    expect(summary.textContent).not.toMatch(/README/i);
    expect(summary.textContent).not.toMatch(/s3:\/\//);
  });

  it("keeps the server's own cause available rather than hiding it", async () => {
    stubAgent({ body: withImageryError(S1_ERROR) });
    render(<AgentPanel />);
    await askAndWait();

    // Inspectable, just not shouted: a documented boundary should be checkable.
    expect(screen.getByText(/Technical cause/i)).toBeInTheDocument();
    expect(screen.getByText(/s3:\/\//)).toBeInTheDocument();
  });

  it("never implies a SAR measurement was produced", async () => {
    stubAgent({ body: withImageryError(S1_ERROR) });
    render(<AgentPanel />);
    await askAndWait();

    const body = document.body.textContent ?? "";
    expect(body).not.toMatch(/backscatter/i);
    expect(body).not.toMatch(/calibrat/i);
    expect(body).not.toMatch(/VV\s*(value|measurement|index)/i);
  });
});


describe("AgentPanel provider failures", () => {
  it("shows the actual quota failure and provider retry hint", async () => {
    await askWith(agentResult({
      status: "planner_unavailable", answer: null,
      trace: { plan: null, steps: [], evidence_refs: [], answer_validation: null },
      evidence: { items: [], execution: null, analysis: null },
      failure: { stage: "planning", code: "rate_limited", message: "Gemini returned HTTP 429.", retry_after_seconds: 44 },
    }));
    expect(screen.getByText(/Planning · rate_limited: Gemini returned HTTP 429/)).toBeInTheDocument();
    expect(screen.getByText(/wait of 44 seconds before retrying/)).toBeInTheDocument();
    expect(screen.queryByText(/Running the same question again may/)).not.toBeInTheDocument();
  });

  it("does not claim complete analysis when synthesis fails without measurements", async () => {
    await askWith(agentResult({
      status: "synthesis_unavailable", answer: null,
      evidence: { items: [], execution: null, analysis: null },
    }));
    expect(screen.getByText(/written summary is missing/)).toBeInTheDocument();
    expect(screen.queryByText(/analysis succeeded|complete and valid|answered from measured values alone/i)).not.toBeInTheDocument();
  });
});

// ===========================================================================
// Deterministic evidence: one index's numbers never stand under another's name
// ===========================================================================

const THREE_INDEX_MEASUREMENTS = [
  { name: "ndvi_valid_pixel_count", value: 33600, unit: "pixels" },
  { name: "ndvi_mean", value: -0.0614, unit: "index" },
  { name: "ndbi_valid_pixel_count", value: 33600, unit: "pixels" },
  { name: "ndbi_mean", value: 0.0118, unit: "index" },
  { name: "ndwi_valid_pixel_count", value: 33600, unit: "pixels" },
  { name: "ndwi_mean", value: 0.1464, unit: "index" },
  {
    name: "ndwi_percent_above_index_threshold_0.3",
    value: 45.41,
    unit: "%",
  },
];

/** An agent result that ran nothing at all - the rate-limited planner case. */
function plannerUnavailable() {
  return agentResult({
    status: "planner_unavailable",
    answer: null,
    trace: { plan: null, steps: [], evidence_refs: [], answer_validation: null },
    evidence: { items: [], execution: null, analysis: null },
  });
}

describe("AgentEvidencePanel - measurements are attributed to their own index", () => {
  function renderWithManual(measurements: unknown[]) {
    return render(
      <AgentEvidencePanel
        evidence={null}
        manual={{
          scene: null,
          imagery: null,
          measurements: measurements as never,
        }}
        bbox={null}
      />,
    );
  }

  it("renders one readout per index that reported a mean", () => {
    const { container } = renderWithManual(THREE_INDEX_MEASUREMENTS);
    const readouts = container.querySelectorAll(".index-readout");
    expect(readouts).toHaveLength(3);
    expect(screen.getByText("ndvi mean")).toBeInTheDocument();
    expect(screen.getByText("ndwi mean")).toBeInTheDocument();
    expect(screen.getByText("ndbi mean")).toBeInTheDocument();
  });

  it("shows each index's own value, not only the first one returned", () => {
    const { container } = renderWithManual(THREE_INDEX_MEASUREMENTS);
    const values = [...container.querySelectorAll(".index-value")].map(
      (node) => node.textContent,
    );
    expect(values).toEqual(["-0.0614", "+0.1464", "+0.0118"]);
  });

  it("attaches the NDWI threshold percentage to NDWI and to nothing else", () => {
    const { container } = renderWithManual(THREE_INDEX_MEASUREMENTS);
    const readouts = [...container.querySelectorAll(".index-readout")];
    const withNote = readouts.filter(
      (node) => node.querySelector(".index-note") !== null,
    );
    // Exactly one readout carries a percentage, and it is the NDWI one - the
    // percentage is an NDWI threshold count and belongs to no other index.
    expect(withNote).toHaveLength(1);
    expect(withNote[0].textContent).toContain("ndwi mean");
    expect(withNote[0].textContent).toContain("45.4");
  });

  it("never labels a pixel-quality percentage as a threshold result", () => {
    // The live NDVI run (S2B_44PMV_20250104, Marina Beach): pixel quality adds
    // its own counts and percentages under the index's prefix, in this order.
    // NDVI has no threshold, so "99.8% above threshold" was the valid-pixel
    // share (33,524 of 33,600) wearing the wrong label.
    const { container } = renderWithManual([
      { name: "ndvi_valid_pixel_count", value: 33524, unit: "pixels" },
      { name: "ndvi_mean", value: -0.0613, unit: "index" },
      { name: "ndvi_quality_total_pixel_count", value: 33600, unit: "pixels" },
      { name: "ndvi_quality_cloud_pixel_count", value: 76, unit: "pixels" },
      { name: "ndvi_quality_valid_percent", value: 99.77, unit: "%" },
      { name: "ndvi_quality_contamination_percent", value: 0.23, unit: "%" },
      { name: "ndwi_quality_total_pixel_count", value: 33600, unit: "pixels" },
      { name: "ndwi_quality_valid_percent", value: 99.77, unit: "%" },
      { name: "ndwi_valid_pixel_count", value: 33524, unit: "pixels" },
      { name: "ndwi_mean", value: 0.1464, unit: "index" },
      {
        name: "ndwi_percent_above_index_threshold_0.3",
        value: 45.41,
        unit: "%",
      },
    ]);
    const readouts = [...container.querySelectorAll(".index-readout")];
    const ndvi = readouts.find((node) => node.textContent?.includes("ndvi mean"));
    const ndwi = readouts.find((node) => node.textContent?.includes("ndwi mean"));
    expect(ndvi?.querySelector(".index-note")).toBeNull();
    expect(ndwi?.querySelector(".index-note")?.textContent).toContain("45.4");
    expect(ndwi?.textContent).toContain("33,524");
    expect(container.textContent).not.toMatch(/99\.8% above/);
  });

  it("captions each index with what THAT index is not a classification of", () => {
    const { container } = renderWithManual(THREE_INDEX_MEASUREMENTS);
    const readouts = [...container.querySelectorAll(".index-readout")];
    const ndvi = readouts.find((node) =>
      node.textContent?.includes("ndvi mean"),
    );
    const ndwi = readouts.find((node) =>
      node.textContent?.includes("ndwi mean"),
    );
    // An NDVI mean captioned "not a validated water classification" is a
    // statement about the wrong index.
    expect(ndvi?.querySelector(".index-caveat")?.textContent).not.toMatch(
      /water/i,
    );
    expect(ndvi?.querySelector(".index-caveat")?.textContent).toMatch(
      /vegetation|land-cover/i,
    );
    expect(ndwi?.querySelector(".index-caveat")?.textContent).toMatch(/water/i);
  });

  it("states the shared valid-pixel count once when every index agrees", () => {
    renderWithManual(THREE_INDEX_MEASUREMENTS);
    expect(screen.getByText("Valid pixels")).toBeInTheDocument();
    expect(screen.getByText("33,600 px")).toBeInTheDocument();
  });

  it("states each index's own pixel count when the counts differ", () => {
    const { container } = renderWithManual([
      { name: "ndwi_valid_pixel_count", value: 33600, unit: "pixels" },
      { name: "ndwi_mean", value: 0.1464, unit: "index" },
      { name: "ndbi_valid_pixel_count", value: 8400, unit: "pixels" },
      { name: "ndbi_mean", value: 0.0118, unit: "index" },
    ]);
    // No single count stands for both, so the field grid must not print one.
    expect(screen.queryByText("Valid pixels")).not.toBeInTheDocument();
    const perIndex = [...container.querySelectorAll(".index-pixels")].map(
      (node) => node.textContent,
    );
    expect(perIndex).toEqual([
      "33,600 valid pixels",
      "8,400 valid pixels",
    ]);
  });

  it("still renders a measurement whose name matches no known index", () => {
    const { container } = renderWithManual([
      { name: "mystery_mean", value: 0.5, unit: "index" },
    ]);
    expect(container.querySelectorAll(".index-readout")).toHaveLength(1);
    expect(screen.getByText("+0.5000")).toBeInTheDocument();
  });
});

describe("AgentEvidencePanel - a run that measured nothing hides nothing", () => {
  it("keeps manual measurements when the agent result carries none", () => {
    // The rate-limited planner returns a result with zero evidence items.
    // Selecting the measurement source on `evidence === null` threw the
    // manual path's real measurements away while its scene fields still
    // rendered - the panel reporting no evidence with the evidence in hand.
    render(
      <AgentEvidencePanel
        evidence={plannerUnavailable().evidence as never}
        manual={{
          scene: null,
          imagery: null,
          measurements: THREE_INDEX_MEASUREMENTS as never,
        }}
        bbox={null}
      />,
    );
    expect(screen.getByText("ndwi mean")).toBeInTheDocument();
    expect(screen.getByText("+0.1464")).toBeInTheDocument();
    expect(screen.queryByText(/No evidence was collected/)).not.toBeInTheDocument();
  });

  it("prefers the agent's own measurements when it has them", () => {
    render(
      <AgentEvidencePanel
        evidence={EVIDENCE as never}
        manual={{
          scene: null,
          imagery: null,
          measurements: [
            { name: "ndwi_mean", value: -0.9, unit: "index" },
          ] as never,
        }}
        bbox={null}
      />,
    );
    // 0.2777 is the agent's; -0.9 is the stale manual value it must not show.
    expect(screen.getByText("+0.2777")).toBeInTheDocument();
    expect(screen.queryByText("-0.9000")).not.toBeInTheDocument();
  });
});


// =========================================================================== #
// A completed result is a completed result, whichever workflow produced it.
//
// The defect: the answer rail and the evidence export were gated on the AGENT
// result. A successful manual analysis therefore left "Run an analysis to
// produce an answer" on screen with export unavailable, while its own
// measurements sat in the panel directly beside it - the UI instructing the
// user to do the thing they had just done.
//
// The manual path still produces no written answer, and none is invented here.
// What changes is that an instruction is replaced by a statement of fact.
// =========================================================================== #

describe("AgentAnswerPanel - completed result is origin-independent", () => {
  it("invites a run when nothing has happened", () => {
    render(<AgentAnswerPanel result={null} />);

    expect(screen.getByRole("status").textContent).toMatch(/Run an analysis/i);
  });

  it("does not instruct a run after a manual analysis completed", () => {
    render(<AgentAnswerPanel result={null} manualComplete />);

    const status = screen.getByRole("status").textContent ?? "";
    expect(status).not.toMatch(/Run an analysis/i);
    expect(status).toMatch(/complete/i);
  });

  it("does not invent a written answer for the manual path", () => {
    // The honest half: manual runs have no question, so there is nothing to
    // answer. Saying so is the point; fabricating prose would not be.
    render(<AgentAnswerPanel result={null} manualComplete />);

    const status = screen.getByRole("status").textContent ?? "";
    expect(status).toMatch(/no written answer/i);
    expect(status).toMatch(/evidence/i);
  });

  it("still says it is working while a run is in flight", () => {
    render(<AgentAnswerPanel result={null} manualComplete busy />);

    expect(screen.getByRole("status").textContent).toMatch(/Analysing/i);
  });
});


// ===========================================================================
// M5.5: a question put back to the user
// ===========================================================================

describe("AgentPanel - clarification", () => {
  const CLARIFICATION = {
    status: "needs_clarification",
    answer: null,
    failure: null,
    clarification: {
      reason: "date_missing",
      message:
        "For which date or period? Give a month and year or a date - for example 'in January 2025'.",
      options: [],
      understood_analyses: ["vegetation (NDVI)"],
      understood_location: "Chennai",
      understood_periods: [],
    },
    trace: { plan: null, steps: [], evidence_refs: [], answer_validation: null },
    evidence: { items: [], execution: null, analysis: null },
  };

  it("shows the server's question under the query box, with what was understood", async () => {
    stubAgent({ body: CLARIFICATION });
    render(<AgentPanel />);

    await askAndWait("Show vegetation around Chennai");

    const prompt = await screen.findByText(/For which date or period\?/);
    const block = prompt.closest(".clarification") as HTMLElement;
    expect(block).toHaveAttribute("data-reason", "date_missing");
    expect(within(block).getByText(/Understood so far/)).toHaveTextContent(
      "vegetation (NDVI) · Chennai",
    );
  });

  it("lists the supported choices the server named, and nothing else", async () => {
    stubAgent({
      body: {
        ...CLARIFICATION,
        clarification: {
          ...CLARIFICATION.clarification,
          reason: "analysis_missing",
          message: "What would you like to analyse?",
          options: ["vegetation (NDVI)", "water (NDWI)"],
          understood_analyses: [],
        },
      },
    });
    render(<AgentPanel />);

    await askAndWait("I want to know something about Chennai");

    const list = await screen.findByRole("list", { name: "Supported choices" });
    expect(within(list).getAllByRole("listitem").map((li) => li.textContent)).toEqual([
      "vegetation (NDVI)",
      "water (NDWI)",
    ]);
  });

  it("asks back when the intent model and the rules disagree", async () => {
    stubAgent({
      body: {
        ...CLARIFICATION,
        clarification: {
          ...CLARIFICATION.clarification,
          reason: "analysis_ambiguous",
          message:
            "The question could mean vegetation (NDVI) or built-up area (NDBI). Which should be computed?",
          options: ["vegetation (NDVI)", "built-up area (NDBI)"],
          understood_analyses: ["vegetation (NDVI)"],
        },
      },
    });
    render(<AgentPanel />);

    await askAndWait("Show NDVI around Pune in 2024");

    const block = (await screen.findByText(/could mean vegetation/)).closest(
      ".clarification",
    ) as HTMLElement;
    expect(block).toHaveAttribute("data-reason", "analysis_ambiguous");
    expect(
      within(block).getAllByRole("listitem").map((li) => li.textContent),
    ).toEqual(["vegetation (NDVI)", "built-up area (NDBI)"]);
  });

  it("reads 'Needs clarification' in the pipeline, not a failure", async () => {
    stubAgent({ body: CLARIFICATION });
    render(<AgentPanel />);

    await askAndWait("Show vegetation around Chennai");

    const pipeline = screen
      .getByRole("heading", { name: "Pipeline" })
      .closest("section") as HTMLElement;
    await waitFor(() =>
      expect(within(pipeline).getByText("Needs clarification")).toBeInTheDocument(),
    );
    expect(within(pipeline).queryByText("Failed")).not.toBeInTheDocument();
  });

  it("presents no answer and no measurement for a clarification", async () => {
    stubAgent({ body: CLARIFICATION });
    render(<AgentPanel />);

    await askAndWait("Show vegetation around Chennai");

    expect(
      await screen.findByText(/No analysis ran: the question needs one more detail/),
    ).toBeInTheDocument();
    expect(screen.queryByText(/mean NDVI/i)).not.toBeInTheDocument();
  });
});
