import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { AgentResult } from "../../api/types";
import { AgentAnswerPanel, AgentObservationPanel } from "./AgentPanel";
import { buildEvidenceReport } from "./evidenceReport";
import { describeObservation, interpretResult, technicalDetails } from "./interpretation";
import {
  NDWI_MARINA,
  NDWI_WITH_LOOK_OBSERVED,
  NDWI_WITH_LOOK_UNAVAILABLE,
  UNSUPPORTED_SHIPS,
  VISUAL_NO_IMAGE,
  VISUAL_OBSERVED,
  VISUAL_UNAVAILABLE,
} from "./m6Fixtures";
import { observationOf, outcomeOf, resultSummary, runStages } from "./resultModel";

function answerPanel(): HTMLElement {
  return screen
    .getByRole("heading", { name: "Analysis result" })
    .closest("section") as HTMLElement;
}

function observationPanel(): HTMLElement {
  return screen
    .getByRole("heading", { name: "Visual observation" })
    .closest("section") as HTMLElement;
}

// Words a newcomer should not meet in the visual copy. The technical record,
// folded away, may use them.
const JARGON = /\b(VLM|vision-language|RGB|TCI|asset|scene id|L2A|Level-2A|provider|inference|qualitative)\b/i;

function primaryText(panel: HTMLElement): string {
  const clone = panel.cloneNode(true) as HTMLElement;
  clone.querySelectorAll("details").forEach((node) => node.remove());
  return clone.textContent ?? "";
}

// =========================================================================== //
// The model: what each visual state is
// =========================================================================== //

describe("visual result model", () => {
  it("an observed description is a result, and its body is the description", () => {
    expect(outcomeOf(VISUAL_OBSERVED).kind).toBe("success");
    const body = resultSummary(VISUAL_OBSERVED);
    expect(body.kind).toBe("visual");
    if (body.kind !== "visual") throw new Error("unreachable");
    expect(body.observation.statement).toMatch(/sandy beach/);
    expect(body.observation.acquired).toBe("2025-01-04");
  });

  it("no model is its own state - never insufficient evidence", () => {
    const outcome = outcomeOf(VISUAL_UNAVAILABLE);
    expect(outcome.kind).toBe("visual_unavailable");
    expect(outcome.label).toBe("Visual description unavailable");
    expect(outcome.reason).toMatch(/needs an AI visual model/);
  });

  it("no image to describe says so instead of the generic line", () => {
    const outcome = outcomeOf(VISUAL_NO_IMAGE);
    expect(outcome.kind).toBe("insufficient_evidence");
    expect(outcome.reason).toBe("No normal-colour satellite image was available to describe.");
  });

  it("a measurement plus an undescribed image is still a measured result", () => {
    expect(outcomeOf(NDWI_WITH_LOOK_UNAVAILABLE).kind).toBe("success");
    expect(resultSummary(NDWI_WITH_LOOK_UNAVAILABLE).kind).toBe("index");
  });

  it("an observation never becomes a measurement or an interpretation", () => {
    expect(interpretResult(VISUAL_OBSERVED)).toBeNull();
    const observation = observationOf(VISUAL_OBSERVED)!;
    expect(observation).not.toHaveProperty("value");
    // The measured case keeps its interpretation, from the measurement alone.
    expect(interpretResult(NDWI_WITH_LOOK_OBSERVED)?.headline).toBe(
      interpretResult({ ...NDWI_MARINA, evidence: NDWI_WITH_LOOK_OBSERVED.evidence } as AgentResult)?.headline,
    );
  });

  it("no visual step means no observation and no visual stage", () => {
    expect(observationOf(NDWI_MARINA)).toBeNull();
    expect(runStages(NDWI_MARINA).map((stage) => stage.name)).not.toContain("Describe the image");
  });

  it("the describe stage says what happened", () => {
    const stage = (result: AgentResult) =>
      runStages(result).find((candidate) => candidate.name === "Describe the image");
    expect(stage(VISUAL_OBSERVED)).toMatchObject({ state: "done", detail: "described by an AI visual model" });
    expect(stage(VISUAL_UNAVAILABLE)).toMatchObject({ state: "attention", detail: "no AI visual model available here" });
    expect(stage(VISUAL_NO_IMAGE)).toMatchObject({ detail: "no normal-colour image to describe" });
  });

  it("the wording names the image and its day, and claims no measurement", () => {
    const wording = describeObservation(observationOf(VISUAL_OBSERVED)!);
    expect(wording.source).toBe("Sentinel-2 satellite image · 4 January 2025");
    expect(wording.howWeKnow).toMatch(/does not measure anything/);
    expect(wording.howWeKnow).toMatch(/place's name was taken out/);
    expect(`${wording.source} ${wording.howWeKnow}`).not.toMatch(JARGON);
  });

  it("the technical record keeps the model, the image and its size", () => {
    const rows = Object.fromEntries(
      (technicalDetails(VISUAL_OBSERVED) ?? []).map((row) => [row.label, row.value]),
    );
    expect(rows["Observation by"]).toBe("local · qwen3-vl:4b-instruct");
    expect(rows["Image size"]).toBe("112 × 300 px");
    expect(rows["Scene ID"]).toBe("S2B_44PMV_20250104_0_L2A");
    expect(rows.Nature).toMatch(/never a measurement/);
  });
});

// =========================================================================== //
// The screen: the three states a reader can meet
// =========================================================================== //

describe("visual answer panel", () => {
  it("observed: 'What I see', the image's day, and who described it", () => {
    render(<AgentAnswerPanel result={VISUAL_OBSERVED} />);
    const panel = answerPanel();
    const block = within(panel).getByRole("heading", { name: "What I see" }).closest("section")!;
    expect(block).toHaveTextContent(/A long sandy beach runs beside dark sea water/);
    expect(block).toHaveTextContent("Sentinel-2 satellite image · 4 January 2025");
    expect(block).toHaveTextContent(/Local · qwen3-vl:4b-instruct/);
    expect(block).toHaveTextContent(/not a measurement/);
    expect(panel.querySelector(".result-value")).toBeNull();
    expect(primaryText(panel)).not.toMatch(JARGON);
  });

  it("unavailable: the image was retrieved, and it says why it was not described", () => {
    render(<AgentAnswerPanel result={VISUAL_UNAVAILABLE} />);
    const panel = answerPanel();
    expect(within(panel).getByText("Visual description unavailable")).toHaveClass("outcome-chip");
    expect(panel).toHaveTextContent("The satellite image was retrieved, but it was not described.");
    expect(panel).toHaveTextContent(/needs an AI visual model, which is not available here/);
    expect(panel).toHaveTextContent(/shown on the map/);
    expect(panel).not.toHaveTextContent(/Insufficient evidence/i);
    expect(panel).not.toHaveTextContent(/No measurement could be made/);
    expect(primaryText(panel)).not.toMatch(JARGON);
    // Which image, in plain words; its record folds away with the sentence.
    const rows = Object.fromEntries(
      [...panel.querySelectorAll(".result-context > div")].map((row) => [
        row.querySelector("dt")?.textContent,
        row.querySelector("dd")?.textContent,
      ]),
    );
    expect(rows).toMatchObject({ "Satellite image": "Sentinel-2B", Date: "4 January 2025" });
    const sentence = within(panel).getByText(/was selected\./);
    expect(sentence.closest("details.technical-details")).not.toBeNull();
    expect(panel.querySelector("details.technical-details")).toHaveTextContent(
      /not produced - no visual model is configured/,
    );
  });

  it("no image: says there was no image, not that the evidence failed", () => {
    render(<AgentAnswerPanel result={VISUAL_NO_IMAGE} />);
    const panel = answerPanel();
    expect(panel).toHaveTextContent("No normal-colour satellite image was available to describe.");
    expect(panel).not.toHaveTextContent(/did not support a measurement/);
  });

  it("measure and look: the measurement first, then what became of the look", () => {
    render(<AgentAnswerPanel result={NDWI_WITH_LOOK_UNAVAILABLE} />);
    const panel = answerPanel();
    expect(panel.querySelector(".result-value")).toHaveTextContent("+0.1466");
    const look = panel.querySelector(".what-i-see")!;
    expect(look).toHaveTextContent(/needs an AI visual model/);
    // The card comes before the look.
    expect(
      panel.querySelector(".result-value")!.compareDocumentPosition(look) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it("measure and look, described: both, kept apart", () => {
    render(<AgentAnswerPanel result={NDWI_WITH_LOOK_OBSERVED} />);
    const panel = answerPanel();
    expect(panel.querySelector(".result-value")).toHaveTextContent("+0.1466");
    const look = panel.querySelector(".what-i-see")!;
    expect(look).toHaveTextContent(/sandy beach/);
    expect(look).not.toHaveTextContent(/0\.1466/);
  });

  it("no visual request: no 'What I see' at all", () => {
    render(<AgentAnswerPanel result={NDWI_MARINA} />);
    expect(answerPanel().querySelector(".what-i-see")).toBeNull();
  });
});

describe("visual observation rail", () => {
  it("observed", () => {
    render(<AgentObservationPanel evidence={VISUAL_OBSERVED.evidence} result={VISUAL_OBSERVED} />);
    const panel = observationPanel();
    expect(panel).toHaveTextContent("What I see");
    expect(panel).toHaveTextContent(/Described by Local · qwen3-vl:4b-instruct/);
  });

  it("unavailable: the server's own reason", () => {
    render(<AgentObservationPanel evidence={VISUAL_UNAVAILABLE.evidence} result={VISUAL_UNAVAILABLE} />);
    expect(observationPanel()).toHaveTextContent(
      "The satellite image was retrieved, but describing it needs an AI visual model, which is not available here.",
    );
  });

  it("not requested", () => {
    render(<AgentObservationPanel evidence={NDWI_MARINA.evidence} result={NDWI_MARINA} />);
    expect(observationPanel()).toHaveTextContent(
      "No visual description was requested for this question, so no image was sent to a model.",
    );
  });

  it("nothing ran: never claims a refused description was not requested", () => {
    // Found live: "Describe the radar image of ..." is refused before anything
    // runs, and the rail said "No visual description was requested".
    render(<AgentObservationPanel evidence={UNSUPPORTED_SHIPS.evidence} result={UNSUPPORTED_SHIPS} />);
    const panel = observationPanel();
    expect(panel).toHaveTextContent("Nothing ran for this question, so no image was sent to a model.");
    expect(panel).not.toHaveTextContent(/requested/);
  });
});

describe("visual evidence export", () => {
  it("records what became of the description, with its scene", () => {
    expect(buildEvidenceReport(VISUAL_UNAVAILABLE, null, "q").visual).toMatchObject({
      status: "unavailable",
      scene_id: "S2B_44PMV_20250104_0_L2A",
      acquired: "2025-01-04",
    });
    expect(buildEvidenceReport(NDWI_MARINA, null, "q").visual).toBeNull();
  });
});
