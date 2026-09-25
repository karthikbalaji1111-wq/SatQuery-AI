import {
  act,
  fireEvent,
  render,
  renderHook,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { AgentResult } from "../../api/types";
import { AgentAnswerPanel, AgentEvidencePanel, AgentPanel } from "./AgentPanel";
import { thresholdNote } from "./resultModel";
import { useAgentRun } from "./agentRun";
import {
  AREA_TOO_LARGE_CHENNAI,
  CLARIFY_CHENNAI,
  LOCATION_UNAVAILABLE,
  NDBI_AMEERPET,
  NDVI_CUBBON,
  NDVI_LALBAGH_STOP,
  NDWI_MARINA,
  NO_SCENES,
  NOT_FOUND,
  REFUSED_BY_RADIOMETRY,
  SAHARA_POINT,
  SAR_MARINA,
  TEMPORAL_MARINA,
  UNSUPPORTED_SHIPS,
} from "./m6Fixtures";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

function respond(body: unknown): Promise<Response> {
  return Promise.resolve({
    ok: true,
    status: 200,
    text: () => Promise.resolve(JSON.stringify(body)),
  } as Response);
}

/** A fetch whose answers are released by the test, one per call, in order. */
function controlledFetch() {
  const pending: ((body: unknown) => void)[] = [];
  const fn = vi.fn().mockImplementation(
    () =>
      new Promise<Response>((resolve) => {
        pending.push((body) => void respond(body).then(resolve));
      }),
  );
  vi.stubGlobal("fetch", fn);
  return {
    fn,
    async answer(index: number, body: unknown) {
      await act(async () => {
        pending[index](body);
      });
    },
  };
}

function answerPanel(): HTMLElement {
  return screen
    .getByRole("heading", { name: "Analysis result" })
    .closest("section") as HTMLElement;
}

function pipeline(): HTMLElement {
  return screen.getByRole("heading", { name: "Pipeline" }).closest("section") as HTMLElement;
}

function context(panel: HTMLElement): Record<string, string> {
  return Object.fromEntries(
    [...panel.querySelectorAll(".result-context > div")].map((row) => [
      row.querySelector("dt")?.textContent ?? "",
      row.querySelector("dd")?.textContent ?? "",
    ]),
  );
}

function typeAndRun(question: string) {
  const input = screen.getByLabelText(/question/i);
  fireEvent.change(input, { target: { value: question } });
  fireEvent.keyDown(input, { key: "Enter" });
}

// =========================================================================== //
// The result, by operation - the backend's own values, laid out as what they are
// =========================================================================== //

describe("M6 result card", () => {
  it("NDVI: the value, then where, when and how clean", () => {
    render(<AgentAnswerPanel result={NDVI_CUBBON} />);
    const panel = answerPanel();

    expect(within(panel).getByText("Result")).toHaveClass("outcome-chip");
    expect(panel.querySelector(".result-op")).toHaveTextContent("Vegetation index · NDVI");
    expect(panel.querySelector(".result-value")).toHaveTextContent("+0.5204");
    // Updated deliberately: plain words; the range is in Technical details.
    expect(panel.querySelector(".result-sub")).toHaveTextContent(
      "Average over 16,988 usable pixels",
    );
    // Updated deliberately (layperson first): everyday labels; the scene id is
    // the title of "Satellite image" and stays in Technical details.
    expect(context(panel)).toEqual({
      Location: "Cubbon Park, Bengaluru",
      Matched: "Cubbon Park, Sampangirama Nagar (leisure · park)",
      Period: "December 2024",
      "Satellite image": "Sentinel-2B",
      Date: "8 December 2024",
      "Images found": "7",
      "Usable pixels": "99.5% · 16,988 of 17,080",
    });
    // The grounded sentence stays - in Technical details, with its checks.
    const sentence = within(panel).getByText(NDVI_CUBBON.answer!);
    expect(sentence).toHaveAttribute("data-role", "summary");
    expect(sentence.closest("details.technical-details")).not.toBeNull();
  });

  it("NDWI: the water index, not a water classification", () => {
    render(<AgentAnswerPanel result={NDWI_MARINA} />);
    const panel = answerPanel();
    expect(panel.querySelector(".result-op")).toHaveTextContent("Water index · NDWI");
    expect(panel.querySelector(".result-value")).toHaveTextContent("+0.1466");
    expect(context(panel)).toMatchObject({ Location: "Marina Beach, Chennai", Period: "January 2025" });
    expect(panel).not.toHaveTextContent(/water body|flood|detected water/i);
  });

  it("NDBI: a negative mean keeps its sign", () => {
    render(<AgentAnswerPanel result={NDBI_AMEERPET} />);
    const panel = answerPanel();
    expect(panel.querySelector(".result-op")).toHaveTextContent("Built-up index · NDBI");
    expect(panel.querySelector(".result-value")).toHaveTextContent("-0.0248");
    expect(panel.querySelector(".result-sub")).toHaveTextContent("196,620 usable pixels");
  });

  it("SAR: each polarization in decibels, and the provider's product named", () => {
    render(<AgentAnswerPanel result={SAR_MARINA} />);
    const panel = answerPanel();
    expect(panel.querySelector(".result-op")).toHaveTextContent("Radar measurements");
    const values = Object.fromEntries(
      [...panel.querySelectorAll(".result-sar > div")].map((row) => [
        row.querySelector("dt")?.textContent,
        row.querySelector("dd")?.textContent,
      ]),
    );
    // Updated deliberately: plain names on the card; VV, VH, γ⁰ and the
    // provider's terrain correction are named in Technical details.
    expect(values).toEqual({
      "First measurement": "-5.44 dB",
      "Second measurement": "-17.85 dB",
      Difference: "12.41 dB",
    });
    expect(panel.querySelector(".result-sub")).toHaveTextContent("Averages over 33,600 usable pixels");
    const technical = panel.querySelector("details.technical-details") as HTMLElement;
    expect(technical).toHaveTextContent("VV (first measurement)");
    expect(technical).toHaveTextContent("terrain-corrected");
    expect(context(panel)).toMatchObject({
      "Satellite image": "Sentinel-1A (radar)",
      Date: "11 January 2025",
    });
  });

  it("temporal NDWI: earlier period → later period → measured change", () => {
    render(<AgentAnswerPanel result={TEMPORAL_MARINA} />);
    const panel = answerPanel();
    expect(panel.querySelector(".result-op")).toHaveTextContent("Water change · two dates");
    const periods = [...panel.querySelectorAll(".result-periods li")].map((li) => ({
      role: li.querySelector(".period-role")?.textContent,
      when: li.querySelector(".period-when")?.textContent,
      value: li.querySelector(".period-value")?.textContent,
      acquired: li.querySelector(".period-acquired")?.textContent,
    }));
    expect(periods).toEqual([
      { role: "Earlier", when: "January 2024", value: "+0.0266", acquired: "image taken 15 January 2024" },
      { role: "Later", when: "January 2025", value: "+0.1466", acquired: "image taken 4 January 2025" },
    ]);
    const pair = [...panel.querySelectorAll(".metric-pair > div")].map(
      (row) => `${row.querySelector("dt")?.textContent} ${row.querySelector("dd")?.textContent}`,
    );
    // Updated deliberately: one change for a reader; the per-pixel change is
    // stated in How we know and Technical details.
    expect(pair).toEqual(["Change +0.1200"]);
    expect(panel).toHaveTextContent("the average change per pixel was +0.1207");
    // A difference is stated, never explained.
    expect(panel).toHaveTextContent("no cause is inferred");
    expect(panel).not.toHaveTextContent(/because|due to|rain|expanded|flood/i);
    // The periods are in the card; the single-period row would repeat them.
    expect(context(panel)).not.toHaveProperty("Period");
    // Each observation has its own pixels; neither stands for both.
    expect(context(panel)["Usable pixels"]).toBe("99.7% earlier · 99.8% later");
    // What changed, in words, above the numbers.
    expect(panel.querySelector(".result-change")).toHaveTextContent(
      "The water signal was stronger in January 2025 than in January 2024.",
    );
  });
});

// =========================================================================== //
// Eight kinds of result, eight different presentations
// =========================================================================== //

describe("M6 outcome states stay distinct", () => {
  const cases: [string, AgentResult, string, RegExp][] = [
    ["clarification", CLARIFY_CHENNAI, "One more detail", /needs one more detail/],
    ["location not found", NOT_FOUND, "Location not found", /the place could not be found/],
    ["location unavailable", LOCATION_UNAVAILABLE, "Location service unavailable", /Location service temporarily unavailable\./],
    ["area too large", AREA_TOO_LARGE_CHENNAI, "Area too large", /the area is too large to measure/],
    ["a point, not an area", SAHARA_POINT, "A point, not an area", /the place is only a point on the map/],
    ["unsupported", UNSUPPORTED_SHIPS, "Not supported", /SatQuery does not do this/],
    ["no scene matched", NO_SCENES, "No measurement", /No satellite scene matched this place and period/],
    ["validated refusal", REFUSED_BY_RADIOMETRY, "Analysis not computed", /The analysis was not computed\./],
  ];

  it.each(cases)("%s", (_, result, chip, statement) => {
    render(<AgentAnswerPanel result={result} />);
    const panel = answerPanel();
    expect(panel.querySelector(".outcome-chip")).toHaveTextContent(chip);
    expect(within(panel).getByRole("status")).toHaveTextContent(statement);
    // None of them shows a measured value.
    expect(panel.querySelector(".result-value, .result-card")).toBeNull();
  });

  it("no two states share a presentation", () => {
    const seen = new Set<string>();
    for (const [, result] of [...cases, ["success", NDVI_CUBBON] as const]) {
      const { unmount } = render(<AgentAnswerPanel result={result} />);
      const panel = answerPanel();
      seen.add(
        `${panel.querySelector(".outcome-chip")?.textContent}|${
          panel.querySelector("[role=status] .answer-notice-summary, .result-op")?.textContent
        }`,
      );
      unmount();
    }
    expect(seen.size).toBe(cases.length + 1);
  });

  it("an outage and a refusal are never a question back", () => {
    for (const result of [LOCATION_UNAVAILABLE, REFUSED_BY_RADIOMETRY, NO_SCENES]) {
      const { unmount } = render(<AgentAnswerPanel result={result} />);
      expect(answerPanel()).not.toHaveTextContent(/needs one more detail/);
      unmount();
    }
  });

  it("a validated refusal keeps the server's own reason, one click away", () => {
    render(<AgentAnswerPanel result={REFUSED_BY_RADIOMETRY} />);
    const why = answerPanel().querySelector("details.answer-notice-reason") as HTMLDetailsElement;
    expect(why.querySelector("summary")).toHaveTextContent("Why");
    expect(why).toHaveTextContent(/Radiometric validation failed \(radiometric_undetermined\)/);
    expect(why.open).toBe(false);
  });

  it("the genuine evidence verdict keeps its own wording", () => {
    render(<AgentAnswerPanel result={NO_SCENES} />);
    expect(screen.getByText("Insufficient evidence to answer the question.")).toHaveAttribute(
      "data-role",
      "verdict",
    );
  });
});

// =========================================================================== //
// Evidence: sources, checks and observations, technical detail folded not lost
// =========================================================================== //

describe("M6 evidence panel", () => {
  function evidencePanel(): HTMLElement {
    return screen
      .getByRole("heading", { name: "Deterministic evidence" })
      .closest("section") as HTMLElement;
  }

  it("states the source and the selection rule the server applied", () => {
    render(<AgentEvidencePanel evidence={NDVI_CUBBON.evidence} />);
    const panel = evidencePanel();
    expect(within(panel).getByText("Source catalog").nextSibling).toHaveTextContent(
      "earth-search.aws.element84.com",
    );
    expect(within(panel).getByText("Scenes matched").nextSibling).toHaveTextContent(
      "7 · lowest cloud cover selected",
    );
  });

  it("names the Sentinel-1 rule for a radar scene, not the optical one", () => {
    render(<AgentEvidencePanel evidence={SAR_MARINA.evidence} />);
    expect(within(evidencePanel()).getByText("Scenes matched").nextSibling).toHaveTextContent(
      "2 · earliest acquisition selected",
    );
  });

  it("shows the quality and validation checks that ran", () => {
    render(<AgentEvidencePanel evidence={NDVI_CUBBON.evidence} />);
    const checks = within(evidencePanel()).getByRole("region", { name: "Quality and validation" });
    const rows = [...checks.querySelectorAll("li")].map(
      (li) => `${li.querySelector(".check-name")?.textContent} = ${li.querySelector(".check-value")?.textContent}`,
    );
    expect(rows).toEqual([
      "Pixel quality · NDVI = 99.5% usable — 16,988 of 17,080 pixels; 92 masked by the Scene Classification Layer (0 cloud, 0 shadow)",
      "Radiometry · S2B_43PGQ_20241208_0_L2A = verified · some metadata not published · baseline 05.11",
      "Geometry · ndvi = grid verified · EPSG:32643 · 10 m · 122 × 140 px",
    ]);
  });

  it("lists both observations of a comparison, each with its own mean", () => {
    render(<AgentEvidencePanel evidence={TEMPORAL_MARINA.evidence} />);
    const pair = within(evidencePanel()).getByRole("region", { name: "Observations compared" });
    expect(pair).toHaveTextContent("Earlier · baseline");
    expect(pair).toHaveTextContent(
      "S2A_44PMV_20240115_0_L2A · 2024-01-15 05:15:05 UTC · 14.3% cloud · NDWI mean +0.0266 over 33,496 valid pixels",
    );
    expect(pair).toHaveTextContent("Later · target");
    expect(pair).toHaveTextContent(
      "S2B_44PMV_20250104_0_L2A · 2025-01-04 05:15:13 UTC · 14.2% cloud · NDWI mean +0.1466 over 33,524 valid pixels",
    );
  });

  it("reports a comparison's checks for both scenes and for the pair", () => {
    render(<AgentEvidencePanel evidence={TEMPORAL_MARINA.evidence} />);
    const checks = within(evidencePanel()).getByRole("region", { name: "Quality and validation" });
    const names = [...checks.querySelectorAll(".check-name")].map((node) => node.textContent);
    expect(names).toEqual([
      "Pixel quality · NDWI · baseline",
      "Pixel quality · NDWI · target",
      "Radiometry · S2A_44PMV_20240115_0_L2A",
      "Radiometry · S2B_44PMV_20250104_0_L2A",
      "Geometry · ndwi",
      "Geometry · ndwi",
      "Geometry · temporal ndwi pair",
    ]);
    expect(checks).toHaveTextContent("baseline 05.10");
    expect(checks).toHaveTextContent("baseline 05.11");
  });

  it("never sets one observation's mean alone as 'the' NDWI of a comparison", () => {
    // Both observations publish "ndwi_mean"; grouping them by name set the
    // EARLIER one under a bare "NDWI mean" heading.
    render(<AgentEvidencePanel evidence={TEMPORAL_MARINA.evidence} />);
    expect(evidencePanel().querySelector(".index-readout")).toBeNull();
  });

  it("keeps every evidence id, folded under technical detail", () => {
    render(<AgentEvidencePanel evidence={NDVI_CUBBON.evidence} />);
    const technical = evidencePanel().querySelector("details.evidence-technical") as HTMLDetailsElement;
    expect(technical.open).toBe(false);
    expect(technical.querySelector("summary")).toHaveTextContent("Technical evidence · 6 items");
    for (const item of NDVI_CUBBON.evidence.items) {
      expect(within(technical).getByText(item.id)).toBeInTheDocument();
    }
  });
});

// =========================================================================== //
// The workspace in motion
// =========================================================================== //

describe("M6 query flow", () => {
  it("Enter runs the question; Shift+Enter does not", async () => {
    const fetchMock = controlledFetch();
    render(<AgentPanel />);
    const input = screen.getByLabelText(/question/i);
    fireEvent.change(input, { target: { value: "Show water around Dal Lake in January 2025" } });

    fireEvent.keyDown(input, { key: "Enter", shiftKey: true });
    expect(fetchMock.fn).not.toHaveBeenCalled();

    fireEvent.keyDown(input, { key: "Enter" });
    await waitFor(() => expect(fetchMock.fn).toHaveBeenCalledTimes(1));
  });

  it("states that the run is in flight - no stage is claimed until it returns", async () => {
    const fetchMock = controlledFetch();
    render(<AgentPanel />);
    typeAndRun("Show vegetation around Cubbon Park, Bengaluru in December 2024");

    await waitFor(() => expect(within(pipeline()).getByRole("status")).toHaveTextContent(
      /Understanding the question, resolving the location, finding satellite scenes and measuring/,
    ));
    expect(within(pipeline()).getByText(/^\d+ s$/)).toBeInTheDocument();
    expect(pipeline().querySelector(".run-stages")).toBeNull();

    await fetchMock.answer(0, NDVI_CUBBON);

    const stages = [...pipeline().querySelectorAll(".run-stages .run-stage-name")].map(
      (node) => node.textContent,
    );
    expect(stages).toEqual([
      "Understand question",
      "Resolve location",
      "Find satellite images",
      "Check the images",
      "Run analysis",
      "Check answer",
    ]);
    expect(within(pipeline()).getByText("Complete")).toBeInTheDocument();
  });

  it("refuses a second submission while one is running - Enter, button or example", async () => {
    const fetchMock = controlledFetch();
    render(<AgentPanel />);
    typeAndRun("Show vegetation around Cubbon Park, Bengaluru in December 2024");
    await waitFor(() => expect(fetchMock.fn).toHaveBeenCalledTimes(1));

    const input = screen.getByLabelText(/question/i);
    fireEvent.keyDown(input, { key: "Enter" });
    fireEvent.click(screen.getByRole("button", { name: /running/i }));
    const example = screen.getByRole("button", { name: "Water index" });
    expect(example).toBeDisabled();
    fireEvent.click(example);

    await act(async () => {});
    expect(fetchMock.fn).toHaveBeenCalledTimes(1);
  });

  it("hands the map the resolved area and the selected scene", async () => {
    const onAoi = vi.fn();
    const fetchMock = controlledFetch();
    render(<AgentPanel onAoi={onAoi} />);
    typeAndRun("Show vegetation around Cubbon Park, Bengaluru in December 2024");
    await fetchMock.answer(0, NDVI_CUBBON);

    expect(onAoi).toHaveBeenLastCalledWith(
      expect.objectContaining({
        west: 77.5879274,
        south: 12.9679621,
        east: 77.5989019,
        north: 12.9803864,
        scene_id: "S2B_43PGQ_20241208_0_L2A",
      }),
    );
  });

  it("keeps the previous result, marked as previous, until the new one replaces it", async () => {
    const fetchMock = controlledFetch();
    render(<AgentPanel />);
    typeAndRun("Show vegetation around Cubbon Park, Bengaluru in December 2024");
    await fetchMock.answer(0, NDVI_CUBBON);
    expect(answerPanel().querySelector(".result-value")).toHaveTextContent("+0.5204");

    typeAndRun("Show water around Marina Beach, Chennai in January 2025");
    await waitFor(() => expect(answerPanel()).toHaveAttribute("data-stale", "true"));
    expect(answerPanel()).toHaveTextContent("Previous result — the new question is still running.");
    expect(answerPanel().querySelector(".result-value")).toHaveTextContent("+0.5204");

    await fetchMock.answer(1, NDWI_MARINA);
    expect(answerPanel()).not.toHaveAttribute("data-stale");
    expect(answerPanel().querySelector(".result-value")).toHaveTextContent("+0.1466");
    expect(answerPanel()).not.toHaveTextContent("+0.5204");
  });

  it("the latest question owns the screen: a late answer to an abandoned one is dropped", async () => {
    // Supersession is reached by another workspace panel retiring the run
    // (clear), since the query box itself refuses a second submission while
    // one is in flight - so it is exercised on the hook directly.
    const fetchMock = controlledFetch();
    const { result } = renderHook(() => useAgentRun());

    await act(async () => {
      void result.current.ask("Show vegetation around Cubbon Park, Bengaluru in December 2024");
    });
    act(() => result.current.clear());
    await act(async () => {
      void result.current.ask("Show water around Marina Beach, Chennai in January 2025");
    });
    expect(fetchMock.fn).toHaveBeenCalledTimes(2);

    await fetchMock.answer(1, NDWI_MARINA);
    await fetchMock.answer(0, NDVI_CUBBON); // the abandoned run lands last
    expect(result.current.result?.answer).toBe(NDWI_MARINA.answer);
    expect(result.current.displayed?.answer).toBe(NDWI_MARINA.answer);
    expect(result.current.stale).toBe(false);
  });

  it("a failed new run clears the previous result instead of leaving it as an answer", async () => {
    const fetchMock = vi.fn()
      .mockImplementationOnce(() => respond(NDVI_CUBBON))
      .mockImplementationOnce(() => Promise.reject(new Error("offline")));
    vi.stubGlobal("fetch", fetchMock);
    render(<AgentPanel />);
    typeAndRun("Show vegetation around Cubbon Park, Bengaluru in December 2024");
    await waitFor(() => expect(answerPanel().querySelector(".result-value")).not.toBeNull());

    typeAndRun("Show water around Marina Beach, Chennai in January 2025");
    await waitFor(() => expect(screen.getByRole("alert")).toBeInTheDocument());
    expect(answerPanel().querySelector(".result-value")).toBeNull();
    expect(answerPanel()).not.toHaveTextContent("+0.5204");
  });
});

// =========================================================================== //
// Final demo audit: a threshold share names its threshold and its index
// =========================================================================== //

describe("audit: threshold shares are labelled as index thresholds", () => {
  it("states the index and the value from the measurement's own name", () => {
    expect(
      thresholdNote({ name: "ndwi_percent_above_index_threshold_0.3", value: 14.118, unit: "%" }),
    ).toBe("14.1% of valid pixels with NDWI > 0.3");
    expect(
      thresholdNote({ name: "ndwi_percent_above_index_threshold_-0.05", value: 50, unit: "%" }),
    ).toBe("50.0% of valid pixels with NDWI > -0.05");
  });

  it("never falls back to a bare 'above threshold' or a class name", () => {
    const note = thresholdNote({ name: "ndwi_share_something", value: 3.2, unit: "%" });
    expect(note).toBe("3.2% of valid pixels above the index threshold");
    expect(note).not.toMatch(/water|flood|built|vegetation/i);
  });

  it("renders it in the evidence readout", () => {
    const evidence = {
      ...NDWI_MARINA.evidence,
      items: [
        ...NDWI_MARINA.evidence.items,
        {
          id: "ndwi.ndwi_percent_above_index_threshold_0.3",
          source: "ndwi",
          measurement: { name: "ndwi_percent_above_index_threshold_0.3", value: 45.49, unit: "%" },
          text: null,
          produced_by: "analysis",
          visual: null,
        },
      ],
    } as unknown as typeof NDWI_MARINA.evidence;
    render(<AgentEvidencePanel evidence={evidence} />);
    expect(screen.getByText("45.5% of valid pixels with NDWI > 0.3")).toBeInTheDocument();
    expect(screen.queryByText(/above\s+threshold/)).toBeNull();
  });
});

// =========================================================================== //
// Final demo audit: the result names what the geocoder actually matched
// =========================================================================== //

describe("audit: the geocoder's match is shown beside the typed place", () => {
  it("a place that matched a point of interest says so - no silent substitution", () => {
    // Live: "Lalbagh, Bengaluru" matched a railway stop; the NDVI covered 4
    // pixels around it. Both facts are now on screen next to the number.
    render(<AgentAnswerPanel result={NDVI_LALBAGH_STOP} />);
    const panel = answerPanel();
    expect(context(panel)).toMatchObject({
      Location: "Lalbagh, Bengaluru",
      Matched: "Lalbagh, Rashtriya Vidyalaya Road (railway · stop)",
    });
    expect(panel.querySelector(".result-sub")).toHaveTextContent("Average over 4 usable pixels");
    const matched = [...panel.querySelectorAll(".result-context dd")].find((dd) =>
      dd.textContent?.startsWith("Lalbagh, Rashtriya"),
    );
    expect(matched).toHaveAttribute("title", expect.stringContaining("Kankanpalya, Ashoka Pillar"));
  });

  it("the evidence records the full match verbatim", () => {
    render(<AgentEvidencePanel evidence={NDVI_LALBAGH_STOP.evidence} />);
    const panel = screen
      .getByRole("heading", { name: "Deterministic evidence" })
      .closest("section") as HTMLElement;
    expect(within(panel).getByText("Geocoder match").nextSibling).toHaveTextContent(
      "Lalbagh, Rashtriya Vidyalaya Road, Kankanpalya, Ashoka Pillar, Bengaluru, Karnataka, India · railway · stop",
    );
  });

  it("an older server without the match shows the typed place only - nothing invented", () => {
    render(<AgentAnswerPanel result={NDWI_MARINA} />);
    expect(context(answerPanel())).not.toHaveProperty("Matched");
  });
});

describe("Layperson first: What this means, How we know, Technical details", () => {
  function section(name: string): HTMLElement | null {
    const heading = within(answerPanel()).queryByRole("heading", { name });
    return heading ? (heading.closest("section") as HTMLElement) : null;
  }
  function technical(): HTMLDetailsElement | null {
    return answerPanel().querySelector("details.technical-details");
  }

  it.each([
    ["NDVI", NDVI_CUBBON, "The selected area shows a positive vegetation signal.", "the vegetation index was +0.5204", "red light and invisible near-infrared light"],
    ["NDWI", NDWI_MARINA, "The selected area shows a positive water signal.", "the water index was +0.1466", "green light with invisible near-infrared light"],
    ["NDBI", NDBI_AMEERPET, "The selected area does not show a positive built-up signal on average.", "the built-up index was -0.0248", "two kinds of invisible infrared light"],
    ["SAR", SAR_MARINA, "The radar image gave two measurements for the selected area: -5.44 dB and -17.85 dB.", "The difference between the two measurements was 12.41 dB", "a radar image rather than a normal camera image"],
    [
      "temporal",
      TEMPORAL_MARINA,
      "The later satellite image shows a stronger water signal than the earlier image.",
      "the water index increased from +0.0266 to +0.1466, a change of +0.1200",
      "We compared satellite images of the same area taken on 15 January 2024 and 4 January 2025",
    ],
  ])("%s: the plain answer, then how we know", (_, result, headline, meaning, how) => {
    render(<AgentAnswerPanel result={result} />);
    const means = section("What this means") as HTMLElement;
    expect(within(means).getByText(headline)).toHaveClass("interpretation-headline");
    expect(means).toHaveTextContent(meaning);
    expect(means.querySelector(".interpretation-caveat")).not.toBeNull();
    expect(section("How we know")).toHaveTextContent(how);
  });

  it("orders the layers: number, meaning, method, context, technical details", () => {
    render(<AgentAnswerPanel result={NDVI_CUBBON} />);
    const panel = answerPanel();
    const order = [
      panel.querySelector(".result-card"),
      section("What this means"),
      section("How we know"),
      panel.querySelector(".result-context"),
      technical(),
    ] as HTMLElement[];
    for (const element of order) expect(element).not.toBeNull();
    for (let i = 1; i < order.length; i += 1) {
      expect(order[i - 1].compareDocumentPosition(order[i]) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    }
  });

  it("keeps the specialist record - folded, not removed", () => {
    render(<AgentAnswerPanel result={NDVI_CUBBON} />);
    const details = technical() as HTMLDetailsElement;
    expect(details.open).toBe(false);
    expect(details.querySelector("summary")).toHaveTextContent("Technical details");
    const rows = Object.fromEntries(
      [...details.querySelectorAll(".technical-list > div")].map((row) => [
        row.querySelector("dt")?.textContent,
        row.querySelector("dd")?.textContent,
      ]),
    );
    expect(rows).toMatchObject({
      Index: "NDVI - Normalized Difference Vegetation Index",
      "NDVI formula": "(NIR − Red) / (NIR + Red) · Sentinel-2 bands B08 and B04",
      "NDVI mean": "+0.5204 (exact 0.5204459203473597)",
      "NDVI valid pixels": "16,988 of 17,080 (92 masked · sentinel-2-scl)",
      "Scene ID": "S2B_43PGQ_20241208_0_L2A",
    });
    expect(rows["Radiometric check"]).toMatch(/^verified/);
    expect(rows["Geometric check"]).toMatch(/^grid verified/);
    // The grounding checks moved with the sentence they checked.
    expect(details.querySelector(".validation-box")).not.toBeNull();
  });

  it("a comparison card names the question it answers, and answers it in words", () => {
    render(<AgentAnswerPanel result={TEMPORAL_MARINA} />);
    const card = answerPanel().querySelector(".result-card") as HTMLElement;
    expect(within(card).getByText("What changed")).toBeInTheDocument();
    expect(card.querySelector(".result-change")).toHaveTextContent(
      "The water signal was stronger in January 2025 than in January 2024.",
    );
    const periods = within(card).getByRole("list", { name: "What changed" });
    expect(periods).toHaveTextContent(/Earlier.*January 2024.*Later.*January 2025/);
    expect(section("What this means")).toHaveTextContent(
      "This shows a change in the satellite measurement. It does not tell us what caused the change.",
    );
  });

  it.each([
    ["a clarification", CLARIFY_CHENNAI],
    ["a place not found", NOT_FOUND],
    ["a location outage", LOCATION_UNAVAILABLE],
    ["an area too large", AREA_TOO_LARGE_CHENNAI],
    ["an unsupported request", UNSUPPORTED_SHIPS],
    ["no measurement", NO_SCENES],
    ["an analysis not computed", REFUSED_BY_RADIOMETRY],
  ])("%s keeps its own wording - no plain-English layers, no technical fold", (_, result) => {
    render(<AgentAnswerPanel result={result} />);
    expect(section("What this means")).toBeNull();
    expect(section("How we know")).toBeNull();
    expect(technical()).toBeNull();
  });
});

describe("Small sample - said under the number, never instead of it", () => {
  function meaning(): HTMLElement {
    return within(answerPanel())
      .getByRole("heading", { name: "What this means" })
      .closest("section") as HTMLElement;
  }

  it("the 4-pixel railway-stop result keeps its number and says how few pixels", () => {
    render(<AgentAnswerPanel result={NDVI_LALBAGH_STOP} />);
    const card = answerPanel().querySelector(".result-card") as HTMLElement;
    expect(within(card).getByText("+0.3728")).toBeInTheDocument();
    const note = within(meaning()).getByRole("note");
    expect(note).toHaveTextContent(
      "Small sample: only 4 usable pixels contributed to this result, so interpret it cautiously.",
    );
    // Inside the interpretation, after its headline - not above the result.
    const headline = meaning().querySelector(".interpretation-headline") as HTMLElement;
    expect(headline.compareDocumentPosition(note) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(card.contains(note)).toBe(false);
  });

  it("a large sample shows no warning", () => {
    render(<AgentAnswerPanel result={NDVI_CUBBON} />);
    expect(within(meaning()).queryByRole("note")).toBeNull();
  });
});

describe("Layperson first: the big result speaks plainly too", () => {
  it("the radar card names no polarization or product code - Technical details does", () => {
    render(<AgentAnswerPanel result={SAR_MARINA} />);
    const card = answerPanel().querySelector(".result-card") as HTMLElement;
    expect(card).not.toHaveTextContent(/\bVV\b|\bVH\b|γ⁰|backscatter/i);
    expect(answerPanel().querySelector("details.technical-details")).toHaveTextContent(/VV .*VH/);
  });

  it("the comparison card names no index code and no paired-pixel term", () => {
    render(<AgentAnswerPanel result={TEMPORAL_MARINA} />);
    const card = answerPanel().querySelector(".result-card") as HTMLElement;
    expect(card).not.toHaveTextContent(/NDWI|paired|valid pixels/i);
    expect(card).toHaveTextContent("Change+0.1200");
  });
});
