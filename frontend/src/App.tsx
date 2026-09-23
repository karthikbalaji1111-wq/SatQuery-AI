import { useState } from "react";

import "./App.css";

import type { AiProvider, ImageryResponse, NdwiOverlay } from "./api/types";
import { BackendStatus } from "./components/BackendStatus";
import { ModelSelector } from "./components/ModelSelector";
import {
  AgentAnswerPanel,
  AgentEvidencePanel,
  AgentObservationPanel,
  AgentPipeline,
  AgentQueryCard,
} from "./features/agent/AgentPanel";
import { useAgentRun } from "./features/agent/agentRun";
import { runContextFrom } from "./features/agent/derive";
import type { MapAoi } from "./features/map/footprint";
import {
  buildEvidenceReport,
  evidenceFilename,
} from "./features/agent/evidenceReport";
import { MapPanel } from "./features/map/MapPanel";
import {
  LocatorPlaceholder,
  WorkspaceIntro,
} from "./features/workspace/WorkspaceIntro";
import { QueryPanel } from "./features/query/QueryPanel";
import type { ManualEvidence } from "./features/query/QueryPanel";

/** A satellite mark. Inline so the shell needs no icon dependency. */
function BrandGlyph() {
  return (
    <span className="brand-glyph" aria-hidden="true">
      <svg viewBox="0 0 24 24" fill="none" strokeWidth="1.8">
        <circle cx="12" cy="12" r="3.2" />
        <ellipse cx="12" cy="12" rx="10" ry="4.6" transform="rotate(-28 12 12)" />
      </svg>
    </span>
  );
}

/**
 * The workspace: a three-column geospatial instrument.
 *
 *   LEFT    query configuration - location, window, sensor, task, candidates
 *   CENTRE  the query, the pipeline, the satellite imagery, the evidence
 *   RIGHT   footprint context, the grounded answer, the model's observation
 *
 * The columns are real containers rather than bands of one grid, because they
 * carry different vertical rhythms. The agent's state lives in `useAgentRun`
 * so its regions can sit in two different columns while reading from one run.
 */
export function App() {
  // The most recent scene preview, held here only so the map can draw it.
  // Both panels produce one; this is the result travelling upward, never a
  // second execution path.
  const [imagery, setImagery] = useState<ImageryResponse | null>(null);
  // Held beside the RGB preview rather than replacing it: they come from
  // different requests and different grids, and the map positions each by its
  // own corners.
  const [ndwi, setNdwi] = useState<NdwiOverlay | null>(null);
  const [change, setChange] = useState<NdwiOverlay | null>(null);
  // The extent a run resolved, so the viewport can show where an analysis
  // happened even when it requested no raster preview.
  const [aoi, setAoi] = useState<MapAoi | null>(null);
  // What the manual configuration path established. Accumulated rather than
  // replaced: the scene arrives with the preview and the measurements arrive
  // with the analysis, and the evidence panel wants both.
  const [manualEpoch, setManualEpoch] = useState(0);
  const [manual, setManual] = useState<ManualEvidence | null>(null);
  // Which backend answers the next run's visual step. Held here and read only
  // when a request is made, so switching it leaves the scene, the evidence,
  // the map and the query exactly as they are.
  // Which backend and model answer the next run's visual step. Held here and
  // read only when a request is made, so switching leaves the scene, the
  // evidence, the map and the query exactly as they are.
  const [inference, setInference] = useState<{
    provider: string;
    model: string;
  } | null>(null);
  // What the server said its default is. Used ONLY to attribute a finished
  // run; the request body still omits an untouched default so the server
  // remains the authority on what that default is.
  const [inferenceDefaults, setInferenceDefaults] = useState<{
    provider: string;
    model: string;
  } | null>(null);

  // Whether a run has resolved anything to place. The map is always mounted
  // (map-first); this only decides whether the intro still sits over it.
  const hasGeography = Boolean(imagery || aoi || ndwi || change);

  function mergeManual(next: ManualEvidence | null) {
    setManual((current) =>
      next === null || current === null
        ? next
        : {
            scene: next.scene ?? current.scene,
            imagery: next.imagery ?? current.imagery,
            sar_backscatter: next.sar_backscatter ?? current.sar_backscatter,
            measurements: next.measurements.length
              ? next.measurements
              : current.measurements,
            // Provenance accumulates the same way the visible evidence does:
            // the intent and execution arrive with the run, the analysis with
            // its result, and the export needs all three. Listed explicitly
            // because this merge builds a new object - a field omitted here is
            // a field silently dropped from the record.
            intent: next.intent ?? current.intent,
            execution: next.execution ?? current.execution,
            analysis: next.analysis ?? current.analysis,
          },
    );
  }

  const run = useAgentRun({
    onStart: () => {
      setManual(null);
      // Unmount the manual request owner so late responses cannot repaint.
      setManualEpoch((epoch) => epoch + 1);
    },
    provider: (inference?.provider ?? null) as AiProvider | null,
    model: inference?.model ?? null,
    // What to ATTRIBUTE the result to when the request named nothing: the
    // deployment default the server itself reported. Never sent.
    defaultProvider: inferenceDefaults?.provider ?? null,
    defaultModel: inferenceDefaults?.model ?? null,
    onImagery: setImagery,
    onNdwi: setNdwi,
    onChange: setChange,
    onAoi: setAoi,
  });

  /**
   * A completed result is on screen - from EITHER workflow.
   *
   * Export and the answer panel were gated on `run.result`, the agent's own
   * outcome, so a successful manual analysis left the rail saying "Run an
   * analysis" with its export disabled while its measurements sat in the panel
   * beside it. The two workflows stay separate; what they share is that both
   * can finish, and the presentation of a finished result should not depend on
   * which one did.
   */
  const manualComplete = Boolean(
    manual &&
      (manual.measurements.length > 0 ||
        manual.scene !== null ||
        manual.sar_backscatter),
  );
  const hasCompletedResult = run.result !== null || manualComplete;

  /**
   * Hand the reader an auditable record of the run.
   *
   * Assembled entirely from state already on screen, so it asks the server for
   * nothing and cannot disagree with what was displayed. It carries the
   * unflattering parts too - a withheld answer, failed grounding checks, a
   * window whose imagery could not be retrieved - because a report that only
   * records successes is not an audit.
   */
  function exportEvidence() {
    // `run.asked`, never `run.question`: the box stays editable after a run,
    // and the report must describe the question that produced these
    // measurements. Falls back to the draft only when nothing has run yet,
    // where the two are necessarily the same.
    const report = buildEvidenceReport(
      run.result,
      manual,
      run.asked?.question ?? run.question,
    );
    const blob = new Blob([JSON.stringify(report, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = evidenceFilename(
      runContextFrom(run.result)?.location ?? null,
    );
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="app">
      <header className="app-header">
        <BrandGlyph />
        {/* The accessible name stays "SatQuery": the AI mark is decorative and
            hidden, so assistive tech and tests read one product name. */}
        <h1>
          SatQuery&#8202;
          <span className="brand-mark" aria-hidden="true">
            AI
          </span>
        </h1>
        <div className="rule-v" aria-hidden="true" />
        <p>Interactive Remote Sensing Analysis</p>
        <div className="header-spacer" />
        {/* Capabilities the backend actually exposes, not navigation to pages
            that do not exist. */}
        <span className="header-caps">S2 · S1</span>
        {/* Selection only - no key reaches the browser. It changes the
            inference backend for the next run and nothing else. */}
        <ModelSelector
          value={inference?.model ?? null}
          onChange={setInference}
          onDefaults={setInferenceDefaults}
        />
        <BackendStatus />
      </header>

      <main className="app-main">
        <aside className="col-config" aria-label="Query configuration">
          <QueryPanel
            key={manualEpoch}
            onStart={() => {
              run.clear();
              setManual(null);
            }}
            onImagery={setImagery}
            onNdwi={setNdwi}
            onChange={setChange}
            onAoi={setAoi}
            runContext={runContextFrom(run.result)}
            onEvidence={mergeManual}
          />
        </aside>

        <div className="col-centre" data-empty={!hasGeography && !run.busy}>
          <AgentQueryCard run={run} />
          <AgentPipeline run={run} />
          <div className="imagery-cell">
            {/* Map-first: the real MapLibre map is the workspace from the first
                paint, opening on a neutral world view rather than on a place
                nobody asked about. Until a query resolves geography, the intro
                is layered over it - never in place of it. */}
            <MapPanel
              aoi={aoi}
              imagery={imagery}
              ndwi={ndwi}
              change={change}
              busy={run.busy}
            />
            {!hasGeography && <WorkspaceIntro busy={run.busy} overlay />}
          </div>
          <AgentEvidencePanel
            evidence={run.result?.evidence ?? null}
            manual={manual}
            bbox={aoi}
            busy={run.busy}
          />
        </div>

        <div className="col-rail">
          <section className="panel footprint-panel" aria-label="Footprint">
            <header className="panel-head">
              <span className="eyebrow">Footprint</span>
              <span className="panel-source">WGS 84 · EPSG:4326</span>
            </header>
            <div className="locator-cell">
              {aoi ? (
                <MapPanel aoi={aoi} variant="locator" />
              ) : (
                <LocatorPlaceholder />
              )}
            </div>
          </section>

          <AgentAnswerPanel
            result={run.result}
            asked={run.asked}
            manualComplete={manualComplete}
            busy={run.busy}
          />
          <AgentObservationPanel
            evidence={run.result?.evidence ?? null}
            result={run.result}
            busy={run.busy}
            onExport={hasCompletedResult ? exportEvidence : undefined}
          />
        </div>
      </main>
    </div>
  );
}
