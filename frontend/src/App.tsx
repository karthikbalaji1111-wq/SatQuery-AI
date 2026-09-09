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

  // A basemap is context for a result. With nothing to place, mounting one
  // would print a default location the user never asked about, so the frames
  // stay empty until something real can go in them.
  const hasGeography = Boolean(imagery || aoi || ndwi || change);

  function mergeManual(next: ManualEvidence | null) {
    setManual((current) =>
      next === null || current === null
        ? next
        : {
            scene: next.scene ?? current.scene,
            imagery: next.imagery ?? current.imagery,
            measurements: next.measurements.length
              ? next.measurements
              : current.measurements,
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
    onImagery: setImagery,
    onNdwi: setNdwi,
    onChange: setChange,
    onAoi: setAoi,
  });

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
    const report = buildEvidenceReport(run.result, manual, run.question);
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
            {hasGeography ? (
              <MapPanel
                aoi={aoi}
                imagery={imagery}
                ndwi={ndwi}
                change={change}
                busy={run.busy}
              />
            ) : (
              <WorkspaceIntro busy={run.busy} />
            )}
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

          <AgentAnswerPanel result={run.result} busy={run.busy} />
          <AgentObservationPanel
            evidence={run.result?.evidence ?? null}
            result={run.result}
            busy={run.busy}
            onExport={run.result === null ? undefined : exportEvidence}
          />
        </div>
      </main>
    </div>
  );
}
