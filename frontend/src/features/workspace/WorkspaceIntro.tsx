/**
 * What the analysis workspace shows before it has any geography to show.
 *
 * The map used to fill this space from the first paint, centred on a
 * hardcoded coordinate. That was wrong twice over: it printed a city the user
 * had not asked about - which reads as a pre-loaded demo rather than a live
 * system - and it spent the largest surface on the page saying nothing. A
 * basemap is context for a result; with no result it is decoration.
 *
 * So this stands in until there is something real to place: the same frame,
 * carrying the one thing a first-time reader actually needs, which is what
 * this system does and what it will put here.
 */

const STAGES: { label: string; detail: string }[] = [
  { label: "Question", detail: "asked in plain language" },
  { label: "Location", detail: "geocoded to an area of interest" },
  { label: "Scenes", detail: "discovered in the Sentinel catalog" },
  { label: "Measurement", detail: "computed from the raster itself" },
  { label: "Answer", detail: "checked against that evidence" },
];

export function WorkspaceIntro({ busy = false }: { busy?: boolean }) {
  if (busy) {
    // The pipeline strip above already reports which stage is running, so this
    // says only that the frame is spoken for - two live progress accounts of
    // one request disagree the moment either lags.
    return (
      <section
        className="workspace-intro"
        data-busy="true"
        aria-labelledby="map-heading"
      >
        <h2 id="map-heading" className="sr-only">
          Satellite scene
        </h2>
        <p className="intro-lead">Running the analysis…</p>
        <p className="intro-note">
          Imagery and evidence appear here as each stage returns.
        </p>
      </section>
    );
  }

  return (
    // Labelled the same whether it holds the scene or stands in for it: this
    // frame IS the satellite-scene region, and renaming it by content would
    // move the landmark out from under anyone navigating by headings.
    <section className="workspace-intro" aria-labelledby="map-heading">
      <h2 id="map-heading" className="sr-only">
        Satellite scene
      </h2>
      <p className="intro-lead">
        Ask a question about a place and a time. SatQuery finds the satellite
        scenes that answer it, measures them, and shows its working.
      </p>

      <ol className="intro-stages">
        {STAGES.map((stage) => (
          <li key={stage.label}>
            <span className="intro-stage-label">{stage.label}</span>
            <span className="intro-stage-detail">{stage.detail}</span>
          </li>
        ))}
      </ol>

      <p className="intro-note">
        The scene imagery appears in this frame. Every number beside it is
        computed from the pixels, never written by a model.
      </p>
    </section>
  );
}

/**
 * The right rail's locator, before an area of interest exists.
 *
 * Same reasoning as above: a locator that has not been told where to look
 * should say so, not point at a default.
 */
export function LocatorPlaceholder() {
  return (
    <div className="locator-placeholder">
      <p className="hint">
        The area of interest appears here once a query resolves a location.
      </p>
    </div>
  );
}
