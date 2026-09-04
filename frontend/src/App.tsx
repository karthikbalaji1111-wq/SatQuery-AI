import { useState } from "react";

import "./App.css";

import type { ImageryResponse, NdwiOverlay } from "./api/types";
import { BackendStatus } from "./components/BackendStatus";
import { AgentPanel } from "./features/agent/AgentPanel";
import { MapPanel } from "./features/map/MapPanel";
import { QueryPanel } from "./features/query/QueryPanel";

export function App() {
  // The most recent scene preview, held here only so the map can draw it.
  // QueryPanel still owns the request; this is the result travelling upward,
  // not a second execution path.
  const [imagery, setImagery] = useState<ImageryResponse | null>(null);
  // The NDWI overlay a query produced. Held beside the RGB preview rather than
  // replacing it: they come from different requests and different grids, and
  // the map positions each by its own corners.
  const [ndwi, setNdwi] = useState<NdwiOverlay | null>(null);
  // The temporal change overlay, held beside the other two: a third raster on
  // a third grid, positioned by its own corners.
  const [change, setChange] = useState<NdwiOverlay | null>(null);

  return (
    <div className="app">
      <header className="app-header">
        <h1>SatQuery</h1>
        <p>Natural-language satellite query platform — foundation build.</p>
        <BackendStatus />
      </header>

      <main className="app-main">
        <AgentPanel />
        <QueryPanel
          onImagery={setImagery}
          onNdwi={setNdwi}
          onChange={setChange}
        />
        <MapPanel imagery={imagery} ndwi={ndwi} change={change} />
      </main>

      <footer className="app-footer">
        <span>SIH 2026 · Problem Statement 26167</span>
      </footer>
    </div>
  );
}
