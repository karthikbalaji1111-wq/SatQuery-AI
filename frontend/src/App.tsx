import { useState } from "react";

import "./App.css";

import type { ImageryResponse } from "./api/types";
import { BackendStatus } from "./components/BackendStatus";
import { AgentPanel } from "./features/agent/AgentPanel";
import { MapPanel } from "./features/map/MapPanel";
import { QueryPanel } from "./features/query/QueryPanel";

export function App() {
  // The most recent scene preview, held here only so the map can draw it.
  // QueryPanel still owns the request; this is the result travelling upward,
  // not a second execution path.
  const [imagery, setImagery] = useState<ImageryResponse | null>(null);

  return (
    <div className="app">
      <header className="app-header">
        <h1>SatQuery</h1>
        <p>Natural-language satellite query platform — foundation build.</p>
        <BackendStatus />
      </header>

      <main className="app-main">
        <AgentPanel />
        <QueryPanel onImagery={setImagery} />
        <MapPanel imagery={imagery} />
      </main>

      <footer className="app-footer">
        <span>SIH 2026 · Problem Statement 26167</span>
      </footer>
    </div>
  );
}
