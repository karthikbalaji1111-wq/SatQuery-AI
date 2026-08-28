import "./App.css";

import { BackendStatus } from "./components/BackendStatus";
import { MapPanel } from "./features/map/MapPanel";
import { QueryPanel } from "./features/query/QueryPanel";

export function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>SatQuery</h1>
        <p>Natural-language satellite query platform — foundation build.</p>
        <BackendStatus />
      </header>

      <main className="app-main">
        <QueryPanel />
        <MapPanel />
      </main>

      <footer className="app-footer">
        <span>SIH 2026 · Problem Statement 26167</span>
      </footer>
    </div>
  );
}
