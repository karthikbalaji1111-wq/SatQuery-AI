import { useRef } from "react";

/**
 * Placeholder for the interactive map. A MapLibre GL instance will mount into
 * `containerRef` in a later phase (add `maplibre-gl` then initialise inside a
 * `useEffect`). Kept dependency-free for now.
 */
export function MapPanel() {
  const containerRef = useRef<HTMLDivElement>(null);

  return (
    <section className="panel" aria-labelledby="map-heading">
      <h2 id="map-heading">Map</h2>
      <div ref={containerRef} className="map-container" role="img" aria-label="Map placeholder">
        <span>MapLibre view will render here.</span>
      </div>
    </section>
  );
}
