# Map feature

Holds the interactive map UI. Currently a dependency-free placeholder.

**Next phase**

1. `npm i maplibre-gl`
2. Initialise a `maplibre-gl.Map` inside a `useEffect` in `MapPanel.tsx`, mounting
   into `containerRef`.
3. Add layer management for Sentinel-1 / Sentinel-2 tiles and change-detection
   overlays, fed by the backend `map` service.
