import { useEffect, useRef, useState } from "react";
import { Map as MapLibreMap } from "maplibre-gl";

import "maplibre-gl/dist/maplibre-gl.css";

import {
  footprintExtent,
  isValidFootprint,
  SATELLITE_LAYER_ID,
  SATELLITE_SOURCE_ID,
} from "./footprint";
import type { MapFactory, MapImagery, MapLike } from "./footprint";

/**
 * The basemap is an EXTERNAL raster tile service (OpenStreetMap), fetched
 * directly by the viewer's browser. It needs no API key, which is why it is
 * used for the demo; a deployment with real traffic must use a provider whose
 * terms permit it. Overridable so a deployment can point elsewhere.
 */
const DEFAULT_BASEMAP_TILES =
  import.meta.env.VITE_BASEMAP_TILE_URL ??
  "https://tile.openstreetmap.org/{z}/{x}/{y}.png";

const DEFAULT_BASEMAP_ATTRIBUTION =
  '<a href="https://www.openstreetmap.org/copyright">© OpenStreetMap contributors</a>';

/** Chennai, so an empty map still opens somewhere relevant to the demo. */
const INITIAL_CENTER: [number, number] = [80.27, 13.08];
const INITIAL_ZOOM = 9;

const createMapLibreMap: MapFactory = ({ container }) =>
  new MapLibreMap({
    container,
    style: {
      version: 8,
      sources: {
        basemap: {
          type: "raster",
          tiles: [DEFAULT_BASEMAP_TILES],
          tileSize: 256,
          attribution: DEFAULT_BASEMAP_ATTRIBUTION,
        },
      },
      layers: [{ id: "basemap", type: "raster", source: "basemap" }],
    },
    center: INITIAL_CENTER,
    zoom: INITIAL_ZOOM,
  }) as unknown as MapLike;

interface MapPanelProps {
  /** The imagery to draw, or `null` for a basemap-only map. */
  imagery?: MapImagery | null;
  /** Injected in tests; jsdom has no WebGL, so the real map cannot be built. */
  createMap?: MapFactory;
}

/**
 * The interactive map.
 *
 * Purely presentational: it renders what it is given. It performs no CRS
 * conversion, computes no affine and derives no corners - the backend is
 * authoritative for raster geometry, and duplicating that arithmetic here is
 * how the two would drift apart. It also fetches nothing: imagery arrives as a
 * prop from the existing query flow.
 *
 * The image is placed with a MapLibre `image` source, which takes four explicit
 * corners. That is not a stylistic choice: a reprojected UTM window is a
 * quadrilateral in WGS84, so an axis-aligned overlay would misplace it - by
 * ~144 m over a city-sized AOI in the measured case.
 */
export function MapPanel({ imagery = null, createMap }: MapPanelProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLike | null>(null);
  const [ready, setReady] = useState(false);
  const [unavailable, setUnavailable] = useState(false);

  // Created once, never per render. The factory is read through a ref so that
  // passing a fresh function each render cannot retrigger this effect.
  const factoryRef = useRef<MapFactory | undefined>(createMap);
  factoryRef.current = createMap;

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // MapLibre needs WebGL2. A browser without it - or a headless environment -
    // must not take the rest of the application down with it, so the failure is
    // contained here and reported honestly instead of thrown.
    let map: MapLike;
    try {
      map = (factoryRef.current ?? createMapLibreMap)({ container });
    } catch {
      setUnavailable(true);
      return;
    }
    mapRef.current = map;
    map.on("load", () => setReady(true));

    return () => {
      map.remove();
      mapRef.current = null;
      setReady(false);
    };
  }, []);

  const corners = imagery?.corners_wgs84;
  const hasFootprint = isValidFootprint(corners);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;

    // Always tear the old overlay down first, so repeated updates cannot
    // accumulate sources or layers inside MapLibre.
    if (map.getLayer(SATELLITE_LAYER_ID)) map.removeLayer(SATELLITE_LAYER_ID);
    if (map.getSource(SATELLITE_SOURCE_ID)) map.removeSource(SATELLITE_SOURCE_ID);

    if (!imagery || !hasFootprint) return;

    map.addSource(SATELLITE_SOURCE_ID, {
      type: "image",
      url: `data:${imagery.media_type};base64,${imagery.image_base64}`,
      // Verbatim, in the order received: [NW, NE, SE, SW].
      coordinates: corners,
    });
    map.addLayer({
      id: SATELLITE_LAYER_ID,
      type: "raster",
      source: SATELLITE_SOURCE_ID,
      paint: { "raster-opacity": 1 },
    } as { id: string });

    // Framing only - without this a small AOI is a few pixels at the initial
    // zoom and reads as "nothing rendered". The overlay's position is still
    // entirely the four corners; this only moves the camera.
    map.fitBounds(footprintExtent(corners), { padding: 24, duration: 0 });
  }, [imagery, corners, hasFootprint, ready]);

  return (
    <section className="panel" aria-labelledby="map-heading">
      <h2 id="map-heading">Map</h2>
      <div
        ref={containerRef}
        className="map-container"
        data-testid="map-container"
      />
      <MapStatus
        imagery={imagery}
        hasFootprint={hasFootprint}
        unavailable={unavailable}
      />
    </section>
  );
}

/** An honest one-line statement of what is - or is not - on the map. */
function MapStatus({
  imagery,
  hasFootprint,
  unavailable,
}: {
  imagery: MapImagery | null;
  hasFootprint: boolean;
  unavailable: boolean;
}) {
  if (unavailable) {
    return (
      <p className="hint" role="status">
        The map could not be started in this browser, which needs WebGL2.
        Everything else on this page still works.
        {imagery
          ? ` Scene ${imagery.scene_id} was retrieved but cannot be drawn here.`
          : ""}
      </p>
    );
  }
  if (!imagery) {
    return (
      <p className="hint" role="status">
        Run a query with imagery to place a scene on the map.
      </p>
    );
  }
  if (!hasFootprint) {
    return (
      <p className="hint" role="status">
        Scene {imagery.scene_id} has no usable geographic footprint, so it is not
        shown on the map.
      </p>
    );
  }
  return (
    <p className="hint">
      Showing scene {imagery.scene_id}, positioned by its four source-derived
      corner coordinates.
    </p>
  );
}
