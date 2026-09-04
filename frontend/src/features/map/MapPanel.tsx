import { useEffect, useRef, useState } from "react";
import { Map as MapLibreMap } from "maplibre-gl";

import "maplibre-gl/dist/maplibre-gl.css";

import {
  CHANGE_LAYER_ID,
  CHANGE_SOURCE_ID,
  footprintExtent,
  isValidFootprint,
  NDWI_LAYER_ID,
  NDWI_SOURCE_ID,
  SATELLITE_LAYER_ID,
  SATELLITE_SOURCE_ID,
} from "./footprint";
import type { MapFactory, MapImagery, MapLike, MapNdwi } from "./footprint";

/**
 * Put one image overlay on the map, or take it off.
 *
 * The same three steps for every raster: tear down whatever is there, then add
 * the source and its layer if - and only if - the new raster has a footprint we
 * validated. Shared so RGB and NDWI cannot drift apart in how they are placed
 * or cleaned up; the caller supplies the ids, the picture and the corners, and
 * nothing here computes geometry.
 */
function syncImageOverlay(
  map: MapLike,
  {
    sourceId,
    layerId,
    raster,
  }: {
    sourceId: string;
    layerId: string;
    raster: { media_type: string; image_base64: string; corners_wgs84: number[][] | null } | null;
  },
): number[][] | null {
  if (map.getLayer(layerId)) map.removeLayer(layerId);
  if (map.getSource(sourceId)) map.removeSource(sourceId);

  const corners = raster?.corners_wgs84;
  if (!raster || !isValidFootprint(corners)) return null;

  map.addSource(sourceId, {
    type: "image",
    url: `data:${raster.media_type};base64,${raster.image_base64}`,
    // Verbatim, in the order received: [NW, NE, SE, SW].
    coordinates: corners,
  });
  map.addLayer({
    id: layerId,
    type: "raster",
    source: sourceId,
    paint: { "raster-opacity": 1 },
  } as { id: string });
  return corners;
}

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
  /** The NDWI raster to draw over it, or `null` for none. */
  ndwi?: MapNdwi | null;
  /** The temporal NDWI change raster, or `null` for none. */
  change?: MapNdwi | null;
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
export function MapPanel({
  imagery = null,
  ndwi = null,
  change = null,
  createMap,
}: MapPanelProps) {
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

  const hasFootprint = isValidFootprint(imagery?.corners_wgs84);
  const hasNdwiFootprint = isValidFootprint(ndwi?.corners_wgs84);
  const hasChangeFootprint = isValidFootprint(change?.corners_wgs84);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;

    const rgbCorners = syncImageOverlay(map, {
      sourceId: SATELLITE_SOURCE_ID,
      layerId: SATELLITE_LAYER_ID,
      raster: imagery,
    });
    // NDWI goes on above the true-colour image...
    const ndwiCorners = syncImageOverlay(map, {
      sourceId: NDWI_SOURCE_ID,
      layerId: NDWI_LAYER_ID,
      raster: ndwi,
    });
    // ...and the temporal change above that, being the most derived result.
    const changeCorners = syncImageOverlay(map, {
      sourceId: CHANGE_SOURCE_ID,
      layerId: CHANGE_LAYER_ID,
      raster: change,
    });

    // Framing only - without this a small AOI is a few pixels at the initial
    // zoom and reads as "nothing rendered". Positions still come entirely from
    // the corners; this only moves the camera. The index wins when both are
    // present, because that is what the user asked to look at.
    const frameTo = changeCorners ?? ndwiCorners ?? rgbCorners;
    if (frameTo) {
      map.fitBounds(footprintExtent(frameTo), { padding: 24, duration: 0 });
    }
  }, [imagery, ndwi, change, ready]);

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
        ndwi={ndwi}
        change={change}
        hasChangeFootprint={hasChangeFootprint}
        hasFootprint={hasFootprint}
        hasNdwiFootprint={hasNdwiFootprint}
        unavailable={unavailable}
      />
    </section>
  );
}

/** An honest one-line statement of what is - or is not - on the map. */
function MapStatus({
  imagery,
  ndwi,
  change,
  hasFootprint,
  hasNdwiFootprint,
  hasChangeFootprint,
  unavailable,
}: {
  imagery: MapImagery | null;
  ndwi: MapNdwi | null;
  change: MapNdwi | null;
  hasFootprint: boolean;
  hasNdwiFootprint: boolean;
  hasChangeFootprint: boolean;
  unavailable: boolean;
}) {
  if (unavailable) {
    return (
      <p className="hint" role="status">
        The map could not be started in this browser, which needs WebGL2.
        Everything else on this page still works.
        {change
          ? ` An NDWI change result for scene ${change.scene_id} was produced but cannot be drawn here.`
          : ndwi
            ? ` An NDWI result for scene ${ndwi.scene_id} was produced but cannot be drawn here.`
            : imagery
              ? ` Scene ${imagery.scene_id} was retrieved but cannot be drawn here.`
              : ""}
      </p>
    );
  }
  if (hasChangeFootprint && change) {
    return (
      <p className="hint">
        Showing NDWI change (target minus baseline) for scene{" "}
        {change.scene_id}. Colour maps the index difference only - it is not a
        map of water gained or lost. Pixels not measured in both observations
        are transparent.
      </p>
    );
  }
  if (change && !hasChangeFootprint) {
    return (
      <p className="hint" role="status">
        An NDWI change result was produced but has no usable geographic
        footprint, so it is not shown on the map.
      </p>
    );
  }
  if (hasNdwiFootprint && ndwi) {
    return (
      <p className="hint">
        Showing the NDWI index for scene {ndwi.scene_id}. Colour maps the index
        value only - a high index is not a water classification. Unmeasured
        pixels are transparent.
      </p>
    );
  }
  if (ndwi && !hasNdwiFootprint) {
    return (
      <p className="hint" role="status">
        An NDWI result was produced but has no usable geographic footprint, so
        it is not shown on the map.
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
