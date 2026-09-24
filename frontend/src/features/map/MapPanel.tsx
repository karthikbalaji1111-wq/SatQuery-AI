import { useEffect, useRef, useState } from "react";
import { Map as MapLibreMap, ScaleControl } from "maplibre-gl";

import "maplibre-gl/dist/maplibre-gl.css";

import {
  CHANGE_LAYER_ID,
  CHANGE_SOURCE_ID,
  FOOTPRINT_LAYER_ID,
  FOOTPRINT_SOURCE_ID,
  footprintExtent,
  isValidFootprint,
  NDWI_LAYER_ID,
  NDWI_SOURCE_ID,
  SATELLITE_LAYER_ID,
  SATELLITE_SOURCE_ID,
} from "./footprint";
import type {
  MapAoi,
  MapFactory,
  MapImagery,
  MapLike,
  MapNdwi,
} from "./footprint";
import { isGeoJsonSource } from "./footprint";
import { configureMapLibreWorker } from "./maplibreWorker";

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
 * Outline the scene's footprint over the imagery (Direction B §7).
 *
 * Drawn from the SAME four corners that position the raster, so the outline can
 * never disagree with the picture it frames - it is the picture's own edge, not
 * a second geometry. Cyan, dashed, no fill: it reads over bright coastline and
 * dark water alike without hiding either.
 */
function syncFootprintOutline(
  map: MapLike,
  corners: number[][] | null,
  locator = false,
): void {
  // A closed LineString for both maps. A Polygon with a fill layer was tried
  // for the locator and rendered nothing on a real MapLibre map - the line
  // path is the one that demonstrably draws, so both use it.
  const ring = corners === null ? [] : [...corners, corners[0]];
  const data = {
    type: "Feature" as const,
    properties: {},
    geometry: { type: "LineString" as const, coordinates: ring },
  };

  // UPDATE the source rather than replacing it.
  //
  // This used to removeSource/addSource on every run of the effect that calls
  // it. A GeoJSON source parses its data in a worker, and tearing it down
  // restarts that parse from nothing - so with the effect running more than
  // once the source never reached a loaded state. Measured on a live map:
  // `isSourceLoaded("scene-footprint")` stayed false indefinitely and both
  // queryRenderedFeatures and querySourceFeatures returned empty, with correct
  // geometry sitting in the source and no error raised. The layer and its paint
  // were fine; nothing was ever tiled for them to draw. That is why the scene
  // footprint and the locator AOI outline were invisible.
  const existing = map.getSource(FOOTPRINT_SOURCE_ID);
  if (isGeoJsonSource(existing)) {
    // An empty ring draws nothing, which is how the outline is cleared. The
    // source and layer stay, so the next update is a data change rather than
    // another teardown.
    existing.setData(data);
    return;
  }

  // Nothing to draw and nothing to clear - do not create an idle source.
  if (corners === null) return;

  map.addSource(FOOTPRINT_SOURCE_ID, { type: "geojson", data });

  // Over imagery the outline is cyan, which reads against bright sand and dark
  // water alike. On the locator's pale grey basemap that cyan disappears, so
  // there it takes the geospatial blue and a heavier stroke - the inset is
  // small, and the footprint is the only reason it exists.
  map.addLayer({
    id: FOOTPRINT_LAYER_ID,
    type: "line",
    source: FOOTPRINT_SOURCE_ID,
    paint: {
      "line-color": locator ? "#0B6BCB" : "#5FD3F3",
      "line-width": locator ? 2.5 : 1.25,
      "line-opacity": 0.95,
      "line-dasharray": [5, 3],
    },
  } as { id: string });
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

/**
 * Where the camera sits for the instant between construction and the first
 * `fitBounds`.
 *
 * This is NOT an empty-state view any more: the panel only mounts once there
 * is real geography to show, and the effect frames it immediately. It used to
 * be what a reader saw before asking anything, which printed a city nobody had
 * asked about and read as a pre-loaded demo - `WorkspaceIntro` holds that
 * space now.
 */
// A neutral world view. The map is on screen before any query (map-first), so
// its opening frame must not name a place the user never asked about; every
// real view comes from fitting the AOI or the scene a run resolves.
const INITIAL_CENTER: [number, number] = [20, 15];
const INITIAL_ZOOM = 1.4;

const createMapLibreMap: MapFactory = ({ container }) => {
  // Before the first map exists: MapLibre resolves its worker relative to its
  // own module URL, which in a production bundle points at an asset that was
  // never emitted. See `maplibreWorker.ts`.
  configureMapLibreWorker();
  const map = new MapLibreMap({
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
      layers: [
        {
          id: "basemap",
          type: "raster",
          source: "basemap",
          // Desaturated toward grey so the basemap reads as cartographic
          // GROUND beneath Earth-observation data rather than as a navigation
          // map - but NOT bleached. An earlier pass lifted the black point to
          // 0.42, which flattened coastlines and road structure into a pale
          // wash and made the viewport look empty; the ground still has to be
          // readable for the footprint and the overlays to mean anything.
          // Saturation alone keeps the imagery the only richly coloured thing
          // on screen, so the contrast is left near neutral.
          //
          // Applied as per-layer raster paint, never as a CSS filter on the
          // container: a filter would also wash out the satellite imagery and
          // the index overlays drawn on top, which are the actual measurements.
          paint: {
            "raster-saturation": -1,
            "raster-brightness-min": 0.02,
            "raster-brightness-max": 0.34,
            "raster-contrast": -0.05,
          },
        },
      ],
    },
    center: INITIAL_CENTER,
    zoom: INITIAL_ZOOM,
    attributionControl: { compact: false },
  });
  // Required for the tile source, and useful: it is the only thing on screen
  // that states the imagery's ground scale (Direction B §7).
  map.addControl(new ScaleControl({ maxWidth: 110, unit: "metric" }), "bottom-left");
  return map as unknown as MapLike;
};

interface MapPanelProps {
  /**
   * The area a run resolved and analysed, even when no raster came back.
   * A query can select a scene without requesting imagery - the analysis is
   * still real, and the viewport should show WHERE it happened rather than
   * claiming nothing was found.
   */
  aoi?: MapAoi | null;
  /** The imagery to draw, or `null` for a basemap-only map. */
  imagery?: MapImagery | null;
  /** The NDWI raster to draw over it, or `null` for none. */
  ndwi?: MapNdwi | null;
  /** The temporal NDWI change raster, or `null` for none. */
  change?: MapNdwi | null;
  /**
   * `imagery` is the analysis viewport: the raster is the subject, so the
   * basemap sinks behind it. `locator` is the footprint inset: no raster, a
   * legible basemap, and the analysed extent outlined on it. Both are the same
   * component because both position everything from the same corners.
   */
  variant?: "imagery" | "locator";
  /** A run is in flight, so the frame is awaiting a scene rather than empty. */
  busy?: boolean;
  /**
   * What is drawn belongs to the PREVIOUS question while a new one runs. The
   * map keeps its place instead of blanking out, and says so; the new result
   * replaces every layer when it arrives.
   */
  stale?: boolean;
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
  aoi = null,
  imagery = null,
  ndwi = null,
  change = null,
  variant = "imagery",
  busy = false,
  stale = false,
  createMap,
}: MapPanelProps) {
  // The locator never draws a raster: it exists to say where, on a map a
  // reader can actually read.
  const locator = variant === "locator";
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

    // The container's width changes after the map is built - the scene spans
    // the full grid until an analysis panel appears beside it. Without this the
    // canvas keeps its original size and MapLibre renders a stretched view of a
    // far wider extent than intended.
    let observer: ResizeObserver | undefined;
    if (typeof ResizeObserver !== "undefined") {
      observer = new ResizeObserver(() => map.resize());
      observer.observe(container);
    }

    return () => {
      observer?.disconnect();
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
      raster: locator ? null : imagery,
    });
    // NDWI goes on above the true-colour image...
    const ndwiCorners = syncImageOverlay(map, {
      sourceId: NDWI_SOURCE_ID,
      layerId: NDWI_LAYER_ID,
      raster: locator ? null : ndwi,
    });
    // ...and the temporal change above that, being the most derived result.
    const changeCorners = syncImageOverlay(map, {
      sourceId: CHANGE_SOURCE_ID,
      layerId: CHANGE_LAYER_ID,
      raster: locator ? null : change,
    });

    // Framing only - without this a small AOI is a few pixels at the initial
    // zoom and reads as "nothing rendered". Positions still come entirely from
    // the corners; this only moves the camera. The index wins when both are
    // present, because that is what the user asked to look at.
    const frameTo = changeCorners ?? ndwiCorners ?? rgbCorners;
    // The outline traces the scene itself, so it follows the true-colour
    // footprint when there is one and the index grid otherwise. With no raster
    // at all it traces the resolved extent, which is what the measurements
    // describe - so the frame always shows the area being talked about.
    const aoiRing =
      aoi === null
        ? null
        : [
            [aoi.west, aoi.north],
            [aoi.east, aoi.north],
            [aoi.east, aoi.south],
            [aoi.west, aoi.south],
          ];
    syncFootprintOutline(map, rgbCorners ?? frameTo ?? aoiRing, locator);

    // With a raster the basemap sinks to a quiet ground so the imagery is the
    // only bright thing on screen; without one it is the only geography there
    // is, so it lifts back to legible. Per-layer paint, never a CSS filter.
    map.setPaintProperty(
      "basemap",
      "raster-brightness-max",
      locator ? 0.99 : frameTo ? 0.44 : 0.72,
    );
    map.setPaintProperty(
      "basemap",
      "raster-contrast",
      locator ? -0.05 : frameTo ? -0.05 : 0.05,
    );

    if (locator && aoi) {
      map.fitBounds(
        [
          [aoi.west, aoi.south],
          [aoi.east, aoi.north],
        ],
        { padding: 26, duration: 0, maxZoom: 12.5 },
      );
    } else if (frameTo) {
      map.fitBounds(footprintExtent(frameTo), { padding: 24, duration: 0 });
    } else if (aoi) {
      // No raster, but the run resolved an area - frame it, so the viewport
      // shows the region the measurements describe.
      map.fitBounds(
        [
          [aoi.west, aoi.south],
          [aoi.east, aoi.north],
        ],
        { padding: 96, duration: 0 },
      );
    }
  }, [aoi, imagery, ndwi, change, locator, ready]);

  return (
    // The imagery cell is full-bleed (Direction B §7): the tiles reach every
    // edge and everything else is an overlay held within 22px of a border, so
    // nothing is ever layered over the centre of the scene. The heading stays
    // in the accessibility tree - the visible title is the caption on the
    // scrim, which names the actual asset rather than the panel.
    <section
      className={`panel map-panel${locator ? " locator-panel" : ""}`}
      aria-labelledby="map-heading"
    >
      <h2 id="map-heading" className="sr-only">
        {locator ? "Footprint" : "Satellite scene"}
      </h2>
      <div
        ref={containerRef}
        className="map-container"
        data-testid="map-container"
      />
      {locator && aoi && <LocatorExtent aoi={aoi} />}
      {!locator && <div className="imagery-scrim" aria-hidden="true" />}
      {!locator && stale && (
        <div className="map-updating" aria-live="polite">
          Previous result · updating for the new question
        </div>
      )}
      {!locator && (
        <ImageryCaption imagery={imagery} ndwi={ndwi} change={change} />
      )}
      {!locator && (
        <LayerToolbar
          hasFootprint={hasFootprint}
          hasNdwiFootprint={hasNdwiFootprint}
          hasChangeFootprint={hasChangeFootprint}
        />
      )}
      {!locator &&
        !hasFootprint &&
        !hasNdwiFootprint &&
        !hasChangeFootprint && <ImageryEmpty aoi={aoi} busy={busy} />}
      {!locator && (
      <MapStatus
        imagery={imagery}
        ndwi={ndwi}
        change={change}
        hasChangeFootprint={hasChangeFootprint}
        hasFootprint={hasFootprint}
        hasNdwiFootprint={hasNdwiFootprint}
        unavailable={unavailable}
      />
      )}
    </section>
  );
}

/**
 * Which rasters are currently drawn, and on what.
 *
 * Reads only the footprint flags the panel already computed, so it can never
 * claim a layer the map is not actually showing. Dimmed chips mean "not
 * present", which is why they are shown at all: an empty toolbar would leave a
 * viewer guessing whether a layer failed or was never requested.
 */
/**
 * The caption on the scrim: what this picture actually is.
 *
 * Built only from fields the imagery response carries - the asset it rendered,
 * the bands it used and the ground sample distance of the returned window. No
 * acquisition time appears here because `ImageryResponse` does not carry one;
 * it is reported in the evidence band, where the STAC record supplies it.
 */
function ImageryCaption({
  imagery,
  ndwi,
  change,
}: {
  imagery: MapImagery | null;
  ndwi: MapNdwi | null;
  change: MapNdwi | null;
}) {
  const scene = change?.scene_id ?? ndwi?.scene_id ?? imagery?.scene_id ?? null;
  if (scene === null) return null;

  const title =
    change !== null
      ? "NDWI change · later minus earlier"
      : ndwi !== null
        ? "Sentinel-2 · NDWI index"
        : imagery?.asset === "vv" || imagery?.asset === "vh"
          ? `Sentinel-1 SAR · ${imagery.asset.toUpperCase()} display`
          : "Sentinel-2 L2A · true colour";
  const detail = [
    scene,
    // A SAR display PNG carries one measured band replicated into three, so a
    // raw join renders "vv vv vv", which reads as a bug rather than as one
    // grayscale band. Collapse repeats; a genuine multi-band composite (the
    // Sentinel-2 "red green blue" true-colour case) still lists every band.
    imagery?.bands?.length
      ? [...new Set(imagery.bands)].join(" ")
      : null,
    imagery?.resolution ? `${imagery.resolution} m` : null,
    imagery?.crs,
  ]
    .filter((part): part is string => Boolean(part))
    .join(" · ");

  return (
    <div className="imagery-caption">
      <span className="imagery-title">{title}</span>
      <span className="imagery-sub">{detail}</span>
    </div>
  );
}

/**
 * Nothing is drawn - but say precisely why.
 *
 * A run that selects a scene without requesting imagery is a successful run,
 * and reporting "no scene loaded" over its results would contradict the
 * evidence beside it. The cases are named separately.
 *
 * The distinction that matters most here is REQUESTED-AND-FAILED versus
 * NEVER-REQUESTED. This frame used to say "Imagery not retrieved" for both,
 * which is a statement of failure - and the common case is not a failure at
 * all: an index reads the raster bands directly and needs no display PNG, so
 * the agent's plan routinely sets `include_imagery: false`. Announcing a
 * retrieval failure over a run whose measurements came from those very pixels
 * tells the reader the pipeline broke when it did exactly what was asked.
 */
function ImageryEmpty({
  aoi,
  busy = false,
}: {
  aoi: MapAoi | null;
  busy?: boolean;
}) {
  if (busy) {
    return (
      <div className="imagery-empty imagery-empty-busy" role="status">
        <span className="imagery-empty-label">Retrieving scene</span>
        <span className="imagery-empty-note">
          Searching the catalog and reading a bounded window of the raster.
        </span>
      </div>
    );
  }
  if (aoi?.scene_id) {
    // A requested retrieval that produced nothing. The server's own cause is
    // reported in the evidence panel's notice; this frame names the outcome.
    if (aoi.imagery_error) {
      return (
        <div className="imagery-empty imagery-empty-aoi" role="status">
          <span className="imagery-empty-label">
            Imagery could not be retrieved
          </span>
          <span className="imagery-empty-note">
            A preview of scene {aoi.scene_id} was requested but could not be
            read; the deterministic evidence panel states the cause. Scene
            discovery and the measurements are unaffected.
          </span>
        </div>
      );
    }
    // Never requested. Said as a property of the request, not of the pipeline.
    if (aoi.imagery_requested === false) {
      return (
        <div className="imagery-empty imagery-empty-aoi">
          <span className="imagery-empty-label">Imagery not requested</span>
          <span className="imagery-empty-note">
            Scene {aoi.scene_id} was measured directly from its raster bands,
            which needs no display image, so this run did not request one. This
            view is centred on the extent the measurements describe.
          </span>
        </div>
      );
    }
    // Requested state unknown: report the absence and assert no cause.
    return (
      <div className="imagery-empty imagery-empty-aoi">
        <span className="imagery-empty-label">No imagery on this map</span>
        <span className="imagery-empty-note">
          Scene {aoi.scene_id} was analysed without a raster preview. This view
          is centred on the extent the measurements describe.
        </span>
      </div>
    );
  }
  if (aoi) {
    // An area is resolved and the view is centred on it. Veiling it as heavily as an
    // empty frame would hide the one real thing the viewport is showing.
    return (
      <div className="imagery-empty imagery-empty-aoi">
        <span className="imagery-empty-label">Area resolved</span>
        <span className="imagery-empty-note">
          This view is centred on the requested area. Run with imagery to place
          a Sentinel scene over it.
        </span>
      </div>
    );
  }
  return (
    <div className="imagery-empty">
      <span className="imagery-empty-label">No scene loaded</span>
      <span className="imagery-empty-note">
        Run a query with imagery to place a Sentinel scene on the map
      </span>
    </div>
  );
}

/**
 * Which rasters are drawn, and which are simply not loaded.
 *
 * Reads only the footprint flags the panel already computed, so it can never
 * claim a layer the map is not showing. An unloaded layer is labelled
 * "- none -" rather than greyed out: dimming alone reads as broken, and the
 * distinction between "unavailable" and "failed" matters here.
 */
function LayerToolbar({
  hasFootprint,
  hasNdwiFootprint,
  hasChangeFootprint,
}: {
  hasFootprint: boolean;
  hasNdwiFootprint: boolean;
  hasChangeFootprint: boolean;
}) {
  const layers = [
    { label: "True colour", on: hasFootprint },
    { label: "Water index", on: hasNdwiFootprint },
    { label: "Index change", on: hasChangeFootprint },
  ];

  return (
    <div className="layer-bar" role="group" aria-label="Map layers">
      {layers.map((layer) => (
        <div
          key={layer.label}
          className="layer-cell"
          data-active={layer.on}
          title={layer.on ? `${layer.label}: drawn` : `${layer.label}: not loaded`}
        >
          <span className="layer-name">{layer.label}</span>
        </div>
      ))}
    </div>
  );
}

/** An honest one-line statement of what is - or is not - on the map. */
/**
 * The area of interest, stated numerically over the locator.
 *
 * The locator centres and zooms on the AOI, but the vector outline that would
 * mark its edges does not render - see the GeoJSON limitation documented in
 * `syncFootprintOutline`. Rather than draw a decorative rectangle in HTML,
 * which would claim a geometry the map is not actually showing, the extent is
 * reported as the numbers it comes from. These are the AOI's real bounds, so a
 * reader can still say exactly which ground is being described.
 */
function LocatorExtent({ aoi }: { aoi: MapAoi }) {
  return (
    <p className="locator-extent">
      <span>
        {aoi.south.toFixed(3)}–{aoi.north.toFixed(3)}° N
      </span>
      <span>
        {aoi.west.toFixed(3)}–{aoi.east.toFixed(3)}° E
      </span>
    </p>
  );
}

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
        Showing NDWI change (later minus earlier) for scene{" "}
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
    // The centred empty state carries this; repeating it here would print the
    // same sentence twice over the same frame.
    return null;
  }
  if (!hasFootprint) {
    return (
      <p className="hint" role="status">
        Scene {imagery.scene_id} has no usable geographic footprint, so it is not
        shown on the map.
      </p>
    );
  }
  return <p className="hint">Positioned by four source-derived corners</p>;
}
