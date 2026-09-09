/**
 * The satellite overlay's identifiers, contract and footprint validation.
 *
 * Separate from `MapPanel.tsx` so that file exports only its component (React
 * Fast Refresh requires that), and so the validation can be read and tested on
 * its own.
 */

/** Ids of the single satellite overlay the map manages. */
export const SATELLITE_SOURCE_ID = "satellite-image";
export const SATELLITE_LAYER_ID = "satellite-image-layer";

/** Ids of the single NDWI overlay the map manages, kept distinct from the RGB
 * one so the two rasters never share a source: they come from different grids
 * and carry different footprints. */
export const NDWI_SOURCE_ID = "ndwi-image";
export const NDWI_LAYER_ID = "ndwi-image-layer";

/** Ids of the temporal NDWI change overlay - again its own source, because it
 * is a third raster on a third grid with its own footprint. */
export const CHANGE_SOURCE_ID = "ndwi-change-image";
export const CHANGE_LAYER_ID = "ndwi-change-image-layer";

/** Ids of the scene footprint outline drawn over the imagery (Direction B §7). */
export const FOOTPRINT_SOURCE_ID = "scene-footprint";
export const FOOTPRINT_LAYER_ID = "scene-footprint-layer";

/**
 * The imagery the map can draw. A structural subset of `ImageryResponse`: the
 * picture and its footprint, and nothing else.
 */
export interface MapImagery {
  scene_id: string;
  media_type: string;
  image_base64: string;
  corners_wgs84: number[][] | null;
  /**
   * Read-only context for the scene metadata strip. Optional because the map
   * does not need them to DRAW anything - positioning comes entirely from
   * `corners_wgs84` - and because a caller may legitimately hold only the
   * drawing subset. They are already present on the `ImageryResponse` the app
   * passes in; nothing new is fetched or derived to show them.
   */
  crs?: string | null;
  width?: number;
  height?: number;
  /** Which asset was rendered, e.g. `visual` or `vv`. Names the picture. */
  asset?: string;
  /** The bands the picture was built from, as the backend named them. */
  bands?: string[];
  /** Ground sample distance of the returned window, in metres. */
  resolution?: number | null;
}

/** The slice of the MapLibre API the map component uses. */
/**
 * The one method this code needs from a GeoJSON source.
 *
 * Narrow on purpose: `getSource` is typed `unknown` because a style holds many
 * source kinds, and only a GeoJSON one can be re-fed. Guarding on the method
 * keeps that check honest rather than casting and hoping.
 */
export interface GeoJsonSourceLike {
  setData(data: unknown): void;
}

export function isGeoJsonSource(value: unknown): value is GeoJsonSourceLike {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as GeoJsonSourceLike).setData === "function"
  );
}

export interface MapLike {
  on(event: string, handler: () => void): void;
  addSource(id: string, source: unknown): void;
  removeSource(id: string): void;
  getSource(id: string): unknown;
  addLayer(layer: { id: string }): void;
  removeLayer(id: string): void;
  getLayer(id: string): unknown;
  fitBounds(bounds: number[][], options?: unknown): void;
  /**
   * Restyle a layer already on the map - used to sink the basemap behind
   * imagery, and to lift it back when the frame carries no raster.
   */
  setPaintProperty(layer: string, name: string, value: unknown): void;
  /**
   * Re-measure the container. MapLibre sizes its canvas once at creation, so a
   * container that changes width afterwards - which this layout does, since the
   * scene spans the full width until an analysis appears beside it - leaves the
   * canvas at its old size and the view showing a completely different extent.
   */
  resize(): void;
  remove(): void;
}

/**
 * The NDWI raster the map can draw. Like `MapImagery`, a structural subset of
 * the backend contract: the picture and its own footprint, nothing else.
 */
export interface MapNdwi {
  scene_id: string;
  media_type: string;
  image_base64: string;
  corners_wgs84: number[][] | null;
}

/** The requested area of interest, as the backend resolved it. */
export interface MapAoi {
  west: number;
  south: number;
  east: number;
  north: number;
  /** The scene the run selected, when one was selected but not rasterised. */
  scene_id?: string | null;
  /**
   * Whether the run ASKED for a picture at all.
   *
   * This is the difference between "we tried and failed" and "we never tried",
   * and the frame must not say the first when the second is true. The agent's
   * `execute_query` step carries `include_imagery`, and it is commonly `false`:
   * an index is computed by reading the raster bands directly, so a display
   * PNG is not needed to answer the question. Reporting that as a failed
   * retrieval blames the pipeline for a request nobody made.
   *
   * `undefined` means the caller does not know - the frame then says only that
   * no imagery is present, and asserts no cause.
   */
  imagery_requested?: boolean;
  /**
   * The server's reason a REQUESTED retrieval produced no picture, when it
   * gave one. Never populated for a retrieval that was never requested.
   */
  imagery_error?: string | null;
}

export type MapFactory = (options: { container: HTMLElement }) => MapLike;

/**
 * Validate a footprint that arrived over the network.
 *
 * `corners_wgs84` is API data, so it is checked rather than trusted, and it is
 * never repaired: a footprint wrong in a way we could "fix" is a footprint we
 * cannot vouch for, and drawing imagery in the wrong place is the exact failure
 * this pipeline exists to prevent. Invalid means no overlay.
 */
export function isValidFootprint(corners: unknown): corners is number[][] {
  return (
    Array.isArray(corners) &&
    corners.length === 4 &&
    corners.every(
      (corner) =>
        Array.isArray(corner) &&
        corner.length === 2 &&
        corner.every(
          (value) => typeof value === "number" && Number.isFinite(value),
        ) &&
        corner[0] >= -180 &&
        corner[0] <= 180 &&
        corner[1] >= -90 &&
        corner[1] <= 90,
    )
  );
}

/**
 * The axis-aligned extent of a validated footprint, as
 * `[[west, south], [east, north]]`.
 *
 * Viewport framing ONLY. The image itself is positioned solely by its four
 * corners; this rectangle is never used to place anything, which is why it is
 * safe for it to be a rectangle at all. It is min/max over numbers the backend
 * already derived - no geospatial calculation happens here.
 */
export function footprintExtent(corners: number[][]): number[][] {
  const lons = corners.map(([lon]) => lon);
  const lats = corners.map(([, lat]) => lat);
  return [
    [Math.min(...lons), Math.min(...lats)],
    [Math.max(...lons), Math.max(...lats)],
  ];
}
