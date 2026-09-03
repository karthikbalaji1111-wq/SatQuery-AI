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

/**
 * The imagery the map can draw. A structural subset of `ImageryResponse`: the
 * picture and its footprint, and nothing else.
 */
export interface MapImagery {
  scene_id: string;
  media_type: string;
  image_base64: string;
  corners_wgs84: number[][] | null;
}

/** The slice of the MapLibre API the map component uses. */
export interface MapLike {
  on(event: string, handler: () => void): void;
  addSource(id: string, source: unknown): void;
  removeSource(id: string): void;
  getSource(id: string): unknown;
  addLayer(layer: { id: string }): void;
  removeLayer(id: string): void;
  getLayer(id: string): unknown;
  fitBounds(bounds: number[][], options?: unknown): void;
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
