# Map feature

An interactive MapLibre map that shows a retrieved Sentinel-2 scene at its
correct geographic location.

## Architecture

```
GET /api/v1/satellite/imagery      (existing endpoint, unchanged)
  -> ImageryResponse.transform     affine of the RETURNED image, source CRS
  -> ImageryResponse.corners_wgs84 four [lon, lat] corners, [NW, NE, SE, SW]
  -> QueryPanel (owns the request) -> App (holds the latest result)
  -> MapPanel  -> MapLibre `image` source
```

The backend is authoritative for all raster geometry. `MapPanel` performs no
CRS conversion, computes no affine, derives no corners and fetches nothing: it
renders what it is handed.

## Why four corners rather than a bounding box

Reprojecting a north-up UTM window (Sentinel-2 is EPSG:326xx) to WGS84 does not
produce an axis-aligned rectangle. Measured on a city-sized Chennai AOI, the
corners deviate by up to ~144 m, so an axis-aligned overlay would visibly
misplace the image. MapLibre's `image` source takes four explicit corners,
which matches the real shape.

`ImageryResponse.bbox` is **request metadata**, not coverage, and is never used
for positioning.

## Basemap

Raster tiles from an **external tile service** (OpenStreetMap by default),
fetched directly by the viewer's browser. No API key is required, which is why
it suits the demo; a real deployment must use a provider whose terms permit its
traffic. Override with `VITE_BASEMAP_TILE_URL`.

## Not implemented here

NDWI or index overlays, temporal overlays, change detection, SAR display, layer
toggles, basemap switching, AOI drawing, GeoJSON, client-side reprojection, and
tile serving. There is no backend `MapService`, and this milestone does not
need one.
