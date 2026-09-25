import { act, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  CHANGE_SOURCE_ID,
  FOOTPRINT_LAYER_ID,
  FOOTPRINT_SOURCE_ID,
  NDWI_LAYER_ID,
  NDWI_SOURCE_ID,
  SATELLITE_LAYER_ID,
  SATELLITE_SOURCE_ID,
CHANGE_LAYER_ID,
} from "./footprint";
import type { MapImagery, MapLike, MapNdwi } from "./footprint";
import { BASEMAP_MAX_ZOOM, MAX_FIT_ZOOM, MapPanel } from "./MapPanel";

afterEach(() => {
  vi.restoreAllMocks();
});

/**
 * A hand-written recording fake, following the project convention: no dynamic
 * module mocking. jsdom has no WebGL, so the real MapLibre map can never be
 * constructed in a test; the component takes its factory as a prop instead.
 */
class FakeMap implements MapLike {
  readonly sources = new Map<string, unknown>();
  readonly layers = new Map<string, unknown>();
  readonly calls: string[] = [];
  readonly fitBoundsCalls: { bounds: number[][]; options?: unknown }[] = [];
  readonly paint: { layer: string; name: string; value: unknown }[] = [];
  removed = false;
  resizes = 0;
  private handlers: Record<string, (() => void)[]> = {};

  on(event: string, handler: () => void): void {
    (this.handlers[event] ??= []).push(handler);
  }
  emit(event: string): void {
    for (const handler of this.handlers[event] ?? []) handler();
  }
  addSource(id: string, source: unknown): void {
    if (this.sources.has(id)) throw new Error(`duplicate source ${id}`);
    // A real MapLibre GeoJSON source exposes setData, and the panel now uses
    // it to UPDATE the footprint instead of replacing the source. A fake
    // without it sends production code down the add-again path that real
    // MapLibre would reject - the fixture has to model the thing it stands in
    // for, or it tests a map that does not exist.
    const stored =
      source !== null &&
      typeof source === "object" &&
      (source as { type?: string }).type === "geojson"
        ? Object.assign({}, source, {
            setData: (data: unknown) => {
              (this.sources.get(id) as { data?: unknown }).data = data;
              this.calls.push(`setData:${id}`);
            },
          })
        : source;
    this.sources.set(id, stored);
    this.calls.push(`addSource:${id}`);
  }
  removeSource(id: string): void {
    this.sources.delete(id);
    this.calls.push(`removeSource:${id}`);
  }
  getSource(id: string): unknown {
    return this.sources.get(id);
  }
  addLayer(layer: { id: string }): void {
    if (this.layers.has(layer.id)) throw new Error(`duplicate layer ${layer.id}`);
    this.layers.set(layer.id, layer);
    this.calls.push(`addLayer:${layer.id}`);
  }
  removeLayer(id: string): void {
    this.layers.delete(id);
    this.calls.push(`removeLayer:${id}`);
  }
  getLayer(id: string): unknown {
    return this.layers.get(id);
  }
  fitBounds(bounds: number[][], options?: unknown): void {
    this.fitBoundsCalls.push({ bounds, options });
    this.calls.push("fitBounds");
  }
  setPaintProperty(layer: string, name: string, value: unknown): void {
    this.paint.push({ layer, name, value });
  }
  resize(): void {
    this.resizes += 1;
    this.calls.push("resize");
  }
  remove(): void {
    this.removed = true;
    this.calls.push("remove");
  }
}

const CORNERS: number[][] = [
  [80.279621036, 13.066131716], // NW
  [80.289951131, 13.066160297], // NE
  [80.29002857, 13.039034352], // SE
  [80.279699601, 13.039005833], // SW
];

function imagery(overrides: Partial<MapImagery> = {}): MapImagery {
  return {
    scene_id: "S2B_44PMV_20250104_0_L2A",
    media_type: "image/png",
    image_base64: "iVBORw0KGgo=",
    corners_wgs84: CORNERS,
    ...overrides,
  };
}

/** Renders with a fake factory and returns the created maps. */
function renderMap(props: { imagery?: MapImagery | null } = {}) {
  const created: FakeMap[] = [];
  const createMap = vi.fn(() => {
    const map = new FakeMap();
    created.push(map);
    return map;
  });
  const view = render(
    <MapPanel imagery={props.imagery ?? null} createMap={createMap} />,
  );
  // MapLibre only accepts sources once the style has loaded.
  act(() => created.forEach((map) => map.emit("load")));
  return { created, createMap, view };
}

/**
 * The coordinates the footprint layer is currently drawing.
 *
 * The panel clears the outline by feeding the source an EMPTY ring rather than
 * destroying it: a GeoJSON source reparses in a worker, and tearing it down on
 * every update meant it never finished loading and the outline never drew at
 * all. So "the outline is gone" is asserted as "nothing to draw", which is the
 * observable contract, rather than as "the source object was destroyed", which
 * was only ever the mechanism.
 */
function footprintRing(map: FakeMap): unknown[] {
  const source = map.getSource(FOOTPRINT_SOURCE_ID) as
    | { data?: { geometry?: { coordinates?: unknown[] } } }
    | undefined;
  return source?.data?.geometry?.coordinates ?? [];
}


describe("MapPanel", () => {
  // --- A. renders without imagery ---------------------------------------- #

  it("renders the panel with no imagery", () => {
    renderMap();
    expect(
      screen.getByRole("heading", { name: /satellite scene/i }),
    ).toBeInTheDocument();
  });

  it("adds no satellite source when there is no imagery", () => {
    const { created } = renderMap();
    expect(created[0].sources.size).toBe(0);
    expect(created[0].layers.size).toBe(0);
  });

  // --- B. the map is created exactly once -------------------------------- #

  it("creates the map exactly once", () => {
    const { createMap } = renderMap({ imagery: imagery() });
    expect(createMap).toHaveBeenCalledTimes(1);
  });

  it("does not recreate the map when imagery changes", () => {
    const created: FakeMap[] = [];
    const createMap = vi.fn(() => {
      const map = new FakeMap();
      created.push(map);
      return map;
    });
    const { rerender } = render(
      <MapPanel imagery={imagery()} createMap={createMap} />,
    );
    act(() => created.forEach((m) => m.emit("load")));
    rerender(
      <MapPanel
        imagery={imagery({ scene_id: "other", image_base64: "AAAA" })}
        createMap={createMap}
      />,
    );

    expect(createMap).toHaveBeenCalledTimes(1);
    expect(created).toHaveLength(1);
  });

  // --- C. exactly one image source --------------------------------------- #

  it("adds exactly one satellite image source and layer", () => {
    const { created } = renderMap({ imagery: imagery() });
    const map = created[0];

    // +1 for the scene footprint outline drawn from the same corners.
    expect(map.sources.size).toBe(2);
    expect(map.layers.size).toBe(2);
    expect(map.getSource(SATELLITE_SOURCE_ID)).toBeDefined();
    expect(map.getLayer(SATELLITE_LAYER_ID)).toBeDefined();
  });

  it("uses an image source type", () => {
    const { created } = renderMap({ imagery: imagery() });
    const source = created[0].getSource(SATELLITE_SOURCE_ID) as {
      type: string;
    };
    expect(source.type).toBe("image");
  });

  // --- D. corners are passed through verbatim, in order ------------------ #

  it("passes the four corners to MapLibre in the received order", () => {
    const { created } = renderMap({ imagery: imagery() });
    const source = created[0].getSource(SATELLITE_SOURCE_ID) as {
      coordinates: number[][];
    };

    expect(source.coordinates).toEqual(CORNERS);
    expect(source.coordinates[0]).toEqual(CORNERS[0]); // NW first
    expect(source.coordinates[2]).toEqual(CORNERS[2]); // SE third
  });

  it("does not reorder or normalise the corners", () => {
    const { created } = renderMap({ imagery: imagery() });
    const source = created[0].getSource(SATELLITE_SOURCE_ID) as {
      coordinates: number[][];
    };
    // Full precision preserved - no rounding on the way through.
    expect(source.coordinates[0][0]).toBe(80.279621036);
  });

  // --- E. the image is the base64 payload -------------------------------- #

  it("uses image_base64 as the image source url", () => {
    const { created } = renderMap({ imagery: imagery() });
    const source = created[0].getSource(SATELLITE_SOURCE_ID) as { url: string };

    expect(source.url).toBe("data:image/png;base64,iVBORw0KGgo=");
  });

  // --- F. replacing imagery cleans up the previous source ---------------- #

  it("removes the previous layer and source before adding the new one", () => {
    const created: FakeMap[] = [];
    const createMap = vi.fn(() => {
      const map = new FakeMap();
      created.push(map);
      return map;
    });
    const { rerender } = render(
      <MapPanel imagery={imagery()} createMap={createMap} />,
    );
    act(() => created.forEach((m) => m.emit("load")));
    rerender(
      <MapPanel
        imagery={imagery({ scene_id: "second", image_base64: "BBBB" })}
        createMap={createMap}
      />,
    );

    const map = created[0];
    // Still exactly one of each - resources did not accumulate.
    // +1 for the scene footprint outline drawn from the same corners.
    expect(map.sources.size).toBe(2);
    expect(map.layers.size).toBe(2);
    // The layer was torn down before being re-added.
    expect(map.calls).toContain(`removeLayer:${SATELLITE_LAYER_ID}`);
    expect(map.calls).toContain(`removeSource:${SATELLITE_SOURCE_ID}`);
    const source = map.getSource(SATELLITE_SOURCE_ID) as { url: string };
    expect(source.url).toBe("data:image/png;base64,BBBB");
  });

  it("removes the overlay when imagery becomes null", () => {
    const created: FakeMap[] = [];
    const createMap = vi.fn(() => {
      const map = new FakeMap();
      created.push(map);
      return map;
    });
    const { rerender } = render(
      <MapPanel imagery={imagery()} createMap={createMap} />,
    );
    act(() => created.forEach((m) => m.emit("load")));
    rerender(<MapPanel imagery={null} createMap={createMap} />);

    // The overlay itself is gone...
    expect(created[0].getSource(SATELLITE_SOURCE_ID)).toBeUndefined();
    expect(created[0].getLayer(SATELLITE_LAYER_ID)).toBeUndefined();
    // ...and the footprint, which is kept and re-fed rather than destroyed,
    // is drawing nothing.
    expect(footprintRing(created[0])).toEqual([]);
  });

  // --- G. unmount ---------------------------------------------------------#

  it("removes the map on unmount", () => {
    const created: FakeMap[] = [];
    const createMap = vi.fn(() => {
      const map = new FakeMap();
      created.push(map);
      return map;
    });
    const { unmount } = render(<MapPanel imagery={null} createMap={createMap} />);
    unmount();

    expect(created[0].removed).toBe(true);
  });

  // --- H. malformed corners are refused, never repaired ------------------ #

  it("renders no overlay when corners are null", () => {
    const { created } = renderMap({
      imagery: imagery({ corners_wgs84: null }),
    });
    expect(created[0].sources.size).toBe(0);
  });

  it.each([
    ["three corners", [CORNERS[0], CORNERS[1], CORNERS[2]]],
    ["five corners", [...CORNERS, CORNERS[0]]],
    ["a corner with one number", [[80.1], CORNERS[1], CORNERS[2], CORNERS[3]]],
    ["a corner with three numbers", [[80.1, 13.0, 1], CORNERS[1], CORNERS[2], CORNERS[3]]],
    ["a NaN", [[NaN, 13.0], CORNERS[1], CORNERS[2], CORNERS[3]]],
    ["an Infinity", [[Infinity, 13.0], CORNERS[1], CORNERS[2], CORNERS[3]]],
    ["longitude out of range", [[181, 13.0], CORNERS[1], CORNERS[2], CORNERS[3]]],
    ["latitude out of range", [[80.1, 91], CORNERS[1], CORNERS[2], CORNERS[3]]],
    ["a non-numeric value", [["80.1", 13.0], CORNERS[1], CORNERS[2], CORNERS[3]]],
  ])("renders no overlay for %s", (_label, corners) => {
    const { created } = renderMap({
      imagery: imagery({ corners_wgs84: corners as number[][] }),
    });
    expect(created[0].sources.size).toBe(0);
  });

  it("explains a malformed footprint without crashing", () => {
    renderMap({ imagery: imagery({ corners_wgs84: null }) });
    expect(screen.getByRole("status")).toBeInTheDocument();
  });

  it("accepts boundary coordinates", () => {
    const { created } = renderMap({
      imagery: imagery({
        corners_wgs84: [
          [-180, 90],
          [180, 90],
          [180, -90],
          [-180, -90],
        ],
      }),
    });
    // +1 for the scene footprint outline drawn from the same corners.
    expect(created[0].sources.size).toBe(2);
  });

  // --- I. no duplicate maps or layers ------------------------------------ #

  it("does not duplicate sources across repeated identical renders", () => {
    const created: FakeMap[] = [];
    const createMap = vi.fn(() => {
      const map = new FakeMap();
      created.push(map);
      return map;
    });
    const props = imagery();
    const { rerender } = render(<MapPanel imagery={props} createMap={createMap} />);
    act(() => created.forEach((m) => m.emit("load")));
    rerender(<MapPanel imagery={props} createMap={createMap} />);
    rerender(<MapPanel imagery={props} createMap={createMap} />);

    expect(created).toHaveLength(1);
    // +1 for the scene footprint outline drawn from the same corners.
    expect(created[0].sources.size).toBe(2);
    expect(created[0].layers.size).toBe(2);
  });

  it("never calls the network itself", () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    renderMap({ imagery: imagery() });
    expect(fetchSpy).not.toHaveBeenCalled();
  });
});

describe("MapPanel when the map cannot start", () => {
  it("does not throw when map creation fails", () => {
    const createMap = vi.fn(() => {
      throw new Error("WebGL2 is required to display this map");
    });
    expect(() =>
      render(<MapPanel imagery={imagery()} createMap={createMap} />),
    ).not.toThrow();
  });

  it("says the map is unavailable rather than failing silently", () => {
    const createMap = vi.fn(() => {
      throw new Error("WebGL2 is required to display this map");
    });
    render(<MapPanel imagery={imagery()} createMap={createMap} />);
    expect(screen.getByRole("status")).toHaveTextContent(/WebGL2/i);
  });
});

describe("MapPanel viewport framing", () => {
  it("fits the view to the footprint extent when imagery is added", () => {
    const { created } = renderMap({ imagery: imagery() });
    const [call] = created[0].fitBoundsCalls;

    expect(created[0].fitBoundsCalls).toHaveLength(1);
    // Axis-aligned extent of the four corners - framing only.
    expect(call.bounds).toEqual([
      [80.279621036, 13.039005833], // [west, south]
      [80.29002857, 13.066160297], // [east, north]
    ]);
    expect(call.options).toEqual({ padding: 24, duration: 0, maxZoom: MAX_FIT_ZOOM });
  });

  it("never frames closer than the basemap has tiles for - a point-sized AOI keeps its context", () => {
    // Live: "India Gate, New Delhi" geocoded to a box a few metres wide, the
    // camera went to z20, and every z20 OSM tile failed (no CORS on the error
    // page) - a blank basemap and a console full of errors in a demo.
    const created: FakeMap[] = [];
    const createMap = vi.fn(() => {
      const map = new FakeMap();
      created.push(map);
      return map;
    });
    render(
      <MapPanel
        aoi={{ west: 77.22949, south: 28.61293, east: 77.22951, north: 28.61295, scene_id: null }}
        createMap={createMap}
      />,
    );
    act(() => created.forEach((map) => map.emit("load")));

    const [call] = created[0].fitBoundsCalls;
    expect(call.options).toMatchObject({ maxZoom: MAX_FIT_ZOOM });
    expect(MAX_FIT_ZOOM).toBeLessThanOrEqual(BASEMAP_MAX_ZOOM);
    expect(BASEMAP_MAX_ZOOM).toBe(19); // OpenStreetMap standard tiles
  });

  it("frames from the corners, not from any other extent", () => {
    const corners = [
      [10, 50],
      [20, 51],
      [21, 40],
      [11, 39],
    ];
    const { created } = renderMap({
      imagery: imagery({ corners_wgs84: corners }),
    });

    expect(created[0].fitBoundsCalls[0].bounds).toEqual([
      [10, 39],
      [21, 51],
    ]);
  });

  it("does not fit the view when there is no imagery", () => {
    const { created } = renderMap();
    expect(created[0].fitBoundsCalls).toHaveLength(0);
  });

  it("does not fit the view when the footprint is invalid", () => {
    const { created } = renderMap({
      imagery: imagery({ corners_wgs84: [[181, 13], [80, 13], [80, 12], [79, 12]] }),
    });
    expect(created[0].fitBoundsCalls).toHaveLength(0);
  });

  it("does not fit the view when corners are null", () => {
    const { created } = renderMap({ imagery: imagery({ corners_wgs84: null }) });
    expect(created[0].fitBoundsCalls).toHaveLength(0);
  });

  it("re-frames when the imagery changes", () => {
    const created: FakeMap[] = [];
    const createMap = vi.fn(() => {
      const map = new FakeMap();
      created.push(map);
      return map;
    });
    const { rerender } = render(
      <MapPanel imagery={imagery()} createMap={createMap} />,
    );
    act(() => created.forEach((m) => m.emit("load")));
    rerender(
      <MapPanel
        imagery={imagery({
          corners_wgs84: [
            [10, 50],
            [20, 51],
            [21, 40],
            [11, 39],
          ],
        })}
        createMap={createMap}
      />,
    );

    expect(created[0].fitBoundsCalls).toHaveLength(2);
    expect(created[0].fitBoundsCalls[1].bounds).toEqual([
      [10, 39],
      [21, 51],
    ]);
  });
});

describe("MapPanel status reporting", () => {
  it("still names the retrieved scene when the map cannot be drawn", () => {
    // Imagery reaching this panel is a fact worth stating even when WebGL is
    // missing: the retrieval succeeded, only the drawing did not.
    const createMap = vi.fn(() => {
      throw new Error("WebGL2 is required to display this map");
    });
    render(<MapPanel imagery={imagery()} createMap={createMap} />);

    expect(screen.getByRole("status")).toHaveTextContent(
      /S2B_44PMV_20250104_0_L2A/,
    );
  });

  it("says nothing about a scene when there is none", () => {
    const createMap = vi.fn(() => {
      throw new Error("WebGL2 is required to display this map");
    });
    render(<MapPanel imagery={null} createMap={createMap} />);

    expect(screen.getByRole("status")).toHaveTextContent(/WebGL2/i);
    expect(screen.getByRole("status")).not.toHaveTextContent(/scene/i);
  });
});

// =========================================================================== //
// Phase 17.1 - the NDWI overlay
// =========================================================================== //
//
// A second image source on the same map, positioned by its own corners. It is
// a different raster from a different grid, so it never borrows the RGB
// footprint - and either overlay can be present without the other.

const NDWI_CORNERS: number[][] = [
  [80.2, 13.06],
  [80.29, 13.061],
  [80.291, 13.03],
  [80.201, 13.029],
];

function ndwi(overrides: Partial<MapNdwi> = {}): MapNdwi {
  return {
    scene_id: "S2B_44PMV_20250104_0_L2A",
    media_type: "image/png",
    image_base64: "TkRXSQ==",
    corners_wgs84: NDWI_CORNERS,
    ...overrides,
  };
}

function renderWith(props: {
  imagery?: MapImagery | null;
  ndwi?: MapNdwi | null;
}) {
  const created: FakeMap[] = [];
  const createMap = vi.fn(() => {
    const map = new FakeMap();
    created.push(map);
    return map;
  });
  const view = render(
    <MapPanel
      imagery={props.imagery ?? null}
      ndwi={props.ndwi ?? null}
      createMap={createMap}
    />,
  );
  act(() => created.forEach((m) => m.emit("load")));
  return { created, createMap, view };
}

describe("MapPanel NDWI overlay", () => {
  it("adds exactly one NDWI image source and layer", () => {
    const { created } = renderWith({ ndwi: ndwi() });
    const map = created[0];

    expect(map.getSource(NDWI_SOURCE_ID)).toBeDefined();
    expect(map.getLayer(NDWI_LAYER_ID)).toBeDefined();
    // +1 for the scene footprint outline drawn from the same corners.
    expect(map.sources.size).toBe(2);
    expect(map.layers.size).toBe(2);
  });

  it("passes the NDWI corners through unchanged", () => {
    const { created } = renderWith({ ndwi: ndwi() });
    const source = created[0].getSource(NDWI_SOURCE_ID) as {
      coordinates: number[][];
      url: string;
      type: string;
    };

    expect(source.type).toBe("image");
    expect(source.coordinates).toEqual(NDWI_CORNERS);
    expect(source.url).toBe("data:image/png;base64,TkRXSQ==");
  });

  it("keeps RGB and NDWI as separate sources", () => {
    const { created } = renderWith({ imagery: imagery(), ndwi: ndwi() });
    const map = created[0];

    // +1 for the scene footprint outline drawn from the same corners.
    expect(map.sources.size).toBe(3);
    expect(map.layers.size).toBe(3);
    const rgb = map.getSource(SATELLITE_SOURCE_ID) as { coordinates: number[][] };
    const index = map.getSource(NDWI_SOURCE_ID) as { coordinates: number[][] };
    // Each is positioned by its OWN footprint.
    expect(rgb.coordinates).toEqual(CORNERS);
    expect(index.coordinates).toEqual(NDWI_CORNERS);
  });

  it("removes the NDWI overlay when it becomes null", () => {
    const created: FakeMap[] = [];
    const createMap = vi.fn(() => {
      const map = new FakeMap();
      created.push(map);
      return map;
    });
    const { rerender } = render(
      <MapPanel imagery={null} ndwi={ndwi()} createMap={createMap} />,
    );
    act(() => created.forEach((m) => m.emit("load")));
    rerender(<MapPanel imagery={null} ndwi={null} createMap={createMap} />);

    expect(created[0].getSource(NDWI_SOURCE_ID)).toBeUndefined();
    // The footprint source is kept and re-fed rather than destroyed, so it
    // remains - drawing nothing.
    expect(footprintRing(created[0])).toEqual([]);
  });

  it("replaces a stale NDWI overlay rather than stacking one", () => {
    const created: FakeMap[] = [];
    const createMap = vi.fn(() => {
      const map = new FakeMap();
      created.push(map);
      return map;
    });
    const { rerender } = render(
      <MapPanel imagery={null} ndwi={ndwi()} createMap={createMap} />,
    );
    act(() => created.forEach((m) => m.emit("load")));
    rerender(
      <MapPanel
        imagery={null}
        ndwi={ndwi({ image_base64: "TkVX", scene_id: "second" })}
        createMap={createMap}
      />,
    );

    const map = created[0];
    // +1 for the scene footprint outline drawn from the same corners.
    expect(map.sources.size).toBe(2);
    expect(map.calls).toContain(`removeLayer:${NDWI_LAYER_ID}`);
    expect((map.getSource(NDWI_SOURCE_ID) as { url: string }).url).toBe(
      "data:image/png;base64,TkVX",
    );
  });

  it.each([
    ["null corners", null],
    ["three corners", [NDWI_CORNERS[0], NDWI_CORNERS[1], NDWI_CORNERS[2]]],
    ["a NaN", [[NaN, 13], NDWI_CORNERS[1], NDWI_CORNERS[2], NDWI_CORNERS[3]]],
    ["latitude out of range", [[80, 91], NDWI_CORNERS[1], NDWI_CORNERS[2], NDWI_CORNERS[3]]],
  ])("renders no NDWI overlay for %s", (_label, corners) => {
    const { created } = renderWith({
      ndwi: ndwi({ corners_wgs84: corners as number[][] }),
    });
    expect(created[0].getSource(NDWI_SOURCE_ID)).toBeUndefined();
  });

  it("keeps the RGB overlay when the NDWI footprint is malformed", () => {
    const { created } = renderWith({
      imagery: imagery(),
      ndwi: ndwi({ corners_wgs84: null }),
    });

    expect(created[0].getSource(SATELLITE_SOURCE_ID)).toBeDefined();
    expect(created[0].getSource(NDWI_SOURCE_ID)).toBeUndefined();
  });

  it("frames to the NDWI footprint when it is present", () => {
    const { created } = renderWith({ imagery: imagery(), ndwi: ndwi() });
    const last = created[0].fitBoundsCalls.at(-1);

    // NDWI is what the user asked to see; its extent wins.
    expect(last?.bounds).toEqual([
      [80.2, 13.029],
      [80.291, 13.061],
    ]);
  });

  it("still frames to the RGB footprint when there is no NDWI", () => {
    const { created } = renderWith({ imagery: imagery() });
    expect(created[0].fitBoundsCalls.at(-1)?.bounds).toEqual([
      [80.279621036, 13.039005833],
      [80.29002857, 13.066160297],
    ]);
  });

  it("names the NDWI scene in the status line", () => {
    renderWith({ ndwi: ndwi() });
    expect(screen.getByText(/Showing the NDWI index/i)).toBeInTheDocument();
  });
});

// =========================================================================== //
// Phase 17.3 - the temporal NDWI change overlay
// =========================================================================== //
//
// A third raster on the same map, through the same shared primitive. It is a
// different grid again, so it carries and is placed by its own corners.

const CHANGE_CORNERS: number[][] = [
  [80.1, 13.1],
  [80.3, 13.101],
  [80.301, 12.9],
  [80.101, 12.899],
];

function change(overrides: Partial<MapNdwi> = {}): MapNdwi {
  return {
    scene_id: "S2B_TARGET",
    media_type: "image/png",
    image_base64: "Q0hBTkdF",
    corners_wgs84: CHANGE_CORNERS,
    ...overrides,
  };
}

function renderAll(props: {
  imagery?: MapImagery | null;
  ndwi?: MapNdwi | null;
  change?: MapNdwi | null;
}) {
  const created: FakeMap[] = [];
  const createMap = vi.fn(() => {
    const map = new FakeMap();
    created.push(map);
    return map;
  });
  render(
    <MapPanel
      imagery={props.imagery ?? null}
      ndwi={props.ndwi ?? null}
      change={props.change ?? null}
      createMap={createMap}
    />,
  );
  act(() => created.forEach((m) => m.emit("load")));
  return { created, createMap };
}

describe("MapPanel temporal change overlay", () => {
  it("adds exactly one change source and layer", () => {
    const { created } = renderAll({ change: change() });
    const map = created[0];

    expect(map.getSource(CHANGE_SOURCE_ID)).toBeDefined();
    expect(map.getLayer(CHANGE_LAYER_ID)).toBeDefined();
    // +1 for the scene footprint outline drawn from the same corners.
    expect(map.sources.size).toBe(2);
  });

  it("passes the change corners through unchanged", () => {
    const { created } = renderAll({ change: change() });
    const source = created[0].getSource(CHANGE_SOURCE_ID) as {
      coordinates: number[][];
      url: string;
    };

    expect(source.coordinates).toEqual(CHANGE_CORNERS);
    expect(source.url).toBe("data:image/png;base64,Q0hBTkdF");
  });

  it("removes the change overlay when it becomes null", () => {
    const created: FakeMap[] = [];
    const createMap = vi.fn(() => {
      const map = new FakeMap();
      created.push(map);
      return map;
    });
    const { rerender } = render(
      <MapPanel imagery={null} ndwi={null} change={change()} createMap={createMap} />,
    );
    act(() => created.forEach((m) => m.emit("load")));
    rerender(
      <MapPanel imagery={null} ndwi={null} change={null} createMap={createMap} />,
    );

    expect(created[0].getSource(CHANGE_SOURCE_ID)).toBeUndefined();
    // The footprint source is kept and re-fed rather than destroyed, so it
    // remains - drawing nothing.
    expect(footprintRing(created[0])).toEqual([]);
  });

  it("replaces rather than stacks a change overlay", () => {
    const created: FakeMap[] = [];
    const createMap = vi.fn(() => {
      const map = new FakeMap();
      created.push(map);
      return map;
    });
    const { rerender } = render(
      <MapPanel imagery={null} ndwi={null} change={change()} createMap={createMap} />,
    );
    act(() => created.forEach((m) => m.emit("load")));
    rerender(
      <MapPanel
        imagery={null}
        ndwi={null}
        change={change({ image_base64: "TkVX" })}
        createMap={createMap}
      />,
    );

    // +1 for the scene footprint outline drawn from the same corners.
    expect(created[0].sources.size).toBe(2);
    expect(created[0].calls).toContain(`removeLayer:${CHANGE_LAYER_ID}`);
    expect((created[0].getSource(CHANGE_SOURCE_ID) as { url: string }).url).toBe(
      "data:image/png;base64,TkVX",
    );
  });

  it("coexists with RGB, each on its own footprint", () => {
    const { created } = renderAll({ imagery: imagery(), change: change() });
    const map = created[0];

    // +1 for the scene footprint outline drawn from the same corners.
    expect(map.sources.size).toBe(3);
    expect(
      (map.getSource(SATELLITE_SOURCE_ID) as { coordinates: number[][] }).coordinates,
    ).toEqual(CORNERS);
    expect(
      (map.getSource(CHANGE_SOURCE_ID) as { coordinates: number[][] }).coordinates,
    ).toEqual(CHANGE_CORNERS);
  });

  it("frames to the change footprint when it is present", () => {
    const { created } = renderAll({
      imagery: imagery(),
      ndwi: ndwi(),
      change: change(),
    });

    expect(created[0].fitBoundsCalls.at(-1)?.bounds).toEqual([
      [80.1, 12.899],
      [80.301, 13.101],
    ]);
  });

  it("rejects a malformed change footprint without touching the others", () => {
    const { created } = renderAll({
      imagery: imagery(),
      change: change({ corners_wgs84: [[181, 13], [80, 13], [80, 12], [79, 12]] }),
    });

    expect(created[0].getSource(CHANGE_SOURCE_ID)).toBeUndefined();
    expect(created[0].getSource(SATELLITE_SOURCE_ID)).toBeDefined();
  });

  it("leaves the single-scene NDWI overlay unchanged", () => {
    const { created } = renderAll({ ndwi: ndwi(), change: change() });
    const map = created[0];

    expect(map.getSource(NDWI_SOURCE_ID)).toBeDefined();
    expect(
      (map.getSource(NDWI_SOURCE_ID) as { coordinates: number[][] }).coordinates,
    ).toEqual(NDWI_CORNERS);
  });
});

describe("MapPanel container resizing", () => {
  it("re-measures the map when its container changes size", () => {
    // The scene spans the full grid width until an analysis appears beside it.
    // MapLibre sizes its canvas once at creation, so without this the view
    // shows a stretched, far wider extent than the one requested.
    const created: FakeMap[] = [];
    const observed: Element[] = [];
    class FakeResizeObserver {
      constructor(private readonly cb: () => void) {}
      observe(el: Element) {
        observed.push(el);
        this.cb();
      }
      disconnect() {}
      unobserve() {}
    }
    vi.stubGlobal("ResizeObserver", FakeResizeObserver);

    const createMap = vi.fn(() => {
      const map = new FakeMap();
      created.push(map);
      return map;
    });
    render(<MapPanel imagery={null} createMap={createMap} />);

    expect(observed).toHaveLength(1);
    expect(created[0].resizes).toBeGreaterThan(0);
  });

  it("still works where ResizeObserver is unavailable", () => {
    vi.stubGlobal("ResizeObserver", undefined);
    const createMap = vi.fn(() => new FakeMap());
    expect(() =>
      render(<MapPanel imagery={null} createMap={createMap} />),
    ).not.toThrow();
  });
});

// --------------------------------------------------------------------------- #
// Scene footprint outline (Direction B §7)
// --------------------------------------------------------------------------- #
//
// The outline is drawn from the SAME four corners that position the raster, so
// it can never disagree with the picture it frames. These pin that identity —
// a footprint derived some other way would be a second, unverified geometry.

describe("MapPanel scene footprint", () => {
  it("outlines the footprint using the imagery's own corners", () => {
    const { created } = renderMap({ imagery: imagery() });
    const source = created[0].getSource(FOOTPRINT_SOURCE_ID) as {
      type: string;
      data: { geometry: { type: string; coordinates: number[][] } };
    };

    expect(source.type).toBe("geojson");
    expect(source.data.geometry.type).toBe("LineString");
    // Four corners in the order received, closed back to the first.
    expect(source.data.geometry.coordinates).toEqual([...CORNERS, CORNERS[0]]);
    expect(created[0].getLayer(FOOTPRINT_LAYER_ID)).toBeDefined();
  });

  it("draws no outline when there is no imagery", () => {
    const { created } = renderMap();
    expect(created[0].getSource(FOOTPRINT_SOURCE_ID)).toBeUndefined();
    expect(created[0].getLayer(FOOTPRINT_LAYER_ID)).toBeUndefined();
  });

  it("draws no outline for a footprint that failed validation", () => {
    const { created } = renderMap({
      imagery: imagery({ corners_wgs84: [[80, 13]] }),
    });
    expect(created[0].getSource(FOOTPRINT_SOURCE_ID)).toBeUndefined();
  });

  it("removes the outline when the imagery is cleared", () => {
    const created: FakeMap[] = [];
    const createMap = vi.fn(() => {
      const map = new FakeMap();
      created.push(map);
      return map;
    });
    const { rerender } = render(
      <MapPanel imagery={imagery()} createMap={createMap} />,
    );
    act(() => created.forEach((map) => map.emit("load")));
    expect(created[0].getLayer(FOOTPRINT_LAYER_ID)).toBeDefined();

    rerender(<MapPanel imagery={null} createMap={createMap} />);
    expect(footprintRing(created[0])).toEqual([]);
  });
});

// --------------------------------------------------------------------------- #
// Area of interest without a raster
// --------------------------------------------------------------------------- #
//
// A query can analyse a scene without requesting an imagery preview. That run
// succeeded, so the viewport must show WHERE it happened and say why nothing is
// drawn — never "no scene loaded" over a completed analysis.

const AOI = { west: 80.28, south: 13.039, east: 80.29, north: 13.066 };

describe("MapPanel area of interest", () => {
  function renderAoi(aoi: unknown) {
    const created: FakeMap[] = [];
    const createMap = vi.fn(() => {
      const map = new FakeMap();
      created.push(map);
      return map;
    });
    render(
      <MapPanel
        aoi={aoi as never}
        imagery={null}
        createMap={createMap}
      />,
    );
    act(() => created.forEach((map) => map.emit("load")));
    return created;
  }

  it("frames the resolved extent when no raster came back", () => {
    const created = renderAoi(AOI);
    expect(created[0].fitBoundsCalls).toHaveLength(1);
    expect(created[0].fitBoundsCalls[0].bounds).toEqual([
      [AOI.west, AOI.south],
      [AOI.east, AOI.north],
    ]);
  });

  it("names the selected scene rather than claiming none was loaded", () => {
    renderAoi({ ...AOI, scene_id: "S2B_44PMV_20250104_0_L2A" });
    expect(screen.getByText(/No imagery on this map/i)).toBeInTheDocument();
    expect(
      screen.getByText(/S2B_44PMV_20250104_0_L2A was analysed/),
    ).toBeInTheDocument();
    expect(screen.queryByText(/No scene loaded/i)).not.toBeInTheDocument();
  });

  it("still says no scene loaded when nothing has run", () => {
    renderAoi(null);
    expect(screen.getByText(/No scene loaded/i)).toBeInTheDocument();
  });

  it("lifts the basemap when there is no raster and sinks it when there is", () => {
    const created = renderAoi(AOI);
    const lifted = created[0].paint.find(
      (p) => p.name === "raster-brightness-max",
    );
    expect(lifted?.value).toBe(0.72);
    // Sunk with a raster, lifted without: the ground never outshines the scene,
    // and never disappears when it is the only geography present.
    expect(lifted?.value).toBeGreaterThan(0.44);

    const withRaster: FakeMap[] = [];
    const createMap = vi.fn(() => {
      const map = new FakeMap();
      withRaster.push(map);
      return map;
    });
    render(<MapPanel imagery={imagery()} createMap={createMap} />);
    act(() => withRaster.forEach((map) => map.emit("load")));
    const sunk = withRaster[0].paint.find(
      (p) => p.name === "raster-brightness-max",
    );
    expect(sunk?.value).toBe(0.44);
  });
});

// The empty-state copy promises an outlined extent; it must actually be drawn.
describe("MapPanel area outline without a raster", () => {
  it("outlines the resolved extent when no raster came back", () => {
    const created: FakeMap[] = [];
    const createMap = vi.fn(() => {
      const map = new FakeMap();
      created.push(map);
      return map;
    });
    render(
      <MapPanel aoi={AOI as never} imagery={null} createMap={createMap} />,
    );
    act(() => created.forEach((map) => map.emit("load")));

    const source = created[0].getSource(FOOTPRINT_SOURCE_ID) as {
      data: { geometry: { coordinates: number[][] } };
    };
    expect(source).toBeDefined();
    // Closed ring over the resolved bbox, NW first.
    expect(source.data.geometry.coordinates).toEqual([
      [AOI.west, AOI.north],
      [AOI.east, AOI.north],
      [AOI.east, AOI.south],
      [AOI.west, AOI.south],
      [AOI.west, AOI.north],
    ]);
  });
});

// The locator draws on a pale basemap, where the over-imagery cyan vanishes.
describe("MapPanel locator footprint", () => {
  function renderLocator() {
    const created: FakeMap[] = [];
    const createMap = vi.fn(() => {
      const map = new FakeMap();
      created.push(map);
      return map;
    });
    render(
      <MapPanel aoi={AOI as never} variant="locator" createMap={createMap} />,
    );
    act(() => created.forEach((map) => map.emit("load")));
    return created[0];
  }

  it("draws the extent in geospatial blue, heavier than over imagery", () => {
    const map = renderLocator();
    const outline = map.getLayer(FOOTPRINT_LAYER_ID) as {
      paint: Record<string, unknown>;
      type: string;
    };
    expect(outline.type).toBe("line");
    expect(outline.paint["line-color"]).toBe("#0B6BCB");
    expect(outline.paint["line-width"]).toBe(2.5);
  });

  it("uses a closed LineString, which is the geometry that renders", () => {
    // A Polygon with a fill layer was tried here and drew nothing on a real
    // MapLibre map; this pins the geometry that actually works.
    const map = renderLocator();
    const source = map.getSource(FOOTPRINT_SOURCE_ID) as {
      data: { geometry: { type: string; coordinates: number[][] } };
    };
    expect(source.data.geometry.type).toBe("LineString");
    const ring = source.data.geometry.coordinates;
    expect(ring).toHaveLength(5);
    expect(ring[0]).toEqual(ring[4]);
  });

  it("draws no raster, whatever it is handed", () => {
    const created: FakeMap[] = [];
    const createMap = vi.fn(() => {
      const map = new FakeMap();
      created.push(map);
      return map;
    });
    render(
      <MapPanel
        aoi={AOI as never}
        imagery={imagery()}
        variant="locator"
        createMap={createMap}
      />,
    );
    act(() => created.forEach((map) => map.emit("load")));
    expect(created[0].getLayer(SATELLITE_LAYER_ID)).toBeUndefined();
  });

  it("keeps the over-imagery outline cyan", () => {
    const created: FakeMap[] = [];
    const createMap = vi.fn(() => {
      const map = new FakeMap();
      created.push(map);
      return map;
    });
    render(<MapPanel imagery={imagery()} createMap={createMap} />);
    act(() => created.forEach((map) => map.emit("load")));
    const outline = created[0].getLayer(FOOTPRINT_LAYER_ID) as {
      paint: Record<string, unknown>;
    };
    expect(outline.paint["line-color"]).toBe("#5FD3F3");
    expect(outline.paint["line-width"]).toBe(1.25);
  });
});

// A resolved area is real information: the frame must say so rather than
// reporting an empty map over an outline it is actually drawing.
describe("MapPanel resolved area without a scene", () => {
  it("reports the area as resolved, not as nothing loaded", () => {
    const created: FakeMap[] = [];
    const createMap = vi.fn(() => {
      const map = new FakeMap();
      created.push(map);
      return map;
    });
    render(
      <MapPanel aoi={AOI as never} imagery={null} createMap={createMap} />,
    );
    act(() => created.forEach((map) => map.emit("load")));

    expect(screen.getByText(/Area resolved/i)).toBeInTheDocument();
    expect(screen.queryByText(/No scene loaded/i)).not.toBeInTheDocument();
    // And the extent it describes is actually drawn.
    expect(created[0].getLayer(FOOTPRINT_LAYER_ID)).toBeDefined();
  });
});


describe("MapPanel SAR provenance", () => {
  it.each(["vv", "vh"])("labels the actual %s display without claiming amplitude or local calibration", (asset) => {
    renderMap({ imagery: imagery({ asset, bands: [asset.toUpperCase()] }) });
    expect(screen.getByText(`Sentinel-1 SAR · ${asset.toUpperCase()} display`)).toBeInTheDocument();
    expect(screen.queryByText(/GRD|amplitude|true colour/)).not.toBeInTheDocument();
  });
});
