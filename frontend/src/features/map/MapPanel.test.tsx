import { act, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  CHANGE_LAYER_ID,
  CHANGE_SOURCE_ID,
  NDWI_LAYER_ID,
  NDWI_SOURCE_ID,
  SATELLITE_LAYER_ID,
  SATELLITE_SOURCE_ID,
} from "./footprint";
import type { MapImagery, MapLike, MapNdwi } from "./footprint";
import { MapPanel } from "./MapPanel";

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
  removed = false;
  private handlers: Record<string, (() => void)[]> = {};

  on(event: string, handler: () => void): void {
    (this.handlers[event] ??= []).push(handler);
  }
  emit(event: string): void {
    for (const handler of this.handlers[event] ?? []) handler();
  }
  addSource(id: string, source: unknown): void {
    if (this.sources.has(id)) throw new Error(`duplicate source ${id}`);
    this.sources.set(id, source);
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

describe("MapPanel", () => {
  // --- A. renders without imagery ---------------------------------------- #

  it("renders the panel with no imagery", () => {
    renderMap();
    expect(screen.getByRole("heading", { name: /map/i })).toBeInTheDocument();
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

    expect(map.sources.size).toBe(1);
    expect(map.layers.size).toBe(1);
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
    expect(map.sources.size).toBe(1);
    expect(map.layers.size).toBe(1);
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

    expect(created[0].sources.size).toBe(0);
    expect(created[0].layers.size).toBe(0);
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
    expect(created[0].sources.size).toBe(1);
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
    expect(created[0].sources.size).toBe(1);
    expect(created[0].layers.size).toBe(1);
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
    expect(call.options).toEqual({ padding: 24, duration: 0 });
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
    expect(map.sources.size).toBe(1);
    expect(map.layers.size).toBe(1);
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

    expect(map.sources.size).toBe(2);
    expect(map.layers.size).toBe(2);
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
    expect(created[0].sources.size).toBe(0);
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
    expect(map.sources.size).toBe(1);
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
    expect(screen.getByText(/NDWI/i)).toBeInTheDocument();
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
    expect(map.sources.size).toBe(1);
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
    expect(created[0].sources.size).toBe(0);
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

    expect(created[0].sources.size).toBe(1);
    expect(created[0].calls).toContain(`removeLayer:${CHANGE_LAYER_ID}`);
    expect((created[0].getSource(CHANGE_SOURCE_ID) as { url: string }).url).toBe(
      "data:image/png;base64,TkVX",
    );
  });

  it("coexists with RGB, each on its own footprint", () => {
    const { created } = renderAll({ imagery: imagery(), change: change() });
    const map = created[0];

    expect(map.sources.size).toBe(2);
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
