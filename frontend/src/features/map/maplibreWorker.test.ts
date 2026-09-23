/**
 * The production map must ship the worker it asks for.
 *
 * MapLibre resolves its worker relative to its own module URL. Rolled into the
 * application chunk by a production build, that resolves to
 * `/assets/maplibre-gl-worker.mjs` - an asset Vite never emitted, because
 * nothing imported it. A single-page host answers that request with
 * `index.html`, so the browser loads HTML as JavaScript. The basemap survives
 * (it needs no worker) while GeoJSON-backed footprint geometry never renders,
 * which is why the failure reads as a partially working map.
 *
 * Two things are pinned here, because either alone can regress silently:
 *
 *   1. the URL is STATED to MapLibre through `setWorkerUrl`, exactly once;
 *   2. it comes from a `?worker&url` import, which is what makes Vite bundle
 *      the worker with its dependencies and emit it as a real asset.
 *
 * (2) is asserted against the module's own source because the bug is a
 * PACKAGING one: `?url` alone type-checks, runs in development and produces a
 * worker whose internal import 404s in production. A unit test cannot see that;
 * the import mechanism is the thing that has to hold. The end-to-end proof is
 * the production-build browser check, which loads the served bundle and
 * verifies the worker response is JavaScript and the footprint layer draws.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";

const { setWorkerUrl } = vi.hoisted(() => ({ setWorkerUrl: vi.fn() }));
vi.mock("maplibre-gl", () => ({ setWorkerUrl }));

import { configureMapLibreWorker, mapLibreWorkerUrl } from "./maplibreWorker";
// Read through Vite rather than the filesystem: under vite-node
// `import.meta.url` is not a `file:` URL, so `fileURLToPath` cannot resolve it.
import SOURCE from "./maplibreWorker.ts?raw";

describe("MapLibre worker packaging", () => {
  beforeEach(() => {
    setWorkerUrl.mockClear();
  });

  it("hands MapLibre a URL rather than leaving it to guess one", () => {
    configureMapLibreWorker();

    expect(setWorkerUrl).toHaveBeenCalledTimes(1);
    const [url] = setWorkerUrl.mock.calls[0] as [string];
    expect(typeof url).toBe("string");
    expect(url.length).toBeGreaterThan(0);
    expect(url).toBe(mapLibreWorkerUrl());
  });

  it("configures the worker once, however many maps are built", () => {
    configureMapLibreWorker();
    configureMapLibreWorker();
    configureMapLibreWorker();

    // Already configured by the first test in this module's lifetime, so the
    // assertion that matters is that repeated calls add nothing.
    expect(setWorkerUrl.mock.calls.length).toBeLessThanOrEqual(1);
  });

  it("imports the worker through Vite's bundling query, not as a bare file", () => {
    // `?worker&url` bundles the worker WITH its dependencies and returns the
    // emitted asset's URL. `?url` would copy a single file whose own import of
    // MapLibre's shared chunk then 404s in production.
    expect(SOURCE).toContain("maplibre-gl-worker.mjs?worker&url");
    expect(SOURCE).toContain("setWorkerUrl");
  });
});
