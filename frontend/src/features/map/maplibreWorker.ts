/**
 * Point MapLibre at a worker URL this build actually ships.
 *
 * **The defect this closes.** MapLibre resolves its worker itself, relative to
 * its own module URL:
 *
 *     const url = import.meta.url;                       // maplibre's own module
 *     new URL("./maplibre-gl-worker.mjs", url).href      // the worker beside it
 *
 * In development that resolves to `/node_modules/maplibre-gl/dist/...`, which
 * the dev server happily serves, so the map works. In a production build
 * MapLibre's code has been rolled into the application chunk, so `import.meta.url`
 * is `/assets/app-<hash>.js` and the worker resolves to
 * `/assets/maplibre-gl-worker.mjs` - **a file Vite never emitted**, because
 * nothing in the graph imports it as an asset. A single-page host answers that
 * request with the SPA's `index.html`, so the browser loads HTML as JavaScript
 * and every worker-backed feature fails while the basemap - which needs no
 * worker - keeps rendering. That is why this looked like a working map with
 * missing footprint geometry rather than a broken one.
 *
 * **Why `?worker&url`.** The worker is a real ES module that imports MapLibre's
 * shared chunk, so copying the single file with `?url` would emit a script whose
 * own import 404s. `?worker&url` makes Vite bundle the worker WITH its
 * dependencies and hand back the emitted asset's hashed URL, which is then
 * stated explicitly through MapLibre's own public `setWorkerUrl`. Nothing here
 * patches or forks MapLibre: it is the configuration hook the library documents
 * for exactly this case.
 */

import { setWorkerUrl } from "maplibre-gl";
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";

let configured = false;

/**
 * Called once, immediately before the first real map is constructed.
 *
 * Idempotent, and deliberately not executed at module scope: a test that
 * injects its own map factory never builds a real map, and should not be made
 * to depend on worker configuration it will not use.
 */
export function configureMapLibreWorker(): void {
  if (configured) return;
  setWorkerUrl(workerUrl);
  configured = true;
}

/** The URL this build will hand MapLibre. Exported so a test can assert it. */
export function mapLibreWorkerUrl(): string {
  return workerUrl;
}
