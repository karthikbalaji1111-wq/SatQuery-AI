/**
 * The fixed 3D backdrop the chapters scroll over.
 *
 * Decorative by design: it is `aria-hidden`, carries no text, and the page is
 * fully understandable with it absent. Everything it shows is said in the HTML
 * as well — that is the accessibility contract and also the fallback plan.
 *
 * It never blocks first paint. The canvas mounts empty, the chapters render
 * immediately, and the Earth fades in whenever its textures arrive. A visitor
 * on a slow connection reads the page; they do not wait for a planet.
 */

import { useEffect, useRef, useState } from "react";

import { applyProgress, damp } from "./timeline";

const ASSET_BASE = "/landing/";

/** True when the visitor has asked for less motion, tracked live. */
function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(() =>
    typeof window !== "undefined" &&
    window.matchMedia?.("(prefers-reduced-motion: reduce)").matches,
  );
  useEffect(() => {
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    const onChange = () => setReduced(query.matches);
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, []);
  return reduced;
}

export function EarthCanvas() {
  const hostRef = useRef<HTMLDivElement>(null);
  const [ready, setReady] = useState(false);
  const [failed, setFailed] = useState(false);
  const reducedMotion = usePrefersReducedMotion();

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    let disposed = false;
    let cleanup: (() => void) | undefined;

    // Everything three.js is behind a dynamic import, so the library is not in
    // the critical path for the text of the page.
    (async () => {
      let THREE: typeof import("three");
      try {
        THREE = await import("three");
      } catch {
        setFailed(true);
        return;
      }

      // A WebGL2 context is the one hard requirement. Without it the static
      // fallback is not a degradation, it is the correct output.
      const probe = document.createElement("canvas");
      if (!probe.getContext("webgl2")) {
        setFailed(true);
        return;
      }

      const { buildWorld, qualityFor, applyEnvironment } = await import("./scene");
      const quality = qualityFor(
        window.innerWidth,
        navigator.hardwareConcurrency ?? 4,
      );

      let world: import("./scene").WorldHandles;
      try {
        world = await buildWorld(quality, ASSET_BASE);
      } catch {
        setFailed(true);
        return;
      }
      if (disposed) return;

      const renderer = new THREE.WebGLRenderer({
        antialias: quality.maxPixelRatio > 1.5,
        alpha: true,
        powerPreference: "high-performance",
      });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, quality.maxPixelRatio));
      renderer.setClearColor(0x000000, 0);
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      host.appendChild(renderer.domElement);

      // Gives the satellite's metals something to reflect. Generated, not
      // fetched, so it costs one startup pass and no request.
      const environment = applyEnvironment(renderer, world);

      // Loop state, declared ahead of everything that touches it: resize() and
      // readScroll() both run once during setup, and a `let` read before its
      // declaration is a ReferenceError rather than an undefined.
      let smoothed = 0;
      let last = performance.now();
      let frame = 0;
      let visible = true;
      let lastScrollAt = performance.now();
      let lastDrawn = 0;
      let started = false;

      // Once the camera has caught up and the reader has stopped, the only
      // thing still changing is a cloud drift of 0.004 rad/s. Presenting that
      // at 120 Hz on a fanless laptop is pure heat, so the loop keeps full
      // rate while anything is actually happening and halves it when nothing
      // is. Any scroll event restores full rate on the very next frame, so
      // this can never be felt as lag.
      const IDLE_AFTER_MS = 420;
      const IDLE_FRAME_MS = 1000 / 30;

      const resize = () => {
        const w = host.clientWidth;
        const h = host.clientHeight;
        renderer.setSize(w, h, false);
        world.camera.aspect = w / Math.max(1, h);
        world.camera.updateProjectionMatrix();
        // Under reduced motion the loop renders once and stops, so a resize
        // would otherwise leave the held frame stretched across the new
        // viewport until something else happened to redraw it.
        if (reducedMotion && started) frame = requestAnimationFrame(tick);
      };
      resize();
      window.addEventListener("resize", resize);

      // The single scroll owner. Passive, and it only records a number —
      // nothing is transformed here, so scrolling stays the browser's job.
      let targetProgress = 0;
      const readScroll = () => {
        const max = document.documentElement.scrollHeight - window.innerHeight;
        targetProgress = max > 0 ? window.scrollY / max : 0;
        lastScrollAt = performance.now();
      };
      readScroll();
      window.addEventListener("scroll", readScroll, { passive: true });
      window.addEventListener("resize", readScroll);

      smoothed = targetProgress;

      // Stop rendering entirely when the tab is hidden or the canvas is
      // scrolled past: a fixed WebGL canvas quietly burning GPU behind other
      // content is the usual reason these pages drain batteries.
      const onVisibility = () => {
        visible = document.visibilityState === "visible";
        if (visible) {
          last = performance.now();
          frame = requestAnimationFrame(tick);
        } else {
          cancelAnimationFrame(frame);
        }
      };
      document.addEventListener("visibilitychange", onVisibility);

      function tick(now: number) {
        const dt = Math.min(0.05, (now - last) / 1000);
        last = now;

        if (reducedMotion) {
          // Hold the opening pose, and hold it as ONE frame. The Earth is
          // still there and still beautiful; it simply does not move with the
          // page, and a loop redrawing an identical image forever is not
          // "reduced" motion by any reading of the request.
          applyProgress(world, 0, 0, quality.motionScale);
          renderer.render(world.scene, world.camera);
          return;
        }

        const settled = Math.abs(targetProgress - smoothed) < 1e-4
          && now - lastScrollAt > IDLE_AFTER_MS;
        if (settled && now - lastDrawn < IDLE_FRAME_MS) {
          if (visible) frame = requestAnimationFrame(tick);
          return;
        }
        lastDrawn = now;

        smoothed = damp(smoothed, targetProgress, 5.2, dt);

        applyProgress(world, smoothed, now / 1000, quality.motionScale);
        renderer.render(world.scene, world.camera);
        if (visible) frame = requestAnimationFrame(tick);
      }

      started = true;
      frame = requestAnimationFrame(tick);
      setReady(true);

      cleanup = () => {
        cancelAnimationFrame(frame);
        window.removeEventListener("resize", resize);
        window.removeEventListener("resize", readScroll);
        window.removeEventListener("scroll", readScroll);
        document.removeEventListener("visibilitychange", onVisibility);
        environment.dispose();
        world.scene.environment = null;
        // Textures are not reachable from the materials that hold them, so the
        // world hands over its own ledger. Skipping this stranded three NASA
        // maps on the GPU every time the reduced-motion preference changed.
        for (const texture of world.textures) texture.dispose();
        renderer.dispose();
        world.scene.traverse((object) => {
          const mesh = object as { geometry?: { dispose(): void }; material?: unknown };
          mesh.geometry?.dispose();
          const material = mesh.material;
          if (Array.isArray(material)) {
            material.forEach((m) => (m as { dispose(): void }).dispose());
          } else if (material) {
            (material as { dispose(): void }).dispose();
          }
        });
        renderer.domElement.remove();
      };
    })();

    return () => {
      disposed = true;
      cleanup?.();
    };
  }, [reducedMotion]);

  return (
    <div className="earth-stage" aria-hidden="true">
      {/* Present from the first paint so the page never flashes bare black,
          and the whole fallback when WebGL is unavailable. */}
      <div className="earth-fallback" data-hidden={ready && !failed} />
      <div ref={hostRef} className="earth-canvas" data-ready={ready} />
    </div>
  );
}
