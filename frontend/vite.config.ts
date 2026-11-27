import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";
import { loadEnv } from "vite";
import { requireProductionApiUrl } from "./src/config/production";

export default defineConfig(({ command, mode }) => {
  if (command === "build") {
    requireProductionApiUrl(loadEnv(mode, process.cwd(), "VITE_").VITE_API_BASE_URL);
  }
  return {
    plugins: [react()],
    build: {
      // The largest chunk is MapLibre itself (about 970 kB minified, 256 kB
      // gzip), split out below; nothing of ours comes near this.
      chunkSizeWarningLimit: 1000,
      rollupOptions: {
        // Separate entries keep three.js out of the workspace bundle. The
        // cinematic landing is the front door at "/" (index.html); the real
        // workspace is served at "/app" (app.html).
        input: {
          app: "app.html",
          landing: "index.html",
        },
        output: {
          // MapLibre is most of the workspace's weight and changes far less
          // often than the app's own code; its own chunk stays cached across
          // releases and keeps the app chunk under Vite's 500 kB warning.
          manualChunks(id: string) {
            if (id.includes("node_modules/maplibre-gl/")) return "maplibre";
            return undefined;
          },
        },
      },
    },
    server: {
      port: 5173,
      host: true,
    },
    test: {
      environment: "jsdom",
      globals: true,
      setupFiles: ["./src/test/setup.ts"],
      css: false,
      coverage: {
        provider: "v8",
        include: ["src/**/*.{ts,tsx}"],
        exclude: ["src/**/*.test.{ts,tsx}", "src/test/**", "src/**/m6Fixtures.ts"],
        // Measured 2026-11-27 over every source file: lines 83.8%, branches
        // 85.9%, functions 91.0%, statements 82.7%. CI fails below these
        // floors (applied only with --coverage).
        thresholds: { lines: 82, branches: 84, functions: 89, statements: 81 },
      },
    },
  };
});
