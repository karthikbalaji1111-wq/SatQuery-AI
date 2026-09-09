import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      // Two entries: the frozen workspace at `/` and the landing page at
      // `/landing.html`. Separate bundles, so three.js never reaches the app.
      input: {
        app: "index.html",
        landing: "landing.html",
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
  },
});
