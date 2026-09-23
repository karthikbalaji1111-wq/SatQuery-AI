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
      rollupOptions: {
        // Separate entries keep three.js out of the workspace bundle.
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
  };
});
