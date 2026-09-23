import { describe, expect, it } from "vitest";
import { requireProductionApiUrl } from "./production";

describe("production API URL", () => {
  it.each([
    undefined, "", "http://api.example.com", "https://localhost", "https://app.localhost",
    "https://localhost.", "https://127.0.0.1", "https://[::1]", "https://10.0.0.1",
    "https://backend", "https://host.docker.internal", "https://api.local",
    "https://user:secret@api.example.com", "https://api.example.com?key=secret",
    "https://api.example.com#fragment", "/api", " https://api.example.com",
  ])("rejects unsafe or absent value %s", (value) => {
    expect(() => requireProductionApiUrl(value)).toThrow("VITE_API_BASE_URL");
  });
  it.each(["https://api.example.com", "https://api.example.com:8443/service"])(
    "accepts explicit HTTPS endpoint %s", (value) => {
      expect(() => requireProductionApiUrl(value)).not.toThrow();
    },
  );
});
