/**
 * The public front door and the way through it.
 *
 * "/" is the cinematic landing (index.html) and "/app" is the real workspace
 * (app.html). Vercel gives the filesystem precedence over rewrites, so the
 * landing can only own "/" by BEING index.html - which is why these tests pin
 * the files, the build entries and the Vercel config together: change one and
 * the front door silently becomes the workspace again (or a dead link).
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.mock("./EarthCanvas", () => ({ EarthCanvas: () => null }));

import { Landing } from "./Landing";

const root = resolve(__dirname, "../..");
const read = (file: string) => readFileSync(resolve(root, file), "utf8");

describe("front door", () => {
  it("serves the cinematic landing at / and the workspace at /app", () => {
    expect(read("index.html")).toContain('src="/src/landing/main.tsx"');
    expect(read("app.html")).toContain('src="/src/main.tsx"');
  });

  it("builds both entries", () => {
    const config = read("vite.config.ts");
    expect(config).toContain('app: "app.html"');
    expect(config).toContain('landing: "index.html"');
  });

  it("maps /app to app.html on Vercel", () => {
    const vercel = JSON.parse(read("vercel.json"));
    expect(vercel.cleanUrls).toBe(true);
  });

  it("sends every Try SatQuery call to action into the workspace", () => {
    vi.stubGlobal(
      "matchMedia",
      vi.fn().mockReturnValue({
        matches: true,
        addEventListener: () => {},
        removeEventListener: () => {},
      }),
    );
    render(<Landing />);
    const ctas = screen.getAllByRole("link", { name: "Try SatQuery" });
    expect(ctas.length).toBeGreaterThan(0);
    for (const cta of ctas) {
      expect(cta).toHaveAttribute("href", "/app");
    }
  });
});
