/**
 * The Earth's rotation is three axes in a fixed hierarchy, and the hierarchy is
 * the point.
 *
 * A planet that yaws, pitches and rolls by independent amounts does not read as
 * a planet - it reads as an object being shaken. What makes it planetary is
 * that the tilt and the roll are small, fixed FRACTIONS of the spin and come
 * from the same keyframes, so they can never drift out of phase. These tests
 * pin that relationship numerically, because it is exactly the kind of thing a
 * later "let's make the motion more dynamic" edit erodes without anyone
 * noticing until the planet looks wrong.
 */

import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { qualityFor } from "./scene";
import { applyProgress, type Driven } from "./timeline";

/** A Driven whose members are real THREE objects but carry no assets. */
function stubWorld(): Driven {
  const shader = (uniforms: Record<string, { value: number }>) =>
    new THREE.ShaderMaterial({ uniforms });
  return {
    camera: new THREE.PerspectiveCamera(),
    earth: new THREE.Mesh(new THREE.BufferGeometry()),
    clouds: new THREE.Mesh(new THREE.BufferGeometry()),
    atmosphere: new THREE.Mesh(
      new THREE.BufferGeometry(),
      shader({ intensity: { value: 0 } }),
    ),
    satellite: new THREE.Group(),
    solarWings: null,
    sunDirection: new THREE.Vector3(-0.46, 0.20, 0.86).normalize(),
    orbit: new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial()),
    stars: new THREE.Points(
      new THREE.BufferGeometry(),
      shader({ opacity: { value: 0 } }),
    ),
  };
}

const deg = (radians: number) => Math.abs(THREE.MathUtils.radToDeg(radians));

/** Sampled inside the journey; the endpoints are deliberately neutral. */
const MID_JOURNEY = [0.17, 0.35, 0.52, 0.68, 0.84];

describe("Earth rotation hierarchy", () => {
  it("keeps the tilt a documented fraction of the spin, never its equal", () => {
    for (const p of MID_JOURNEY) {
      const w = stubWorld();
      applyProgress(w, p, 0);
      const ratio = deg(w.earth.rotation.x) / deg(w.earth.rotation.y);
      expect(ratio).toBeGreaterThanOrEqual(0.05);
      expect(ratio).toBeLessThanOrEqual(0.1);
    }
  });

  it("keeps the roll smaller still", () => {
    for (const p of MID_JOURNEY) {
      const w = stubWorld();
      applyProgress(w, p, 0);
      const ratio = deg(w.earth.rotation.z) / deg(w.earth.rotation.y);
      expect(ratio).toBeGreaterThanOrEqual(0.01);
      expect(ratio).toBeLessThanOrEqual(0.03);
    }
  });

  it("establishes the tilt before the spin, so the planet leans rather than wobbles", () => {
    const w = stubWorld();
    applyProgress(w, 0.52, 0);
    // XYZ is THREE's default, but it is load-bearing here rather than incidental.
    expect(w.earth.rotation.order).toBe("XYZ");
  });

  it("opens and closes on a neutral axis", () => {
    for (const p of [0, 1]) {
      const w = stubWorld();
      applyProgress(w, p, 0);
      expect(deg(w.earth.rotation.x)).toBeCloseTo(0, 6);
      expect(deg(w.earth.rotation.z)).toBeCloseTo(0, 6);
    }
  });

  it("gives the clouds the surface's own tilt and roll", () => {
    const w = stubWorld();
    applyProgress(w, 0.68, 0);
    // Shearing the cloud shell off the coastline is the visible failure here.
    expect(w.clouds!.rotation.x).toBeCloseTo(w.earth.rotation.x, 10);
    expect(w.clouds!.rotation.z).toBeCloseTo(w.earth.rotation.z, 10);
  });
});

describe("responsive motion amplitude", () => {
  it("damps the secondary axes on a small device", () => {
    const full = stubWorld();
    const phone = stubWorld();
    applyProgress(full, 0.68, 0, 1);
    applyProgress(phone, 0.68, 0, 0.35);
    expect(deg(phone.earth.rotation.x)).toBeLessThan(deg(full.earth.rotation.x));
    expect(deg(phone.earth.rotation.z)).toBeLessThan(deg(full.earth.rotation.z));
  });

  it("tells every device the same story", () => {
    const full = stubWorld();
    const phone = stubWorld();
    applyProgress(full, 0.68, 0, 1);
    applyProgress(phone, 0.68, 0, 0.35);
    // Spin, distance and framing are the narrative. Only the carriage changes.
    expect(phone.earth.rotation.y).toBeCloseTo(full.earth.rotation.y, 10);
    expect(phone.camera.position.x).toBeCloseTo(full.camera.position.x, 10);
    expect(phone.camera.position.z).toBeCloseTo(full.camera.position.z, 10);
  });
});

describe("scroll reversibility", () => {
  it("returns to the identical pose when the reader scrolls back up", () => {
    const down = stubWorld();
    const up = stubWorld();
    applyProgress(down, 0.35, 0);
    for (const p of [0.5, 0.8, 0.5, 0.35]) applyProgress(up, p, 0);
    // Pure in progress: the same scroll position is the same frame, whichever
    // direction the reader arrived from.
    expect(up.earth.rotation.x).toBeCloseTo(down.earth.rotation.x, 10);
    expect(up.earth.rotation.y).toBeCloseTo(down.earth.rotation.y, 10);
    expect(up.earth.rotation.z).toBeCloseTo(down.earth.rotation.z, 10);
  });
});

describe("device motion tiers", () => {
  it("gives a phone the least secondary motion and a desktop the most", () => {
    // Eight cores throughout, so width is the variable under test.
    const phone = qualityFor(390, 8);
    const tablet = qualityFor(1024, 8);
    const desktop = qualityFor(1440, 8);
    expect(phone.motionScale).toBeLessThan(tablet.motionScale);
    expect(tablet.motionScale).toBeLessThan(desktop.motionScale);
    expect(desktop.motionScale).toBe(1);
  });

  it("treats a few cores as a small device however wide the screen", () => {
    // A 4-core machine driving a large panel is still a machine that should
    // not be asked for the full move.
    expect(qualityFor(1920, 4).motionScale).toBe(qualityFor(390, 8).motionScale);
  });

  it("never scales motion to nothing, which would read as a broken canvas", () => {
    for (const w of [320, 390, 768, 1024, 1280, 1440, 1920]) {
      expect(qualityFor(w, 8).motionScale).toBeGreaterThan(0);
    }
  });
});
