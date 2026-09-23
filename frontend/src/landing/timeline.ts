/**
 * The one place scroll becomes motion.
 *
 * A single progress value in [0, 1] drives the camera, the Earth's rotation,
 * the satellite's position on its orbit and every layer's opacity. Nothing else
 * in the page listens to scroll. That is the whole architecture, and it is what
 * lets the seven chapters read as one continuous camera move instead of seven
 * animations that happen to be adjacent.
 *
 * Scroll is never applied directly to the camera. The raw value is a target;
 * what the renderer reads is a value that eases toward it every frame. Mapping
 * the scrollbar straight onto a transform is what makes scroll-driven 3D feel
 * jittery on a trackpad, because the input arrives in coarse discrete jumps.
 *
 * Native scrolling is untouched: nothing here calls preventDefault, and the
 * page scrolls exactly as the browser intends.
 */

import * as THREE from "three";

/** Chapter boundaries as fractions of total scroll. */
export const CHAPTERS = [
  "hero",
  "problem",
  "pipeline",
  "multimodal",
  "questions",
  "evidence",
  "close",
] as const;

export type ChapterName = (typeof CHAPTERS)[number];

/** A camera pose, in spherical terms around the Earth. */
interface Pose {
  /** Distance from the Earth's centre, in Earth radii. */
  distance: number;
  /** Vertical angle in degrees; positive looks down from above. */
  elevation: number;
  /** Horizontal angle in degrees. */
  azimuth: number;
  /** Where the camera aims, offset from the Earth's centre. */
  targetY: number;
  /**
   * Horizontal aim offset. Looking left of the planet pushes it right in
   * frame, which is how the chapters keep a clear column for their text
   * instead of setting type over a coastline.
   */
  targetX: number;
  /** Earth spin, in degrees, accumulated by this point in the story. */
  spin: number;
  /**
   * Axial tilt in degrees, about X. Applied BEFORE the spin, so the planet
   * turns about a tilted axis the way a planet does, rather than nodding
   * independently of its own rotation.
   */
  tiltX: number;
  /** Roll in degrees, about Z. The smallest of the three, and last applied. */
  rollZ: number;
  /**
   * How much of the frame the satellite is entitled to, in [0, 1].
   *
   * It is a channel of the same storyboard rather than a separate effect,
   * so it is interpolated by the same curve as the camera and can never
   * step. It rises where the chapter is ABOUT the spacecraft and falls
   * where the reader needs the text, which is the whole of its logic.
   */
  satellite: number;
}

/**
 * The storyboard.
 *
 * A note on azimuth: it barely moves. An early version swung the camera a full
 * half-turn, which put the last four chapters on the night side and ended the
 * page on a black planet - the opposite of the calm, lit ending it wants. The
 * journey is carried by `spin` instead, which brings new continents round
 * while the camera stays where the sun is. The camera drifts just enough to
 * feel alive.
 *
 * Read down the `distance` column and the journey is legible on its own: start
 * far enough that the Earth is a body in space, close steadily through the
 * middle chapters as the subject narrows from "the planet" to "one measurement",
 * then pull back at the end. `spin` only ever increases, so the Earth never
 * rewinds — scrolling up reverses the camera, not time.
 */
const KEYFRAMES: { at: number; pose: Pose }[] = [
  // Hero: the whole planet, held right of frame so the headline has a clean
  // column. Far enough that it reads as a body in space, not a texture.
  { at: 0.00, pose: { distance: 4.15, elevation: 8, azimuth: 0, targetY: 0.02, targetX: -0.95, spin: 0, tiltX: 0, rollZ: 0, satellite: 0.30 } },
  // Problem: drifts left and closer; the Earth keeps turning. The spacecraft
  // steps back here - this chapter is four paragraphs of text.
  { at: 0.17, pose: { distance: 4.05, elevation: 13, azimuth: 16, targetY: 0.02, targetX: 0.88, spin: 34, tiltX: -2.4, rollZ: 0.6, satellite: 0.20 } },
  // Pipeline: closer, planet back to the right of the text column. "Satellite
  // data" is a step on this list, so the source of it comes forward.
  { at: 0.35, pose: { distance: 3.75, elevation: 5, azimuth: 34, targetY: 0.0, targetX: -0.92, spin: 78, tiltX: -4.9, rollZ: 1.4, satellite: 0.74 } },
  // Multimodal: lower and nearer. This chapter IS the two sensors, so this is
  // the satellite's moment and the only place it reaches full prominence.
  { at: 0.52, pose: { distance: 3.45, elevation: -5, azimuth: 52, targetY: -0.03, targetX: 0.85, spin: 122, tiltX: -7.6, rollZ: 2.2, satellite: 1.0 } },
  // Questions: the limb fills one side; we are close to the surface now. The
  // product screenshot needs the eye, so the orbit recedes again.
  { at: 0.68, pose: { distance: 3.10, elevation: -11, azimuth: 44, targetY: -0.05, targetX: -0.88, spin: 168, tiltX: -9.6, rollZ: 2.8, satellite: 0.26 } },
  // Evidence: closest approach - over a place, not a planet. The spacecraft
  // returns at half strength as the visible provenance of the measurements.
  { at: 0.84, pose: { distance: 2.75, elevation: -15, azimuth: 28, targetY: -0.04, targetX: 0.80, spin: 212, tiltX: -11.0, rollZ: 3.6, satellite: 0.56 } },
  // Close: pull back and rise. The ending is calm, so the camera retreats.
  { at: 1.00, pose: { distance: 4.45, elevation: 14, azimuth: 8, targetY: 0.02, targetX: 0, spin: 252, tiltX: 0, rollZ: 0, satellite: 0.34 } },
];

type Channel = keyof Pose;

const CHANNELS: Channel[] = [
  "distance", "elevation", "azimuth", "targetY",
  "targetX", "spin", "tiltX", "rollZ", "satellite",
];

const STOPS = KEYFRAMES.map((k) => k.at);

/**
 * Tangents for a monotone cubic (Fritsch-Carlson, 1980).
 *
 * This replaced a per-segment smoothstep, and the reason is the single most
 * visible thing on the page. Easing INSIDE each segment forces the derivative
 * to zero at BOTH ends of every segment, so the camera decelerated to a dead
 * stop at all seven keyframes and accelerated away again: seven little
 * animations, exactly what the storyboard is arranged to avoid. Here the
 * tangent at an interior keyframe is a weighted harmonic mean of the secants
 * either side, so speed carries THROUGH the keyframe and the whole page is one
 * move.
 *
 * Monotone, not Catmull-Rom, because an ordinary spline overshoots: it would
 * push `spin` backwards on the way into a slowing segment (a planet visibly
 * rewinding) and `distance` below its own minimum. The harmonic mean cannot
 * leave the interval bounded by its neighbours, so no channel can ever exceed
 * the range the storyboard above actually states.
 *
 * The tangent is deliberately zeroed in two cases. At a local extremum - where
 * the secants disagree in sign - because a value that reverses direction must
 * pass through zero speed to do it, and anything else is a bounce. And at both
 * ends, which is what gives the hero its settled opening and the last chapter
 * its calm arrival.
 */
function monotoneTangents(xs: number[], ys: number[]): number[] {
  const n = xs.length;
  const secant: number[] = [];
  for (let i = 0; i < n - 1; i += 1) {
    secant.push((ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]));
  }
  const m = new Array<number>(n).fill(0);
  for (let i = 1; i < n - 1; i += 1) {
    const a = secant[i - 1];
    const b = secant[i];
    // Opposite signs (or a flat run) is a turning point: stop to turn.
    if (a * b <= 0) {
      m[i] = 0;
      continue;
    }
    const h1 = xs[i] - xs[i - 1];
    const h2 = xs[i + 1] - xs[i];
    const w1 = 2 * h2 + h1;
    const w2 = h2 + 2 * h1;
    m[i] = (w1 + w2) / (w1 / a + w2 / b);
  }
  return m;
}

/** Per-channel sample points and tangents, computed once at module load. */
const TRACKS: Record<Channel, { ys: number[]; ms: number[] }> = (() => {
  const out = {} as Record<Channel, { ys: number[]; ms: number[] }>;
  for (const c of CHANNELS) {
    const ys = KEYFRAMES.map((k) => k.pose[c]);
    out[c] = { ys, ms: monotoneTangents(STOPS, ys) };
  }
  return out;
})();

/** Cubic Hermite on one segment. */
function hermite(t: number, h: number, y0: number, y1: number, m0: number, m1: number): number {
  const t2 = t * t;
  const t3 = t2 * t;
  return (
    (2 * t3 - 3 * t2 + 1) * y0 +
    (t3 - 2 * t2 + t) * h * m0 +
    (-2 * t3 + 3 * t2) * y1 +
    (t3 - t2) * h * m1
  );
}

function smoothstep(t: number): number {
  const x = Math.min(1, Math.max(0, t));
  return x * x * (3 - 2 * x);
}

function poseAt(progress: number): Pose {
  const p = Math.min(1, Math.max(0, progress));
  let i = 0;
  for (let k = 0; k < STOPS.length - 1; k += 1) {
    if (p >= STOPS[k] && p <= STOPS[k + 1]) {
      i = k;
      break;
    }
    if (p > STOPS[k + 1]) i = k + 1 < STOPS.length - 1 ? k + 1 : k;
  }
  const h = STOPS[i + 1] - STOPS[i];
  const t = h > 0 ? (p - STOPS[i]) / h : 0;
  const out = {} as Pose;
  for (const c of CHANNELS) {
    const { ys, ms } = TRACKS[c];
    out[c] = hermite(t, h, ys[i], ys[i + 1], ms[i], ms[i + 1]);
  }
  return out;
}

/** Progress within one chapter, 0 at its start and 1 at the next chapter. */
export function chapterProgress(progress: number, index: number): number {
  const start = index / CHAPTERS.length;
  const end = (index + 1) / CHAPTERS.length;
  return Math.min(1, Math.max(0, (progress - start) / (end - start)));
}

/** A band that rises and falls, for things visible during one chapter only. */
function band(progress: number, from: number, to: number, fade = 0.06): number {
  const rise = smoothstep((progress - from) / fade);
  const fall = 1 - smoothstep((progress - (to - fade)) / fade);
  return Math.min(rise, fall);
}

export interface Driven {
  camera: THREE.PerspectiveCamera;
  earth: THREE.Mesh;
  clouds: THREE.Mesh | null;
  atmosphere: THREE.Mesh;
  satellite: THREE.Group;
  /**
   * The solar array, if the asset carried one as its own node. Null is a
   * normal state, not a failure: the page must survive a missing or older
   * model, and everything else still runs when it does.
   */
  solarWings: THREE.Object3D | null;
  orbit: THREE.Line;
  stars: THREE.Points;
  /** Where the sun is. The arrays track it, so the timeline needs to know. */
  sunDirection: THREE.Vector3;
}

const TMP_TARGET = new THREE.Vector3();
const TMP_SUN = new THREE.Vector3();
const TMP_MATRIX = new THREE.Matrix4();

/** The distance range the storyboard actually spans, for normalising against. */
const FAR = 4.6;
const NEAR = 2.6;

/** Orbit geometry. Exported so the drawn path and the body cannot disagree. */
export const ORBIT_INCLINATION_DEG = 24;
const ORBIT_MIN = 1.30;
const ORBIT_MAX = 1.64;
/** Model span is 3.76 units; this puts it at roughly 0.18 Earth radii. */
const SATELLITE_SCALE = 0.0475;

/**
 * Place everything for a given progress. Pure: same input, same scene state.
 *
 * `elapsed` is used ONLY for the cloud drift, which is the single thing allowed
 * its own clock — clouds that freeze when the reader stops scrolling look
 * broken, where a planet that stops turning looks deliberate.
 */
export function applyProgress(
  d: Driven,
  progress: number,
  elapsed: number,
  motionScale = 1,
): void {
  const pose = poseAt(progress);

  const el = THREE.MathUtils.degToRad(pose.elevation);
  const az = THREE.MathUtils.degToRad(pose.azimuth);
  d.camera.position.set(
    Math.cos(el) * Math.sin(az) * pose.distance,
    Math.sin(el) * pose.distance,
    Math.cos(el) * Math.cos(az) * pose.distance,
  );
  TMP_TARGET.set(pose.targetX, pose.targetY, 0);
  d.camera.lookAt(TMP_TARGET);

  // How far down the approach we are, from the camera's own distance rather
  // than from raw scroll. Anything keyed to this stays honest when the camera
  // retreats at the end, where a progress-keyed value would keep climbing.
  const closeness = THREE.MathUtils.clamp((FAR - pose.distance) / (FAR - NEAR), 0, 1);

  // The Earth's own rotation, driven entirely by the story.
  //
  // Three axes, in a strict hierarchy: the Y spin is the motion, the X tilt is
  // a fraction of it, and the Z roll is a fraction of that. They come from the
  // SAME keyframes, so they cannot drift out of phase with each other - which
  // is the difference between a planet turning on a tilted axis and three
  // independent animations that happen to run at once.
  //
  // Three.js applies Euler angles in XYZ order, so the tilt is established
  // first and the spin then happens about the TILTED axis. That ordering is
  // the whole reason this reads as planetary rather than as an object being
  // rocked: swap it and the planet wobbles instead of leaning.
  const spin = THREE.MathUtils.degToRad(pose.spin);
  const tilt = THREE.MathUtils.degToRad(pose.tiltX) * motionScale;
  const roll = THREE.MathUtils.degToRad(pose.rollZ) * motionScale;
  d.earth.rotation.set(tilt, spin, roll);
  if (d.clouds) {
    // Clouds lead the surface very slightly, which is the cheapest way to make
    // a two-sphere Earth stop looking like a decal. They take the SAME tilt and
    // roll as the surface: a cloud shell on its own axis would shear visibly
    // against the coastline it is supposed to be sitting above.
    d.clouds.rotation.set(tilt, spin * 1.035 + elapsed * 0.004, roll);
  }

  // Satellite: one lap over the whole page, so it is always somewhere
  // deliberate rather than spinning distractingly. Prominence rides the same
  // storyboard curve as the camera, so it can widen or close its orbit without
  // ever stepping.
  const prominence = THREE.MathUtils.clamp(pose.satellite, 0, 1);
  const orbitAngle = progress * Math.PI * 2.15 + 0.6;
  const orbitRadius = ORBIT_MIN + (ORBIT_MAX - ORBIT_MIN) * prominence;
  const inclination = THREE.MathUtils.degToRad(ORBIT_INCLINATION_DEG);
  const x = Math.cos(orbitAngle) * orbitRadius;
  const z = Math.sin(orbitAngle) * orbitRadius;
  d.satellite.position.set(
    x,
    Math.sin(inclination) * z,
    Math.cos(inclination) * z,
  );
  // Apparent size carries a little of the prominence too, but only a little:
  // the orbit does the work, and a spacecraft that visibly inflates is a
  // cartoon. 0.88x to 1.12x across the whole page.
  d.satellite.scale.setScalar(SATELLITE_SCALE * (0.88 + 0.24 * prominence));
  // Keep its instrument pointed at the Earth: a nadir-pointing satellite is
  // the whole premise, and one that tumbles would contradict the product.
  // lookAt aims the model's -Z, which is the axis the asset's aperture was
  // authored down.
  d.satellite.lookAt(0, 0, 0);

  if (d.solarWings) {
    // A real array turns on its boom to hold the sun. Because the bus is
    // locked nadir-down, that angle changes continuously around the orbit,
    // which gives the spacecraft a slow motion of its own that no amount of
    // added spin could imitate - it is the one rotation that is actually
    // explained by where it is.
    const parent = d.solarWings.parent ?? d.satellite;
    parent.updateWorldMatrix(true, false);
    TMP_MATRIX.copy(parent.matrixWorld).invert();
    TMP_SUN.copy(d.sunDirection).transformDirection(TMP_MATRIX);
    // Panel normal rests along +Z locally; rotating by this about X points it
    // as near the sun as one degree of freedom allows.
    d.solarWings.rotation.x = Math.atan2(-TMP_SUN.y, TMP_SUN.z);
  }

  // The orbit path is a diagram, so it shows exactly when the spacecraft
  // matters and fades with it. One source for both means the line can never be
  // drawn around an orbit the reader has stopped being shown.
  const orbitMat = d.orbit.material as THREE.LineBasicMaterial;
  orbitMat.opacity = 0.40 * prominence * band(progress, 0.06, 0.96, 0.10);
  d.orbit.scale.setScalar(orbitRadius);

  // Stars thin out as the camera closes on the limb, where a real sky is
  // washed out - and come back as it retreats at the end. Keyed to distance,
  // so the ending returns to a field of stars instead of fading to nothing.
  const starMat = d.stars.material as THREE.ShaderMaterial;
  starMat.uniforms.opacity.value =
    (0.30 + 0.30 * (1 - closeness)) * band(progress, 0.015, 0.99, 0.08);
  d.stars.rotation.y = progress * 0.12;

  // The atmosphere thickens along the line of sight as the camera drops toward
  // the limb, which is what a real one does. Distance-keyed for the same
  // reason as the stars: at the end the camera pulls back, and the glow has to
  // relax with it rather than keep growing.
  const atmoMat = d.atmosphere.material as THREE.ShaderMaterial;
  atmoMat.uniforms.intensity.value = 0.72 + closeness * 0.62;
}

/**
 * Eases a value toward a target at a rate independent of frame rate.
 *
 * `1 - exp(-k*dt)` rather than a fixed per-frame fraction: the latter eases
 * twice as fast on a 120 Hz display as on a 60 Hz one, which is how a
 * scroll-driven page ends up feeling different on different machines.
 */
export function damp(current: number, target: number, lambda: number, dt: number): number {
  return current + (target - current) * (1 - Math.exp(-lambda * dt));
}
