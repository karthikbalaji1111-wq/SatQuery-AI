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
  { at: 0.00, pose: { distance: 4.15, elevation: 8, azimuth: 0, targetY: 0.02, targetX: -0.95, spin: 0 } },
  // Problem: drifts left and closer; the Earth keeps turning.
  { at: 0.17, pose: { distance: 4.05, elevation: 13, azimuth: 16, targetY: 0.02, targetX: 0.88, spin: 34 } },
  // Pipeline: closer, planet back to the right of the text column.
  { at: 0.35, pose: { distance: 3.75, elevation: 5, azimuth: 34, targetY: 0.0, targetX: -0.92, spin: 78 } },
  // Multimodal: lower and nearer, the satellite crosses the frame.
  { at: 0.52, pose: { distance: 3.45, elevation: -5, azimuth: 52, targetY: -0.03, targetX: 0.85, spin: 122 } },
  // Questions: the limb fills one side; we are close to the surface now.
  { at: 0.68, pose: { distance: 3.10, elevation: -11, azimuth: 44, targetY: -0.05, targetX: -0.88, spin: 168 } },
  // Evidence: closest approach - over a place, not a planet.
  { at: 0.84, pose: { distance: 2.75, elevation: -15, azimuth: 28, targetY: -0.04, targetX: 0.80, spin: 212 } },
  // Close: pull back and rise. The ending is calm, so the camera retreats.
  { at: 1.00, pose: { distance: 4.45, elevation: 14, azimuth: 8, targetY: 0.02, targetX: 0, spin: 252 } },
];

function smoothstep(t: number): number {
  const x = Math.min(1, Math.max(0, t));
  return x * x * (3 - 2 * x);
}

function poseAt(progress: number): Pose {
  const p = Math.min(1, Math.max(0, progress));
  let a = KEYFRAMES[0];
  let b = KEYFRAMES[KEYFRAMES.length - 1];
  for (let i = 0; i < KEYFRAMES.length - 1; i += 1) {
    if (p >= KEYFRAMES[i].at && p <= KEYFRAMES[i + 1].at) {
      a = KEYFRAMES[i];
      b = KEYFRAMES[i + 1];
      break;
    }
  }
  const span = b.at - a.at || 1;
  // Eased between keyframes, so the camera arrives and leaves each chapter
  // gently rather than changing velocity at the boundary.
  const t = smoothstep((p - a.at) / span);
  const mix = (from: number, to: number) => from + (to - from) * t;
  return {
    distance: mix(a.pose.distance, b.pose.distance),
    elevation: mix(a.pose.elevation, b.pose.elevation),
    azimuth: mix(a.pose.azimuth, b.pose.azimuth),
    targetY: mix(a.pose.targetY, b.pose.targetY),
    targetX: mix(a.pose.targetX, b.pose.targetX),
    spin: mix(a.pose.spin, b.pose.spin),
  };
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
  orbit: THREE.Line;
  stars: THREE.Points;
}

const TMP_TARGET = new THREE.Vector3();

/**
 * Place everything for a given progress. Pure: same input, same scene state.
 *
 * `elapsed` is used ONLY for the cloud drift, which is the single thing allowed
 * its own clock — clouds that freeze when the reader stops scrolling look
 * broken, where a planet that stops turning looks deliberate.
 */
export function applyProgress(d: Driven, progress: number, elapsed: number): void {
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

  // The Earth's own rotation, driven entirely by the story.
  const spin = THREE.MathUtils.degToRad(pose.spin);
  d.earth.rotation.y = spin;
  if (d.clouds) {
    // Clouds lead the surface very slightly, which is the cheapest way to make
    // a two-sphere Earth stop looking like a decal.
    d.clouds.rotation.y = spin * 1.035 + elapsed * 0.004;
  }

  // Satellite: one lap over the whole page, so it is always somewhere
  // deliberate rather than spinning distractingly.
  const orbitAngle = progress * Math.PI * 2.15 + 0.6;
  const orbitRadius = 1.42;
  const inclination = THREE.MathUtils.degToRad(24);
  const x = Math.cos(orbitAngle) * orbitRadius;
  const z = Math.sin(orbitAngle) * orbitRadius;
  d.satellite.position.set(
    x,
    Math.sin(inclination) * z,
    Math.cos(inclination) * z,
  );
  // Keep its instrument pointed at the Earth: a nadir-pointing satellite is
  // the whole premise, and one that tumbles would contradict the product.
  d.satellite.lookAt(0, 0, 0);

  // The orbit path is a diagram, so it appears only while the story is about
  // where the data comes from, and stays out of the way otherwise.
  const orbitMat = d.orbit.material as THREE.LineBasicMaterial;
  orbitMat.opacity = 0.42 * band(progress, 0.12, 0.60, 0.09);

  // Stars fade in as the camera settles into space and out again as it closes
  // on the surface, where they would be washed out by the limb anyway.
  const starMat = d.stars.material as THREE.ShaderMaterial;
  starMat.uniforms.opacity.value = 0.55 * band(progress, 0.02, 0.92, 0.12);
  d.stars.rotation.y = progress * 0.12;

  // The atmosphere strengthens as the camera drops toward the limb, which is
  // exactly when a real one thickens along the line of sight.
  const atmoMat = d.atmosphere.material as THREE.ShaderMaterial;
  atmoMat.uniforms.intensity.value = 0.75 + smoothstep(progress * 1.15) * 0.55;
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
