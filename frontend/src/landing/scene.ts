/**
 * The world the landing page travels through.
 *
 * One Earth, one satellite, one camera, one timeline. Everything the visitor
 * sees move is a function of a single scroll progress value in [0, 1] — there
 * is no second animator, no per-section listener, and nothing animates on its
 * own clock except a barely-perceptible cloud drift. That constraint is what
 * makes the page feel like one continuous shot rather than seven effects.
 *
 * The Earth is textured with NASA imagery rather than a procedural sphere,
 * because the brief is Earth observation: a visitor should recognise a
 * coastline they know. The day/night terminator is computed in the shader from
 * the same sun direction that lights the satellite, so the lit limb, the city
 * lights and the atmospheric rim all agree with one another.
 */

import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

export interface SceneQuality {
  /** Sphere tessellation. Lower on weak devices. */
  segments: number;
  /** Device pixel ratio ceiling. The single biggest GPU cost. */
  maxPixelRatio: number;
  clouds: boolean;
  stars: number;
}

export function qualityFor(width: number, cores: number): SceneQuality {
  // Deliberately coarse tiers. A long device-capability probe would cost more
  // than it saves; the honest signals are "small screen" and "few cores".
  if (width < 760 || cores <= 4) {
    return { segments: 40, maxPixelRatio: 1.5, clouds: false, stars: 700 };
  }
  if (width < 1200) {
    return { segments: 56, maxPixelRatio: 1.75, clouds: true, stars: 1100 };
  }
  return { segments: 72, maxPixelRatio: 2, clouds: true, stars: 1600 };
}

const EARTH_RADIUS = 1;

/**
 * The Earth surface shader.
 *
 * Day and night are two textures blended across the terminator rather than a
 * lit/unlit switch: the real transition is a soft band a few hundred kilometres
 * wide, and a hard edge is the single thing that makes a CG Earth look like a
 * game asset. City lights are multiplied in only on the night side, and only
 * where the day texture is dark, which keeps them off snowfields and cloud.
 */
const earthVertex = /* glsl */ `
  varying vec2 vUv;
  varying vec3 vNormalW;
  void main() {
    vUv = uv;
    vNormalW = normalize(mat3(modelMatrix) * normal);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const earthFragment = /* glsl */ `
  uniform sampler2D dayMap;
  uniform sampler2D nightMap;
  uniform vec3 sunDirection;
  uniform float nightLift;
  varying vec2 vUv;
  varying vec3 vNormalW;

  void main() {
    vec3 day = texture2D(dayMap, vUv).rgb;
    vec3 night = texture2D(nightMap, vUv).rgb;

    // Blue Marble is a radiometrically faithful product, which means its
    // oceans are nearly black and it reads as underexposed on a screen. A
    // gain plus a gentle lift in the shadows brings the continents up without
    // flattening the land, and a touch of blue in the dark end keeps the
    // oceans oceanic rather than grey.
    day = pow(day, vec3(0.82)) * 1.55;
    day += vec3(0.010, 0.026, 0.055) * (1.0 - smoothstep(0.0, 0.35, day));
    day = min(day, vec3(1.0));

    float lambert = dot(normalize(vNormalW), normalize(sunDirection));
    // A wide, smooth terminator. The offset pushes the band just past the
    // geometric horizon, which is where atmosphere actually puts it.
    float daylight = smoothstep(-0.18, 0.32, lambert);

    // City lights only where the surface is genuinely dark - this stops the
    // night texture's haze from greying out the oceans.
    float lit = smoothstep(0.06, 0.34, dot(night, vec3(0.299, 0.587, 0.114)));
    vec3 cities = night * lit * 1.25;

    vec3 shadow = day * nightLift + cities;
    vec3 color = mix(shadow, day, daylight);

    // A touch of warmth exactly at the terminator: sunrise seen from orbit.
    float rim = smoothstep(0.0, 0.22, daylight) * (1.0 - smoothstep(0.22, 0.55, daylight));
    color += vec3(0.30, 0.16, 0.06) * rim * 0.5;

    gl_FragColor = vec4(color, 1.0);
  }
`;

/**
 * The atmosphere: a slightly larger sphere rendered from the inside.
 *
 * Fresnel against the view vector, gated by the sun so the glow only appears on
 * the lit limb. Additive, low intensity, no bloom pass — the brief's "no neon
 * outline" is a real risk here and the cheapest way to avoid it is to keep the
 * peak value low rather than to post-process it away.
 */
const atmosphereVertex = /* glsl */ `
  varying vec3 vNormalW;
  varying vec3 vViewW;
  void main() {
    vNormalW = normalize(mat3(modelMatrix) * normal);
    vec4 world = modelMatrix * vec4(position, 1.0);
    vViewW = normalize(cameraPosition - world.xyz);
    gl_Position = projectionMatrix * viewMatrix * world;
  }
`;

const atmosphereFragment = /* glsl */ `
  uniform vec3 sunDirection;
  uniform float intensity;
  varying vec3 vNormalW;
  varying vec3 vViewW;

  void main() {
    float fres = 1.0 - max(dot(normalize(vNormalW), normalize(vViewW)), 0.0);
    fres = pow(fres, 5.4);
    float sun = smoothstep(-0.45, 0.35, dot(normalize(vNormalW), normalize(sunDirection)));
    // Desaturated toward daylight white at the bright end: a pure saturated
    // blue ring is what makes a CG Earth look like a logo.
    vec3 tint = mix(vec3(0.13, 0.26, 0.50), vec3(0.62, 0.76, 0.96), sun);
    gl_FragColor = vec4(tint, fres * sun * intensity * 0.62);
  }
`;

export interface WorldHandles {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  earth: THREE.Mesh;
  clouds: THREE.Mesh | null;
  atmosphere: THREE.Mesh;
  satellite: THREE.Group;
  orbit: THREE.Line;
  stars: THREE.Points;
  sunDirection: THREE.Vector3;
}

function makeStars(count: number): THREE.Points {
  const positions = new Float32Array(count * 3);
  const sizes = new Float32Array(count);
  for (let i = 0; i < count; i += 1) {
    // Shell distribution, well outside the Earth so nothing intersects it.
    const r = 26 + Math.random() * 30;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
    positions[i * 3 + 1] = r * Math.cos(phi);
    positions[i * 3 + 2] = r * Math.sin(phi) * Math.sin(theta);
    // A few bright ones among many faint: an even field reads as noise.
    sizes[i] = Math.random() < 0.06 ? 0.10 : 0.045 + Math.random() * 0.03;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("size", new THREE.BufferAttribute(sizes, 1));

  const material = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    uniforms: { opacity: { value: 0.0 } },
    vertexShader: /* glsl */ `
      attribute float size;
      varying float vSize;
      void main() {
        vSize = size;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = size * 620.0 / -mv.z;
        gl_Position = projectionMatrix * mv;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform float opacity;
      varying float vSize;
      void main() {
        // Round, soft-edged points. Square stars are the giveaway of an
        // untouched default point material.
        float d = length(gl_PointCoord - vec2(0.5));
        if (d > 0.5) discard;
        float a = smoothstep(0.5, 0.05, d);
        gl_FragColor = vec4(vec3(1.0), a * opacity);
      }
    `,
  });
  return new THREE.Points(geometry, material);
}

/** The orbit the satellite follows, drawn faintly and revealed on demand. */
function makeOrbit(radius: number, inclination: number): THREE.Line {
  const points: THREE.Vector3[] = [];
  for (let i = 0; i <= 160; i += 1) {
    const t = (i / 160) * Math.PI * 2;
    points.push(new THREE.Vector3(Math.cos(t) * radius, 0, Math.sin(t) * radius));
  }
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({
    color: 0x7fb4e8,
    transparent: true,
    opacity: 0,
  });
  const line = new THREE.Line(geometry, material);
  line.rotation.x = inclination;
  return line;
}

export async function buildWorld(
  quality: SceneQuality,
  baseUrl: string,
): Promise<WorldHandles> {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 200);

  const loader = new THREE.TextureLoader();
  const load = (file: string) =>
    new Promise<THREE.Texture>((resolve, reject) => {
      loader.load(`${baseUrl}${file}`, resolve, undefined, reject);
    });

  const [dayMap, nightMap] = await Promise.all([
    load("earth_day.jpg"),
    load("earth_night.jpg"),
  ]);
  for (const map of [dayMap, nightMap]) {
    map.colorSpace = THREE.SRGBColorSpace;
    map.anisotropy = 4;
  }

  // The sun sits to the left and slightly above, but well toward the camera:
  // the terminator has to fall ACROSS the visible disc, not behind it. A first
  // pass put it too far round and the hero showed a mostly-unlit planet, which
  // read as a rendering failure rather than as night.
  const sunDirection = new THREE.Vector3(-0.46, 0.20, 0.86).normalize();

  const earth = new THREE.Mesh(
    new THREE.SphereGeometry(EARTH_RADIUS, quality.segments, quality.segments / 2),
    new THREE.ShaderMaterial({
      uniforms: {
        dayMap: { value: dayMap },
        nightMap: { value: nightMap },
        sunDirection: { value: sunDirection },
        // Not pure black: the night side of a real planet is lit by moonlight
        // and airglow, and crushing it to zero loses the whole limb.
        nightLift: { value: 0.085 },
      },
      vertexShader: earthVertex,
      fragmentShader: earthFragment,
    }),
  );
  scene.add(earth);

  let clouds: THREE.Mesh | null = null;
  if (quality.clouds) {
    const cloudMap = await load("earth_clouds.jpg");
    cloudMap.colorSpace = THREE.SRGBColorSpace;
    clouds = new THREE.Mesh(
      new THREE.SphereGeometry(EARTH_RADIUS * 1.012, quality.segments, quality.segments / 2),
      new THREE.ShaderMaterial({
        transparent: true,
        depthWrite: false,
        uniforms: {
          cloudMap: { value: cloudMap },
          sunDirection: { value: sunDirection },
          opacity: { value: 0.42 },
        },
        vertexShader: earthVertex,
        fragmentShader: /* glsl */ `
          uniform sampler2D cloudMap;
          uniform vec3 sunDirection;
          uniform float opacity;
          varying vec2 vUv;
          varying vec3 vNormalW;
          void main() {
            // The map is greyscale-ish; its luminance is the cloud mask.
            float c = texture2D(cloudMap, vUv).r;
            c = smoothstep(0.28, 0.95, c);
            float daylight = smoothstep(-0.12, 0.34, dot(normalize(vNormalW), normalize(sunDirection)));
            // Clouds are only visible where there is sun to light them.
            gl_FragColor = vec4(vec3(1.0), c * opacity * daylight);
          }
        `,
      }),
    );
    scene.add(clouds);
  }

  const atmosphere = new THREE.Mesh(
    new THREE.SphereGeometry(EARTH_RADIUS * 1.022, 48, 24),
    new THREE.ShaderMaterial({
      uniforms: { sunDirection: { value: sunDirection }, intensity: { value: 0.9 } },
      vertexShader: atmosphereVertex,
      fragmentShader: atmosphereFragment,
      transparent: true,
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide,
      depthWrite: false,
    }),
  );
  scene.add(atmosphere);

  const stars = makeStars(quality.stars);
  scene.add(stars);

  const orbit = makeOrbit(1.42, THREE.MathUtils.degToRad(24));
  scene.add(orbit);

  // Lighting for the satellite only — the Earth is shaded by its own shader.
  const key = new THREE.DirectionalLight(0xfff4e6, 3.1);
  key.position.copy(sunDirection).multiplyScalar(10);
  scene.add(key);
  // Bounce from the planet below, which is what actually fills a satellite's
  // shadow side in orbit.
  const bounce = new THREE.DirectionalLight(0x4d7ab8, 0.55);
  bounce.position.set(0, -6, 2);
  scene.add(bounce);
  scene.add(new THREE.AmbientLight(0x223044, 0.7));

  const satellite = new THREE.Group();
  scene.add(satellite);
  try {
    const gltf = await new GLTFLoader().loadAsync(`${baseUrl}satellite.glb`);
    gltf.scene.scale.setScalar(0.035);
    satellite.add(gltf.scene);
  } catch {
    // The page must survive a missing asset: the story is carried by the HTML,
    // and an Earth without its satellite is still an Earth.
  }

  return { scene, camera, earth, clouds, atmosphere, satellite, orbit, stars, sunDirection };
}
