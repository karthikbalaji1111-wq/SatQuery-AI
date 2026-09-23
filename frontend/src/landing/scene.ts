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

import { ORBIT_INCLINATION_DEG } from "./timeline";

export interface SceneQuality {
  /** Sphere tessellation. Lower on weak devices. */
  segments: number;
  /** Device pixel ratio ceiling. The single biggest GPU cost. */
  maxPixelRatio: number;
  clouds: boolean;
  stars: number;
  /**
   * How much of the choreography's secondary motion this device performs.
   *
   * 1 is the full desktop move. Lower values shrink the tilt and roll ONLY -
   * the story (spin, distance, framing) is identical on every device, because
   * a phone visitor should get the same narrative, just carried with less
   * movement. Scaling the whole pose instead would put the small screen on a
   * different journey from the large one.
   */
  motionScale: number;
}

export function qualityFor(width: number, cores: number): SceneQuality {
  // Deliberately coarse tiers. A long device-capability probe would cost more
  // than it saves; the honest signals are "small screen" and "few cores".
  if (width < 760 || cores <= 4) {
    return { segments: 40, maxPixelRatio: 1.5, clouds: false, stars: 700, motionScale: 0.35 };
  }
  if (width < 1200) {
    return { segments: 56, maxPixelRatio: 1.75, clouds: true, stars: 1100, motionScale: 0.6 };
  }
  return { segments: 72, maxPixelRatio: 2, clouds: true, stars: 1600, motionScale: 1 };
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
  varying vec3 vViewW;
  void main() {
    vUv = uv;
    vNormalW = normalize(mat3(modelMatrix) * normal);
    vec4 world = modelMatrix * vec4(position, 1.0);
    vViewW = normalize(cameraPosition - world.xyz);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const earthFragment = /* glsl */ `
  uniform sampler2D dayMap;
  uniform sampler2D nightMap;
  uniform vec3 sunDirection;
  uniform float nightLift;
  uniform float glint;
  varying vec2 vUv;
  varying vec3 vNormalW;
  varying vec3 vViewW;

  void main() {
    vec3 raw = texture2D(dayMap, vUv).rgb;
    vec3 day = raw;
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

    // Sun glint off water.
    //
    // The ocean mask is blue dominance over red, taken from the UNGAINED
    // sample: Blue Marble's water is strongly blue-biased where soil, snow and
    // cloud are not, so no separate specular map has to be shipped for it.
    // This was checked against a Blender render of the same texture before it
    // was written here - land stayed matte and the highlight landed on water.
    //
    // The highlight is divided down rather than clamped, so it rolls off
    // toward white instead of clipping into a hard disc, which is the usual
    // way a CG ocean ends up with a headlight on it.
    float ocean = clamp((raw.b - raw.r) * 5.0, 0.0, 1.0);
    vec3 halfway = normalize(normalize(sunDirection) + normalize(vViewW));
    float spec = pow(max(dot(normalize(vNormalW), halfway), 0.0), 96.0);
    spec = spec / (1.0 + spec * 0.85);
    color += vec3(0.82, 0.89, 1.0) * spec * ocean * daylight * glint;

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
  /** The satellite's solar array, when the asset ships one as its own node. */
  solarWings: THREE.Object3D | null;
  orbit: THREE.Line;
  stars: THREE.Points;
  sunDirection: THREE.Vector3;
  /**
   * Every texture this world owns.
   *
   * Disposing a material does NOT dispose the textures its uniforms hold, and
   * this world is rebuilt whenever the reduced-motion preference changes - so
   * without an explicit ledger each toggle stranded three NASA maps on the GPU.
   */
  textures: THREE.Texture[];
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

/**
 * The orbit the satellite follows, drawn faintly and revealed on demand.
 *
 * Built at unit radius and scaled by the timeline, so the drawn path and the
 * body on it read from one number and cannot drift apart when the orbit opens
 * out in the chapters that are about the spacecraft.
 */
function makeOrbit(inclination: number): THREE.Line {
  const points: THREE.Vector3[] = [];
  for (let i = 0; i <= 160; i += 1) {
    const t = (i / 160) * Math.PI * 2;
    points.push(new THREE.Vector3(Math.cos(t), 0, Math.sin(t)));
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

  // All three at once. The cloud map used to load only after the other two had
  // resolved, which put a whole extra round trip in front of first paint for no
  // reason - they are independent requests.
  const [dayMap, nightMap, cloudMap] = await Promise.all([
    load("earth_day.jpg"),
    load("earth_night.jpg"),
    quality.clouds ? load("earth_clouds.jpg") : Promise.resolve(null),
  ]);
  const textures: THREE.Texture[] = [dayMap, nightMap];
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
        // Deliberately low. A glint is a hint that the surface is wet, not a
        // light source; past about 0.8 it reads as a lens flare.
        glint: { value: 0.55 },
      },
      vertexShader: earthVertex,
      fragmentShader: earthFragment,
    }),
  );
  scene.add(earth);

  let clouds: THREE.Mesh | null = null;
  if (cloudMap) {
    textures.push(cloudMap);
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

  const orbit = makeOrbit(THREE.MathUtils.degToRad(ORBIT_INCLINATION_DEG));
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
  let solarWings: THREE.Object3D | null = null;
  try {
    // Authored in Blender for this page. The aperture is modelled down -Z,
    // which is the axis three's lookAt() aims, so pointing the instrument at
    // the Earth needs no correction transform here. The previous asset was
    // built down -Y and had been flying past the planet sideways.
    const gltf = await new GLTFLoader().loadAsync(`${baseUrl}satellite-eo.glb`);
    satellite.add(gltf.scene);
    solarWings = gltf.scene.getObjectByName("SolarWings") ?? null;
    gltf.scene.traverse((object) => {
      const mesh = object as THREE.Mesh;
      const material = mesh.material as THREE.MeshStandardMaterial | undefined;
      if (material && "envMapIntensity" in material) {
        // The bus is mostly metal, and metal with no environment to reflect is
        // black. This is what makes the environment below load-bearing rather
        // than decorative.
        material.envMapIntensity = 1.0;
      }
    });
  } catch {
    // The page must survive a missing asset: the story is carried by the HTML,
    // and an Earth without its satellite is still an Earth.
  }

  return {
    scene, camera, earth, clouds, atmosphere,
    satellite, solarWings, orbit, stars, sunDirection, textures,
  };
}


/**
 * A two-kilobyte sky for the spacecraft to reflect.
 *
 * Everything on the satellite is a metal, and a metal lit only by punctual
 * lights has nothing to return to the camera but a few specular pinpoints — it
 * renders as a black cut-out. Physically the fix is an environment, and in low
 * Earth orbit that environment is simple enough to write down: a bright planet
 * filling the hemisphere below, near-black sky above, and the sun.
 *
 * So it is generated rather than downloaded. A 64x32 equirectangular map costs
 * one PMREM pass at startup and no network request at all, which is the right
 * trade when the alternative is shipping an HDR to light an object that is
 * never more than a few hundred pixels across.
 */
function makeEnvironmentSource(sun: THREE.Vector3): THREE.DataTexture {
  const w = 64;
  const h = 32;
  const data = new Uint8Array(w * h * 4);
  const dir = new THREE.Vector3();
  for (let y = 0; y < h; y += 1) {
    const phi = ((y + 0.5) / h) * Math.PI;
    for (let x = 0; x < w; x += 1) {
      const theta = ((x + 0.5) / w) * Math.PI * 2;
      dir.set(
        Math.sin(phi) * Math.cos(theta),
        Math.cos(phi),
        Math.sin(phi) * Math.sin(theta),
      );
      // Earthshine from below, falling off toward the horizon.
      const below = Math.max(0, -dir.y) ** 0.7;
      // The sun as a small, very bright cap rather than a point, so the
      // roughness blur has something with area to work from.
      const solar = Math.max(0, dir.dot(sun)) ** 150;
      const r = 6 + below * 40 + solar * 250;
      const g = 9 + below * 76 + solar * 240;
      const b = 16 + below * 122 + solar * 220;
      const i = (y * w + x) * 4;
      data[i] = Math.min(255, r);
      data[i + 1] = Math.min(255, g);
      data[i + 2] = Math.min(255, b);
      data[i + 3] = 255;
    }
  }
  const texture = new THREE.DataTexture(data, w, h, THREE.RGBAFormat);
  texture.mapping = THREE.EquirectangularReflectionMapping;
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.needsUpdate = true;
  return texture;
}

/**
 * Installs that environment. Needs the renderer, so it runs after the world is
 * built rather than inside it.
 *
 * Only the satellite's standard materials consume it — the Earth, atmosphere,
 * clouds and stars are all raw ShaderMaterials and are untouched by
 * `scene.environment`, so this cannot disturb the tuned planet.
 */
export function applyEnvironment(
  renderer: THREE.WebGLRenderer,
  world: WorldHandles,
): THREE.WebGLRenderTarget {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const source = makeEnvironmentSource(world.sunDirection);
  const target = pmrem.fromEquirectangular(source);
  world.scene.environment = target.texture;
  source.dispose();
  pmrem.dispose();
  return target;
}
