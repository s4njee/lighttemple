/**
 * The procedural valley for the cascaded-shadow experiment: a heightfield terrain (fine in the middle,
 * coarse at the rim) plus thousands of instanced props (conifers, broadleaf trees, rocks, standing
 * stones, telegraph poles), all generated on the CPU from deterministic noise.
 *
 * Everything is split into a grid of chunks so each pass (camera or shadow cascade) can skip the chunks
 * it cannot see: the terrain's index buffer and every prop kind's instances are laid out chunk by chunk,
 * so the visible chunks of a pass collapse into a few contiguous draws.
 */

import { mat4, vec3, type Mat4 } from 'wgpu-matrix';
import { hexToLinear } from '../../gpu/color';
import type { MeshData } from '../../gpu/geometry';
import { PROP_STRIDE } from '../../scene/props';

/** Half the side of the square terrain, in metres. */
export const WORLD_HALF = 700;
/** Chunks per side, for culling (terrain and props). */
const CHUNKS = 8;
/** Radius of the camera's fly-through ring, kept clear of trees. */
export const FLY_RADIUS = 78;
/** The standing stones sit on a mound at the origin. */
const CIRCLE_RADIUS = 11;

/** Material ids, stored in each object's `material.y` and switched on in the shader. */
export const MATERIAL = { terrain: 0, foliage: 1, rock: 2, wood: 3 } as const;

// --- Noise -------------------------------------------------------------------

function hash(ix: number, iz: number, seed: number): number {
  let h = Math.imul(ix, 374761393) + Math.imul(iz, 668265263) + Math.imul(seed, 1442695041);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

function valueNoise(x: number, z: number, seed: number): number {
  const ix = Math.floor(x);
  const iz = Math.floor(z);
  const fx = x - ix;
  const fz = z - iz;
  const ux = fx * fx * fx * (fx * (fx * 6 - 15) + 10);
  const uz = fz * fz * fz * (fz * (fz * 6 - 15) + 10);
  const a = hash(ix, iz, seed);
  const b = hash(ix + 1, iz, seed);
  const c = hash(ix, iz + 1, seed);
  const d = hash(ix + 1, iz + 1, seed);
  return a + (b - a) * ux + (c - a) * uz + (a - b - c + d) * ux * uz;
}

/** Fractal value noise in [0, 1]. */
function fbm(x: number, z: number, octaves: number, seed: number): number {
  let sum = 0;
  let amp = 0.5;
  let norm = 0;
  for (let i = 0; i < octaves; i++) {
    sum += amp * valueNoise(x, z, seed + i * 17);
    norm += amp;
    // Rotate a little each octave so the lattice doesn't line up.
    const nx = 1.6 * x + 1.2 * z;
    z = -1.2 * x + 1.6 * z;
    x = nx;
    amp *= 0.5;
  }
  return sum / norm;
}

/** Ridged noise in [0, 1]: sharp crests where value noise crosses its midpoint. */
function ridged(x: number, z: number, octaves: number, seed: number): number {
  let sum = 0;
  let amp = 0.5;
  let norm = 0;
  let weight = 1;
  for (let i = 0; i < octaves; i++) {
    const n = 1 - Math.abs(2 * valueNoise(x, z, seed + i * 31) - 1);
    const r = n * n * weight;
    weight = Math.min(1, r * 1.5);
    sum += amp * r;
    norm += amp;
    const nx = 1.6 * x + 1.2 * z;
    z = -1.2 * x + 1.6 * z;
    x = nx;
    amp *= 0.5;
  }
  return sum / norm;
}

const smoothstep = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

/** Small deterministic PRNG (mulberry32). */
function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// --- Landscape -----------------------------------------------------------------

/** Direction (angle in the xz plane) of the gap in the mountains: toward the default sun. */
const GAP_ANGLE = -2.0;

/** A dirt track winding across the valley: its centre line z(x). */
const pathZ = (x: number) => -50 + 16 * Math.sin(x * 0.011 + 0.6) + 7 * Math.sin(x * 0.029 + 2.0);

/** Approximate distance to the track's centre line. */
export function pathDistance(x: number, z: number): number {
  const slope = (pathZ(x + 0.5) - pathZ(x - 0.5)) / 1;
  return Math.abs(z - pathZ(x)) / Math.sqrt(1 + slope * slope);
}

/** Terrain height (metres) at world (x, z). */
export function terrainHeight(x: number, z: number): number {
  const r = Math.hypot(x, z);
  // Rolling hills, calmer on the valley floor.
  let h = (fbm(x * 0.0045, z * 0.0045, 4, 1) - 0.5) * 44;
  h += (fbm(x * 0.02, z * 0.02, 3, 2) - 0.5) * 5;
  h *= 0.25 + 0.75 * smoothstep(30, 190, r);
  // The mound the stone circle stands on.
  h += 4.5 * Math.exp(-(r * r) / (2 * 20 * 20));
  // Mountains around the rim, lower to the south-west where the valley opens toward the default sun
  // (otherwise a low sun would set behind them and the whole valley would sit in their shadow).
  const gap = Math.abs(Math.atan2(Math.sin(Math.atan2(z, x) - GAP_ANGLE), Math.cos(Math.atan2(z, x) - GAP_ANGLE)));
  const rim = smoothstep(200, 560, r) * (0.2 + 0.8 * smoothstep(0.35, 1.3, gap));
  h += rim * rim * ridged(x * 0.0032 + 3.1, z * 0.0032 - 1.7, 5, 3) * 190;
  // Sink the track a little.
  h -= 0.4 * (1 - smoothstep(0.5, 3.5, pathDistance(x, z)));
  return h;
}

/** 0..1, where the forests grow. */
function forestMask(x: number, z: number): number {
  const r = Math.hypot(x, z);
  const n = fbm(x * 0.008 + 11.3, z * 0.008 - 5.1, 3, 4);
  // Dense woods beyond the valley floor, scattered copses inside it.
  const woods = smoothstep(0.42, 0.55, n) * smoothstep(105, 160, r);
  const copses = smoothstep(0.64, 0.7, n) * smoothstep(32, 45, r);
  return Math.max(woods, copses) * (1 - smoothstep(330, 420, r));
}

// --- Terrain mesh ------------------------------------------------------------------

/** A contiguous range (of indices for the terrain, of instances for props) and its world bounds. */
export interface Chunk {
  first: number;
  count: number;
  min: [number, number, number];
  max: [number, number, number];
}

/** Grid warp: fine cells near the middle where the camera is, coarse ones at the far rim. */
const warp = (u: number) => WORLD_HALF * (0.3 * u + 0.7 * u * u * u);

function buildTerrain(n: number): { mesh: MeshData; chunks: Chunk[] } {
  const row = n + 1;
  const pos = new Float32Array(row * row * 3);
  for (let j = 0; j <= n; j++) {
    const z = warp((j / n) * 2 - 1);
    for (let i = 0; i <= n; i++) {
      const x = warp((i / n) * 2 - 1);
      const o = (j * row + i) * 3;
      pos[o] = x;
      pos[o + 1] = terrainHeight(x, z);
      pos[o + 2] = z;
    }
  }
  // Normals from the mesh itself (not the analytic surface), so shading agrees with the shadows it casts.
  const vertices = new Float32Array(row * row * 8);
  const at = (i: number, j: number) => {
    const o = (Math.min(n, Math.max(0, j)) * row + Math.min(n, Math.max(0, i))) * 3;
    return vec3.fromValues(pos[o], pos[o + 1], pos[o + 2]);
  };
  for (let j = 0; j <= n; j++) {
    for (let i = 0; i <= n; i++) {
      const dx = vec3.subtract(at(i + 1, j), at(i - 1, j));
      const dz = vec3.subtract(at(i, j + 1), at(i, j - 1));
      const nrm = vec3.normalize(vec3.cross(dz, dx));
      const o = j * row + i;
      const x = pos[o * 3];
      const z = pos[o * 3 + 2];
      // uv carries terrain masks for the shader: x = dirt track, y = forest floor.
      const track = 1 - smoothstep(1.3, 2.8, pathDistance(x, z));
      vertices.set([x, pos[o * 3 + 1], z, nrm[0], nrm[1], nrm[2], track, forestMask(x, z)], o * 8);
    }
  }

  // Indices chunk by chunk, so each chunk is one contiguous range.
  const per = n / CHUNKS;
  const indices = new Uint32Array(n * n * 6);
  const chunks: Chunk[] = [];
  let k = 0;
  for (let cz = 0; cz < CHUNKS; cz++) {
    for (let cx = 0; cx < CHUNKS; cx++) {
      const first = k;
      const min: [number, number, number] = [Infinity, Infinity, Infinity];
      const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
      for (let j = cz * per; j < (cz + 1) * per; j++) {
        for (let i = cx * per; i < (cx + 1) * per; i++) {
          const a = j * row + i;
          indices.set([a, a + row, a + 1, a + 1, a + row, a + row + 1], k);
          k += 6;
        }
      }
      for (let j = cz * per; j <= (cz + 1) * per; j++) {
        for (let i = cx * per; i <= (cx + 1) * per; i++) {
          const o = (j * row + i) * 3;
          for (let c = 0; c < 3; c++) {
            min[c] = Math.min(min[c], pos[o + c]);
            max[c] = Math.max(max[c], pos[o + c]);
          }
        }
      }
      chunks.push({ first, count: k - first, min, max });
    }
  }
  return { mesh: { vertices, indices }, chunks };
}

// --- Prop meshes -------------------------------------------------------------------
// Faceted (flat-shaded) low-poly meshes. uv.x flags trunk/bark (1) vs. the instance colour (0); uv.y is
// the vertex's relative height, used as a cheap ambient-occlusion term.

type V3 = [number, number, number];

class MeshBuilder {
  private v: number[] = [];
  /** `base`..`base + height` maps to uv.y 0..1. */
  constructor(
    private readonly height: number,
    private readonly base = 0,
  ) {}

  /** Adds a flat-shaded triangle, flipped if needed so its normal points away from `inside`. */
  tri(a: V3, b: V3, c: V3, inside: V3, flag = 0): void {
    const n = vec3.normalize(vec3.cross(vec3.subtract(b, a), vec3.subtract(c, a)));
    const centroid = vec3.scale(vec3.add(vec3.add(a, b), c), 1 / 3);
    if (vec3.dot(n, vec3.subtract(centroid, inside)) < 0) {
      [b, c] = [c, b];
      vec3.negate(n, n);
    }
    for (const p of [a, b, c]) this.v.push(p[0], p[1], p[2], n[0], n[1], n[2], flag, (p[1] - this.base) / this.height);
  }

  /** A faceted cone (or, with `topRadius` > 0, a prism/frustum) around the y axis. */
  cone(y0: number, radius: number, h: number, segments: number, flag = 0, topRadius = 0, phase = 0): void {
    const inside: V3 = [0, y0 + h * 0.3, 0];
    const ring = (r: number, y: number) =>
      Array.from({ length: segments }, (_, i) => {
        const a = phase + (i / segments) * Math.PI * 2;
        return [Math.cos(a) * r, y, Math.sin(a) * r] as V3;
      });
    const bottom = ring(radius, y0);
    const top = topRadius > 0 ? ring(topRadius, y0 + h) : null;
    const apex: V3 = [0, y0 + h, 0];
    for (let i = 0; i < segments; i++) {
      const j = (i + 1) % segments;
      if (top) {
        this.tri(bottom[i], bottom[j], top[j], inside, flag);
        this.tri(bottom[i], top[j], top[i], inside, flag);
      } else {
        this.tri(bottom[i], bottom[j], apex, inside, flag);
      }
    }
  }

  /** A jittered icosphere (80 faces, or the plain 20-face icosahedron when `coarse`). */
  blob(center: V3, scale: V3, jitter: number, seed: number, coarse = false, flag = 0): void {
    for (const [a, b, c] of coarse ? icosahedron() : icosphere()) {
      const p = [a, b, c].map((q) => {
        // Jitter keyed by position, so shared vertices move together and the mesh stays closed.
        const k = 1 + jitter * (hash(Math.round(q[0] * 1000), Math.round(q[1] * 1000) + Math.round(q[2] * 7919), seed) - 0.5) * 2;
        return [center[0] + q[0] * scale[0] * k, center[1] + q[1] * scale[1] * k, center[2] + q[2] * scale[2] * k] as V3;
      });
      this.tri(p[0], p[1], p[2], center, flag);
    }
  }

  box(center: V3, size: V3, flag = 0): void {
    const h = size.map((s) => s / 2);
    const corner = (sx: number, sy: number, sz: number): V3 => [center[0] + sx * h[0], center[1] + sy * h[1], center[2] + sz * h[2]];
    const faces: [V3, V3, V3, V3][] = [];
    for (const s of [-1, 1]) {
      faces.push([corner(s, -1, -1), corner(s, 1, -1), corner(s, 1, 1), corner(s, -1, 1)]);
      faces.push([corner(-1, s, -1), corner(1, s, -1), corner(1, s, 1), corner(-1, s, 1)]);
      faces.push([corner(-1, -1, s), corner(1, -1, s), corner(1, 1, s), corner(-1, 1, s)]);
    }
    for (const [a, b, c, d] of faces) {
      this.tri(a, b, c, center, flag);
      this.tri(a, c, d, center, flag);
    }
  }

  build(): MeshData {
    const vertices = new Float32Array(this.v);
    return { vertices, indices: Uint32Array.from({ length: vertices.length / 8 }, (_, i) => i) };
  }
}

function icosahedron(): V3[][] {
  const t = (1 + Math.sqrt(5)) / 2;
  const v: V3[] = [
    [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
    [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
    [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
  ].map((p) => {
    const q = vec3.normalize(p);
    return [q[0], q[1], q[2]] as V3;
  });
  const f = [
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
  ];
  return f.map((tri) => tri.map((i) => v[i]));
}

let icoCache: V3[][] | null = null;
function icosphere(): V3[][] {
  if (icoCache) return icoCache;
  const mid = (a: V3, b: V3): V3 => {
    const m = vec3.normalize(vec3.add(a, b));
    return [m[0], m[1], m[2]];
  };
  icoCache = [];
  for (const [a, b, c] of icosahedron()) {
    const ab = mid(a, b);
    const bc = mid(b, c);
    const ca = mid(c, a);
    icoCache.push([a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]);
  }
  return icoCache;
}

/** Unit-height conifer: a short trunk and three stacked cones. */
function conifer(): MeshData {
  const m = new MeshBuilder(1);
  m.cone(0, 0.035, 0.3, 5, 1, 0.025);
  m.cone(0.12, 0.27, 0.46, 8, 0, 0, 0.2);
  m.cone(0.38, 0.2, 0.38, 8, 0, 0, 0.6);
  m.cone(0.6, 0.13, 0.4, 7, 0, 0, 1.1);
  return m.build();
}

/** Unit-height broadleaf tree: trunk and a lumpy crown of two blobs. */
function broadleaf(): MeshData {
  const m = new MeshBuilder(1);
  m.cone(0, 0.045, 0.5, 5, 1, 0.03);
  m.blob([0, 0.6, 0], [0.36, 0.3, 0.36], 0.25, 5);
  m.blob([0.12, 0.8, 0.05], [0.22, 0.2, 0.22], 0.2, 9, true);
  return m.build();
}

/** Unit rock, centred on the origin (sunk into the ground when placed). */
function rock(): MeshData {
  const m = new MeshBuilder(2, -1);
  m.blob([0, 0, 0], [1, 1, 1], 0.35, 13);
  return m.build();
}

/** A 9 m telegraph pole with a cross-arm, in metres. */
function pole(): MeshData {
  const m = new MeshBuilder(9);
  m.box([0, 4.5, 0], [0.24, 9, 0.24], 1);
  m.box([0, 8.3, 0], [2.2, 0.16, 0.16], 1);
  for (const x of [-0.9, -0.3, 0.3, 0.9]) m.box([x, 8.45, 0], [0.06, 0.14, 0.06], 0);
  return m.build();
}

// --- Instances ---------------------------------------------------------------------

export type PropKind = 'conifer' | 'broadleaf' | 'rock' | 'pole';

interface Instance {
  model: Mat4;
  color: [number, number, number];
  roughness: number;
  material: number;
  /** Bounding sphere (world). */
  x: number;
  y: number;
  z: number;
  radius: number;
}

export interface PropGroup {
  kind: PropKind;
  mesh: MeshData;
  /** Instance ranges, one per non-empty chunk, in chunk order. */
  chunks: Chunk[];
  count: number;
}

export interface World {
  terrain: { mesh: MeshData; chunks: Chunk[] };
  groups: PropGroup[];
  /** One `Object` (PROP_WGSL layout) per instance; object 0 is the terrain. */
  objects: Float32Array<ArrayBuffer>;
  /** Total triangles across terrain and every instance. */
  triangles: number;
}

const PALETTE = {
  conifer: ['#2c4426', '#33502b', '#28402c', '#3b5730', '#2f4a33'],
  broadleaf: ['#5d7630', '#6c8236', '#7d8a36', '#94913a', '#b3773a', '#c08f3e'],
  rock: ['#8a857a', '#79746b', '#968f82', '#a29a8b'],
  stone: ['#a49d8f', '#b2aa9a', '#968f83'],
};

export interface WorldOptions {
  /** Terrain grid cells per side (a multiple of 8). */
  grid: number;
  /** Tree density multiplier. */
  density: number;
}

export function buildWorld({ grid, density }: WorldOptions): World {
  const random = rng(1234);
  const pick = <T>(list: T[]) => list[Math.floor(random() * list.length)];
  const instances: Record<PropKind, Instance[]> = { conifer: [], broadleaf: [], rock: [], pole: [] };
  const slope = (x: number, z: number) => {
    const e = 1.5;
    const dx = terrainHeight(x + e, z) - terrainHeight(x - e, z);
    const dz = terrainHeight(x, z + e) - terrainHeight(x, z - e);
    return Math.hypot(dx, dz) / (2 * e);
  };
  const place = (
    kind: PropKind,
    x: number,
    z: number,
    o: { scale: [number, number, number]; yaw?: number; tilt?: number; sink?: number; color: string; roughness: number; material: number; radius: number },
  ) => {
    const y = terrainHeight(x, z) - (o.sink ?? 0);
    let model = mat4.translation([x, y, z]);
    model = mat4.rotateY(model, o.yaw ?? random() * Math.PI * 2);
    if (o.tilt) model = mat4.rotateZ(model, o.tilt);
    model = mat4.scale(model, o.scale);
    const maxScale = Math.max(...o.scale);
    instances[kind].push({
      model,
      color: hexToLinear(o.color),
      roughness: o.roughness,
      material: o.material,
      x,
      y: y + o.scale[1] * 0.5,
      z,
      radius: o.radius * maxScale,
    });
  };

  // Trees: a jittered grid, thinned by the forest mask.
  const cell = 5.5 / Math.sqrt(density);
  const extent = 420;
  for (let gz = -extent; gz < extent; gz += cell) {
    for (let gx = -extent; gx < extent; gx += cell) {
      const x = gx + random() * cell;
      const z = gz + random() * cell;
      const r = Math.hypot(x, z);
      if (r > extent) continue;
      const f = forestMask(x, z);
      const meadow = r > 30 ? 0.006 : 0;
      if (random() > Math.max(f * 0.9, meadow)) continue;
      // Keep clearings: the track, the stone circle and the camera's fly-through ring.
      if (pathDistance(x, z) < 4.5 || r < 24 || Math.abs(r - FLY_RADIUS) < 10) continue;
      if (slope(x, z) > 0.75) continue;
      const h = terrainHeight(x, z);
      const coniferChance = 0.45 + 0.5 * smoothstep(0, 40, h) + 0.3 * smoothstep(120, 260, r);
      if (random() < coniferChance) {
        const s = 8 + random() * 9;
        place('conifer', x, z, {
          scale: [s * (0.85 + random() * 0.3), s, s * (0.85 + random() * 0.3)],
          sink: 0.3,
          color: pick(PALETTE.conifer),
          roughness: 0.85,
          material: MATERIAL.foliage,
          radius: 0.6,
        });
      } else {
        const s = 6 + random() * 5;
        place('broadleaf', x, z, {
          scale: [s * (0.9 + random() * 0.4), s, s * (0.9 + random() * 0.4)],
          sink: 0.3,
          color: pick(PALETTE.broadleaf),
          roughness: 0.8,
          material: MATERIAL.foliage,
          radius: 0.7,
        });
      }
    }
  }

  // Rocks: scattered, in loose clusters, more on the slopes.
  const rockCount = Math.round(500 * density);
  for (let i = 0; i < rockCount; i++) {
    const a = random() * Math.PI * 2;
    const r = 30 + Math.sqrt(random()) * 380;
    const cx = Math.cos(a) * r;
    const cz = Math.sin(a) * r;
    const n = 1 + Math.floor(random() * 3);
    for (let k = 0; k < n; k++) {
      const x = cx + (random() - 0.5) * 8;
      const z = cz + (random() - 0.5) * 8;
      if (pathDistance(x, z) < 3 || Math.abs(Math.hypot(x, z) - FLY_RADIUS) < 6) continue;
      const s = 0.4 + random() ** 3 * 2.6 * (1 + slope(x, z));
      place('rock', x, z, {
        scale: [s * (0.9 + random() * 0.6), s * (0.5 + random() * 0.4), s * (0.9 + random() * 0.6)],
        sink: s * 0.25,
        color: pick(PALETTE.rock),
        roughness: 0.75,
        material: MATERIAL.rock,
        radius: 1.3,
      });
    }
  }

  // Standing stones on the mound, one fallen.
  const stones = 13;
  for (let i = 0; i < stones; i++) {
    const a = (i / stones) * Math.PI * 2 + (random() - 0.5) * 0.12;
    const x = Math.cos(a) * CIRCLE_RADIUS;
    const z = Math.sin(a) * CIRCLE_RADIUS;
    const fallen = i === 4;
    const h = 1.6 + random() * 0.7;
    place('rock', x, z, {
      scale: fallen ? [h, 0.45, 0.6] : [0.75 + random() * 0.2, h, 0.45 + random() * 0.15],
      yaw: -a + (random() - 0.5) * 0.3,
      tilt: fallen ? 0 : (random() - 0.5) * 0.12,
      sink: fallen ? 0.1 : 0.2,
      color: pick(PALETTE.stone),
      roughness: 0.7,
      material: MATERIAL.rock,
      radius: 1.3,
    });
  }

  // Telegraph poles along the track: thin, tall casters that show off shadow resolution.
  for (let x = -330; x <= 330; x += 34) {
    const z = pathZ(x) + 5.5;
    const heading = Math.atan2(pathZ(x + 1) - pathZ(x - 1), 2);
    place('pole', x, z, {
      scale: [1, 1, 1],
      yaw: Math.PI / 2 - heading, // cross-arm across the track
      tilt: (random() - 0.5) * 0.05,
      sink: 0.2,
      color: '#5a4636',
      roughness: 0.8,
      material: MATERIAL.wood,
      radius: 5,
    });
  }

  // Pack: object 0 is the terrain, then every kind's instances sorted by chunk.
  const meshes: Record<PropKind, MeshData> = { conifer: conifer(), broadleaf: broadleaf(), rock: rock(), pole: pole() };
  const total = 1 + Object.values(instances).reduce((n, list) => n + list.length, 0);
  const floats = PROP_STRIDE / 4;
  const objects = new Float32Array(total * floats);
  objects.set(mat4.identity(), 0);
  objects.set([1, 1, 1, 0.9, 0, MATERIAL.terrain, 0, 0], 16);
  const chunkOf = (x: number, z: number) => {
    const c = (v: number) => Math.min(CHUNKS - 1, Math.max(0, Math.floor(((v / WORLD_HALF) * 0.5 + 0.5) * CHUNKS)));
    return c(z) * CHUNKS + c(x);
  };

  const terrain = buildTerrain(grid);
  let next = 1;
  let triangles = terrain.mesh.indices.length / 3;
  const groups: PropGroup[] = [];
  for (const kind of Object.keys(instances) as PropKind[]) {
    const list = instances[kind].map((inst) => ({ inst, chunk: chunkOf(inst.x, inst.z) })).sort((a, b) => a.chunk - b.chunk);
    const chunks: Chunk[] = [];
    let current = -1;
    for (const { inst, chunk } of list) {
      if (chunk !== current) {
        current = chunk;
        chunks.push({ first: next, count: 0, min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] });
      }
      const last = chunks[chunks.length - 1];
      last.count++;
      const p = [inst.x, inst.y, inst.z];
      for (let c = 0; c < 3; c++) {
        last.min[c] = Math.min(last.min[c], p[c] - inst.radius);
        last.max[c] = Math.max(last.max[c], p[c] + inst.radius);
      }
      const o = next * floats;
      objects.set(inst.model, o);
      objects.set([...inst.color, inst.roughness, 0, inst.material, 0, 0], o + 16);
      next++;
    }
    const mesh = meshes[kind];
    triangles += (mesh.indices.length / 3) * list.length;
    groups.push({ kind, mesh, chunks, count: list.length });
  }
  return { terrain, groups, objects, triangles };
}

/** A sphere around everything that can cast a shadow (shadow frusta must reach back to it). */
export function worldBounds(): { center: [number, number, number]; radius: number } {
  return { center: [0, 60, 0], radius: Math.hypot(WORLD_HALF, WORLD_HALF, 160) };
}

/** A point on the camera's fly-through ring and the direction of travel. */
export function flyPoint(angle: number): { x: number; z: number; dirX: number; dirZ: number } {
  return { x: Math.cos(angle) * FLY_RADIUS, z: Math.sin(angle) * FLY_RADIUS, dirX: -Math.sin(angle), dirZ: Math.cos(angle) };
}
