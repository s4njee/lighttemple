/**
 * More procedural meshes for the 3D scenes, in the shared layout of `gpu/geometry.ts`
 * (position, normal, uv; counter-clockwise front faces).
 *
 *   lathe(profile)        surface of revolution: candles, vases, lamp caps, pots, plinths
 *   roomShell(w, h, d)    an inward-facing, subdivided box: a room seen from inside (or, with
 *                         back-face culling, a "dollhouse" cut-away seen from outside)
 *
 * Both are subdivided on purpose: non-linear shadow projections (dual-paraboloid maps) bend
 * straight edges, so they need vertices to bend.
 */

import type { MeshData } from '../gpu/geometry';

type V3 = [number, number, number];

const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/** Pushes triangle (a, b, c), flipped if needed so it is counter-clockwise around `n`. */
function pushTriangle(idx: number[], v: number[], a: number, b: number, c: number, n: V3): void {
  const p = (i: number): V3 => [v[i * 8], v[i * 8 + 1], v[i * 8 + 2]];
  const area = cross(sub(p(b), p(a)), sub(p(c), p(a)));
  const facing = dot(area, n);
  if (Math.abs(facing) < 1e-12) return; // degenerate (a pole of the lathe)
  if (facing > 0) idx.push(a, b, c);
  else idx.push(a, c, b);
}

export interface LatheOptions {
  /** Radial segments (default 48). */
  segments?: number;
  /**
   * Profile corners sharper than this (radians between neighbouring segments) get split
   * normals; smoother joints are averaged. Default 0.6 (~35°).
   */
  creaseAngle?: number;
}

/**
 * Revolves a 2D profile of `[radius, y]` points around the y axis. List the points bottom to
 * top; start or end at radius 0 to close the shape with a cap. Normals point away from the axis
 * on walls, down on bottom caps and up on top caps (the outward side of a profile drawn
 * bottom to top).
 */
export function lathe(profile: [number, number][], o: LatheOptions = {}): MeshData {
  const segments = o.segments ?? 48;
  const crease = o.creaseAngle ?? 0.6;
  const v: number[] = [];
  const idx: number[] = [];
  // Outward 2D normal (radial, y) of each profile segment: the tangent rotated clockwise.
  const segNormals = profile.slice(0, -1).map(([r0, y0], i) => {
    const [r1, y1] = profile[i + 1];
    const dr = r1 - r0;
    const dy = y1 - y0;
    const len = Math.hypot(dr, dy) || 1;
    return [dy / len, -dr / len] as [number, number];
  });
  const endNormal = (seg: number, end: 0 | 1): [number, number] => {
    const own = segNormals[seg];
    const other = segNormals[seg + (end ? 1 : -1)];
    if (!other) return own;
    const angle = Math.acos(Math.max(-1, Math.min(1, own[0] * other[0] + own[1] * other[1])));
    if (angle > crease) return own;
    const s = [own[0] + other[0], own[1] + other[1]];
    const len = Math.hypot(s[0], s[1]) || 1;
    return [s[0] / len, s[1] / len];
  };
  // Total profile length, for the v texture coordinate.
  const lengths = [0];
  segNormals.forEach((_, i) => {
    const [r0, y0] = profile[i];
    const [r1, y1] = profile[i + 1];
    lengths.push(lengths[i] + Math.hypot(r1 - r0, y1 - y0));
  });
  const total = lengths[lengths.length - 1] || 1;

  // Each profile segment gets its own pair of rings, so creases stay sharp.
  segNormals.forEach((faceN, s) => {
    const base = v.length / 8;
    for (const end of [0, 1] as const) {
      const [r, y] = profile[s + end];
      const [nr, ny] = endNormal(s, end);
      for (let j = 0; j <= segments; j++) {
        const a = (j / segments) * Math.PI * 2;
        const c = Math.cos(a);
        const sn = Math.sin(a);
        v.push(r * c, y, r * sn, nr * c, ny, nr * sn, j / segments, lengths[s + end] / total);
      }
    }
    const row = segments + 1;
    for (let j = 0; j < segments; j++) {
      const a = (j + 0.5) / segments * Math.PI * 2;
      const n: V3 = [faceN[0] * Math.cos(a), faceN[1], faceN[0] * Math.sin(a)];
      const i0 = base + j;
      const i1 = base + j + 1;
      const i2 = base + row + j;
      const i3 = base + row + j + 1;
      pushTriangle(idx, v, i0, i1, i3, n);
      pushTriangle(idx, v, i0, i3, i2, n);
    }
  });
  return { vertices: new Float32Array(v), indices: new Uint32Array(idx) };
}

/** A closed cylinder (base at y = 0) as a lathe. */
export function cylinder(radius: number, height: number, segments = 32): MeshData {
  return lathe(
    [
      [0, 0],
      [radius, 0],
      [radius, height],
      [0, height],
    ],
    { segments },
  );
}

/**
 * An inward-facing box of size w × h × d with its floor at y = 0, centred on x and z, each face
 * split into a grid of roughly `cell`-sized quads. uv is in world units along each face.
 */
export function roomShell(w: number, h: number, d: number, cell = 0.5): MeshData {
  const v: number[] = [];
  const idx: number[] = [];
  // Inward normal, centre, and the two half-extent axes of each face.
  const faces: [V3, V3, V3, V3][] = [
    [[0, 1, 0], [0, 0, 0], [w / 2, 0, 0], [0, 0, d / 2]], // floor
    [[0, -1, 0], [0, h, 0], [w / 2, 0, 0], [0, 0, d / 2]], // ceiling
    [[1, 0, 0], [-w / 2, h / 2, 0], [0, 0, d / 2], [0, h / 2, 0]], // left wall
    [[-1, 0, 0], [w / 2, h / 2, 0], [0, 0, d / 2], [0, h / 2, 0]], // right wall
    [[0, 0, 1], [0, h / 2, -d / 2], [w / 2, 0, 0], [0, h / 2, 0]], // back wall
    [[0, 0, -1], [0, h / 2, d / 2], [w / 2, 0, 0], [0, h / 2, 0]], // front wall
  ];
  for (const [n, c, ax, ay] of faces) {
    const lx = 2 * Math.hypot(...ax);
    const ly = 2 * Math.hypot(...ay);
    const nx = Math.max(1, Math.round(lx / cell));
    const ny = Math.max(1, Math.round(ly / cell));
    const base = v.length / 8;
    for (let j = 0; j <= ny; j++) {
      for (let i = 0; i <= nx; i++) {
        const s = (i / nx) * 2 - 1;
        const t = (j / ny) * 2 - 1;
        const p = [0, 1, 2].map((k) => c[k] + s * ax[k] + t * ay[k]);
        v.push(...p, ...n, ((s + 1) / 2) * lx, ((t + 1) / 2) * ly);
      }
    }
    const row = nx + 1;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const a = base + j * row + i;
        pushTriangle(idx, v, a, a + 1, a + row + 1, n);
        pushTriangle(idx, v, a, a + row + 1, a + row, n);
      }
    }
  }
  return { vertices: new Float32Array(v), indices: new Uint32Array(idx) };
}

/** An outward-facing box like `geometry.box`, but each face split into `n × n` quads. */
export function subdividedBox(sx: number, sy: number, sz: number, n = 4): MeshData {
  const v: number[] = [];
  const idx: number[] = [];
  const h: V3 = [sx / 2, sy / 2, sz / 2];
  const faces: [V3, V3, V3][] = [
    [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
    [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
    [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
    [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
    [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
    [[0, 0, -1], [1, 0, 0], [0, 1, 0]],
  ];
  for (const [nrm, t, b] of faces) {
    const base = v.length / 8;
    for (let j = 0; j <= n; j++) {
      for (let i = 0; i <= n; i++) {
        const s = (i / n) * 2 - 1;
        const r = (j / n) * 2 - 1;
        const p = [0, 1, 2].map((k) => (nrm[k] + s * t[k] + r * b[k]) * h[k]);
        v.push(...p, ...nrm, i / n, j / n);
      }
    }
    const row = n + 1;
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        const a = base + j * row + i;
        pushTriangle(idx, v, a, a + 1, a + row + 1, nrm);
        pushTriangle(idx, v, a, a + row + 1, a + row, nrm);
      }
    }
  }
  return { vertices: new Float32Array(v), indices: new Uint32Array(idx) };
}
