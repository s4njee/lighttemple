/**
 * The studio: a seamless paper sweep and two sets of subjects (a product still life and a
 * roughness × metalness sphere chart), plus the capsule proxies their approximate area-light
 * shadows and contact occlusion are computed from.
 */

import { mat4, type Mat4 } from 'wgpu-matrix';
import { box, sphere, torus, uploadMesh, type MeshData, type Mesh } from '../../gpu/geometry';
import { lathe, cylinder } from '../../scene/shapes';
import type { Prop } from '../../scene/props';

/** Marks the sweep in `Prop.extra` (fades out at its edges, ignores the roughness override). */
export const SWEEP_ID = 1;

/**
 * A photographer's seamless backdrop: a floor running from z = `front` back to z = `back`,
 * bending up through a quarter circle of `radius` into a wall `height` tall; `width` wide along
 * x. Faces point up / forward (+y, +z).
 */
export function sweep(width: number, front: number, back: number, radius: number, height: number): MeshData {
  // Profile (z, y, nz, ny) from the front edge to the top of the wall.
  const profile: number[][] = [];
  const flat = 8;
  for (let i = 0; i <= flat; i++) profile.push([front + (back - front) * (i / flat), 0, 0, 1]);
  const bend = 24;
  for (let i = 1; i <= bend; i++) {
    const a = (i / bend) * (Math.PI / 2);
    profile.push([back - radius * Math.sin(a), radius * (1 - Math.cos(a)), Math.sin(a), Math.cos(a)]);
  }
  for (let i = 1; i <= flat; i++) profile.push([back - radius, radius + (height - radius) * (i / flat), 1, 0]);
  const columns = 16;
  const v: number[] = [];
  const idx: number[] = [];
  let length = 0;
  profile.forEach(([z, y, nz, ny], j) => {
    if (j > 0) length += Math.hypot(z - profile[j - 1][0], y - profile[j - 1][1]);
    for (let i = 0; i <= columns; i++) v.push((i / columns - 0.5) * width, y, z, 0, ny, nz, (i / columns) * width, length);
  });
  const row = columns + 1;
  for (let j = 0; j < profile.length - 1; j++) {
    for (let i = 0; i < columns; i++) {
      const a = j * row + i;
      // Counter-clockwise seen from the front/top.
      idx.push(a, a + 1, a + row, a + 1, a + row + 1, a + row);
    }
  }
  return { vertices: new Float32Array(v), indices: new Uint32Array(idx) };
}

/**
 * A soft box housing: a prism along z whose cross-section is a regular `sides`-gon of circumradius
 * `radius` (first corner at angle `phase`), from the back (z = -1) to the diffuser (z = 0, normal
 * +z). 4 sides at 45° and radius √2 give the [-1, 1]² square; many sides give a disc.
 */
export function prism(sides: number, phase: number, radius: number): MeshData {
  const v: number[] = [];
  const idx: number[] = [];
  const corner = (k: number): [number, number] => {
    const a = phase + (k / sides) * Math.PI * 2;
    return [Math.cos(a) * radius, Math.sin(a) * radius];
  };
  // Caps: a fan around the centre, wound counter-clockwise seen along their normal.
  for (const [z, nz] of [
    [0, 1],
    [-1, -1],
  ]) {
    const base = v.length / 8;
    v.push(0, 0, z, 0, 0, nz, 0.5, 0.5);
    for (let k = 0; k < sides; k++) {
      const [x, y] = corner(k);
      v.push(x, y, z, 0, 0, nz, 0.5 + x / 2, 0.5 + y / 2);
    }
    for (let k = 0; k < sides; k++) {
      const a = base + 1 + k;
      const b = base + 1 + ((k + 1) % sides);
      if (nz > 0) idx.push(base, a, b);
      else idx.push(base, b, a);
    }
  }
  // Sides: one flat quad per edge.
  for (let k = 0; k < sides; k++) {
    const [x0, y0] = corner(k);
    const [x1, y1] = corner(k + 1);
    const nx = y1 - y0;
    const ny = x0 - x1;
    const len = Math.hypot(nx, ny) || 1;
    const base = v.length / 8;
    for (const [x, y, z] of [
      [x0, y0, 0],
      [x1, y1, 0],
      [x1, y1, -1],
      [x0, y0, -1],
    ]) {
      v.push(x, y, z, nx / len, ny / len, 0, 0, 0);
    }
    idx.push(base, base + 2, base + 1, base, base + 3, base + 2);
  }
  return { vertices: new Float32Array(v), indices: new Uint32Array(idx) };
}

/** A capsule (segment a–b, radius r) that stands in for an object in the shadow/AO estimates. */
export interface Proxy {
  a: [number, number, number];
  b: [number, number, number];
  radius: number;
  /** Index of the prop it approximates (it never shadows itself). */
  owner: number;
}

export interface StudioScene {
  props: Prop[];
  proxies: Proxy[];
  /** Where the lights aim and the camera looks. */
  aim: [number, number, number];
  cameraDistance: number;
}

/** Catmull-Rom subdivision of a lathe profile (end points kept), so silhouettes stay round. */
function smooth(points: [number, number][], steps = 6): [number, number][] {
  const out: [number, number][] = [points[0]];
  for (let i = 0; i < points.length - 1; i++) {
    const p0 = points[Math.max(i - 1, 0)];
    const p1 = points[i];
    const p2 = points[i + 1];
    const p3 = points[Math.min(i + 2, points.length - 1)];
    for (let k = 1; k <= steps; k++) {
      const t = k / steps;
      const t2 = t * t;
      const t3 = t2 * t;
      const f = (a: number, b: number, c: number, d: number) =>
        0.5 * (2 * b + (c - a) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (3 * b - a - 3 * c + d) * t3);
      out.push([f(p0[0], p1[0], p2[0], p3[0]), f(p0[1], p1[1], p2[1], p3[1])]);
    }
  }
  return out;
}

export type Meshes = ReturnType<typeof createMeshes>;

export function createMeshes(device: GPUDevice) {
  const meshes = {
    sweep: uploadMesh(device, sweep(28, 10, -2.4, 2.8, 10), 'ltc:sweep'),
    sphere: uploadMesh(device, sphere(1, 96, 64), 'ltc:sphere'),
    cube: uploadMesh(device, box(1, 1, 1), 'ltc:cube'),
    ring: uploadMesh(device, torus(0.34, 0.075, 48, 128), 'ltc:ring'),
    can: uploadMesh(device, cylinder(1, 1, 96), 'ltc:can'),
    // A perfume-style bottle: soft shoulders and a waist, so reflections bend along it.
    bottle: uploadMesh(
      device,
      lathe(
        [
          [0, 0],
          ...smooth([
            [0.3, 0],
            [0.36, 0.035],
          [0.42, 0.16],
          [0.46, 0.4],
          [0.45, 0.66],
          [0.4, 0.86],
          [0.3, 1.02],
          [0.18, 1.13],
            [0.125, 1.22],
            [0.12, 1.36],
          ]),
          [0, 1.36],
        ],
        { segments: 128, creaseAngle: 0.7 },
      ),
      'ltc:bottle',
    ),
    cap: uploadMesh(
      device,
      lathe(
        [
          [0, 1.35],
          [0.17, 1.35],
          [0.175, 1.37],
          [0.175, 1.6],
          [0.16, 1.625],
          [0, 1.625],
        ],
        { segments: 96, creaseAngle: 0.5 },
      ),
      'ltc:cap',
    ),
  };
  const housings = {
    rect: uploadMesh(device, prism(4, Math.PI / 4, Math.SQRT2), 'ltc:housing-rect'),
    disc: uploadMesh(device, prism(64, 0, 1), 'ltc:housing-disc'),
  };
  return {
    ...meshes,
    housings,
    destroy: () => [...Object.values(meshes), ...Object.values(housings)].forEach((m: Mesh) => m.destroy()),
  };
}

const T = (x: number, y: number, z: number) => mat4.translation([x, y, z]);
const S = (m: Mat4, s: number | [number, number, number]) => mat4.scale(m, typeof s === 'number' ? [s, s, s] : s);

/** Prop 0 of every scene; its colour and roughness are live params, set before each update. */
const sweepProp = (m: Meshes): Prop => ({ mesh: m.sweep, color: '#808080', roughness: 0.5, extra: SWEEP_ID, castsShadow: false });

export function productScene(m: Meshes): StudioScene {
  return {
    aim: [0, 0.7, 0],
    cameraDistance: 6.2,
    props: [
      sweepProp(m),
      // Polished chrome bottle with a matte black cap: the hero.
      { mesh: m.bottle, color: '#f4f3f1', roughness: 0.06, metallic: 1, transform: () => T(0, 0, 0) },
      { mesh: m.cap, color: '#141414', roughness: 0.62, transform: () => T(0, 0, 0) },
      // Glossy lacquered plastic.
      { mesh: m.sphere, color: '#b8261b', roughness: 0.16, transform: () => S(T(1.12, 0.42, 0.62), 0.42) },
      // Brushed (rough) aluminium can.
      { mesh: m.can, color: '#c9cbd0', roughness: 0.36, metallic: 1, transform: () => S(T(-1.12, 0, -0.18), [0.3, 0.92, 0.3]) },
      // Matte clay block.
      { mesh: m.cube, color: '#d9d1c4', roughness: 0.92, transform: () => S(mat4.rotateY(T(-0.62, 0.25, 1.18), 0.55), 0.5) },
      // Polished gold ring standing on edge.
      { mesh: m.ring, color: '#ffc77a', roughness: 0.12, metallic: 1, transform: () => mat4.rotateX(mat4.rotateY(T(0.62, 0.415, 1.42), -0.5), Math.PI / 2) },
      // Small glossy black pebble.
      { mesh: m.sphere, color: '#0b0b0c', roughness: 0.08, transform: () => S(T(1.62, 0.17, -0.42), 0.17) },
    ],
    proxies: [
      { a: [0, 0.44, 0], b: [0, 0.7, 0], radius: 0.44, owner: 1 },
      { a: [0, 1.1, 0], b: [0, 1.5, 0], radius: 0.17, owner: 2 },
      { a: [1.12, 0.42, 0.62], b: [1.12, 0.42, 0.62], radius: 0.42, owner: 3 },
      { a: [-1.12, 0.3, -0.18], b: [-1.12, 0.62, -0.18], radius: 0.3, owner: 4 },
      { a: [-0.62, 0.25, 1.18], b: [-0.62, 0.25, 1.18], radius: 0.3, owner: 5 },
      { a: [1.62, 0.17, -0.42], b: [1.62, 0.17, -0.42], radius: 0.17, owner: 7 },
    ],
  };
}

/** Two rows of spheres, roughness 0 → 1 left to right: lacquered plastic in front, polished metal behind. */
export function sphereChart(m: Meshes): StudioScene {
  const props: Prop[] = [sweepProp(m)];
  const proxies: Proxy[] = [];
  const columns = 6;
  const r = 0.3;
  for (let row = 0; row < 2; row++) {
    for (let col = 0; col < columns; col++) {
      const x = (col - (columns - 1) / 2) * 0.74;
      const z = row ? -0.55 : 0.45;
      props.push({
        mesh: m.sphere,
        color: row ? '#e8e3dc' : '#2a55b8',
        roughness: col / (columns - 1),
        metallic: row,
        transform: () => S(T(x, r, z), r),
      });
      proxies.push({ a: [x, r, z], b: [x, r, z], radius: r, owner: props.length - 1 });
    }
  }
  return { aim: [0, 0.4, 0], cameraDistance: 5.6, props, proxies };
}
