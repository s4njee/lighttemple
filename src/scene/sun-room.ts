/**
 * The "sun room" shared by the E4 bounce-light experiments (Irradiance Probes, Reflective Shadow
 * Maps): a closed room whose only opening is a big window in the left wall, so all the sunlight
 * arrives as one patch on a red rug and everything else is lit by bounce light.
 *
 * The room is built from a short list of primitives (yaw-rotated boxes and spheres). The same list
 * feeds the rasteriser (one `PropSet` entry per primitive) and, when `traceable`, a storage buffer
 * of `Prim`s that compute shaders can ray-cast analytically (splice `ROOM_PRIM_WGSL` and
 * `sun-room-trace.wgsl`), so a probe bake sees exactly the geometry the camera does.
 *
 * Walls are slabs around the interior x ∈ [-3, 3], y ∈ [0, 3], z ∈ [-2.5, 2.5]. The camera looks
 * in from outside, "dollhouse" style: `hiddenFor(cameraPos)` lists the walls between the camera and
 * the interior, which camera passes skip. Light passes draw everything, so the room stays closed
 * for the sun and for the bounce light.
 */

import { mat4, vec3, type Mat4, type Vec3 } from 'wgpu-matrix';
import { hexToLinear } from '../gpu/color';
import { box, sphere, uploadMesh, type Mesh } from '../gpu/geometry';
import type { OrbitCameraOptions } from './orbit-camera';
import type { Sphere } from './light-frustum';
import { PropSet, type Prop } from './props';

type V3 = [number, number, number];

/** Interior dimensions (metres) and the window opening in the left wall (x = -w/2). */
export const SUN_ROOM = {
  min: [-3, 0, -2.5] as V3,
  max: [3, 3, 2.5] as V3,
  wall: 0.15,
  /** Window: z0, z1, y0, y1. */
  window: [-1.5, 0.9, 0.55, 2.45] as const,
} as const;

/** Material ids, stored in the prop's `extra` (Object.material.y) and in Prim.extra.x. */
export const ROOM_MAT = { plain: 0, floor: 1, wall: 2, ceiling: 3, rug: 4, wood: 5, fabric: 6, lamp: 7, art: 8, sky: 9 } as const;

export interface RoomPrim {
  kind: 'box' | 'sphere';
  center: V3;
  /** Half extents (box) or [radius, ·, ·] (sphere). */
  half: V3;
  /** Rotation about +y, radians. */
  yaw?: number;
  /** Albedo, sRGB hex. */
  color: string;
  roughness: number;
  metallic?: number;
  material: number;
  /** Glows with the scene's lamp colour (the lamp shade). */
  emissive?: boolean;
  /** Wall slab: its outward normal. Hidden from cameras standing on that side of its inner face. */
  outward?: V3;
  /**
   * The view out of the window (material `sky`): drawn in camera passes only, never in light
   * passes or the traced primitives, so it blocks neither the sun nor the probes' rays.
   */
  backdrop?: boolean;
}

/** Interior wall/floor/ceiling colours that the experiments let you change. */
export interface SunRoomColors {
  rug: string;
  walls: string;
}

export const DEFAULT_ROOM_COLORS: SunRoomColors = { rug: '#b3262e', walls: '#e6dfd3' };

/** Default sun: low enough to throw a long patch through the window onto the rug. */
export const DEFAULT_SUN = { azimuth: -62, elevation: 38 };

export const SUN_ROOM_CAMERA: OrbitCameraOptions = {
  target: [0, 1.0, -0.2],
  distance: 10.5,
  yaw: 0.55,
  pitch: 0.52,
  minPitch: 0.08,
  maxPitch: 1.35,
  minDistance: 4,
  maxDistance: 20,
};

/** A sphere around the whole room including its walls (for fitting the sun's shadow frustum). */
export const SUN_ROOM_BOUNDS: Sphere = { center: vec3.fromValues(0, 1.5, 0), radius: 4.35 };

/** Unit vector toward the sun from azimuth (around +y, 0 = +z) and elevation, in degrees. */
export function sunDirection(azimuthDeg: number, elevationDeg: number): Vec3 {
  const az = (azimuthDeg * Math.PI) / 180;
  const el = (elevationDeg * Math.PI) / 180;
  return vec3.fromValues(Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az));
}

/**
 * Radiance of the outdoors seen through the window (for `room_outdoor` in sun-room.wgsl): the sky
 * colour at the zenith, a paler horizon, and a lawn lit by the sun and the sky. Inputs are linear
 * and already scaled by their intensities; `toSun` is a unit vector.
 */
export function outdoorColors(sun: ArrayLike<number>, sky: ArrayLike<number>, toSun: ArrayLike<number>, skyIntensity = 1) {
  const zenith = [sky[0], sky[1], sky[2]];
  const horizon = zenith.map((c) => c * 0.6 + 0.5 * skyIntensity);
  const lawn = [0.07, 0.11, 0.045];
  const ground = lawn.map((a, i) => (a * (sun[i] * Math.max(0, toSun[1]) + Math.PI * 0.5 * (zenith[i] + horizon[i]))) / Math.PI);
  return { zenith, horizon, ground };
}

/** The room's primitives. Walls first; the colours of rug and walls come from `colors`. */
function buildPrims(colors: SunRoomColors): RoomPrim[] {
  const [x0, y0, z0] = SUN_ROOM.min;
  const [x1, y1, z1] = SUN_ROOM.max;
  const t = SUN_ROOM.wall;
  const ht = t / 2;
  const cx = (x0 + x1) / 2;
  const cz = (z0 + z1) / 2;
  const hx = (x1 - x0) / 2;
  const hy = (y1 - y0) / 2;
  const hz = (z1 - z0) / 2;
  const [wz0, wz1, wy0, wy1] = SUN_ROOM.window;
  const walls = colors.walls;
  const prims: RoomPrim[] = [];
  const wall = (center: V3, half: V3, outward: V3, material: number = ROOM_MAT.wall, color = walls) =>
    prims.push({ kind: 'box', center, half, color, roughness: 0.9, material, outward });

  // Shell: floor and ceiling slabs span the whole footprint, the walls fit between them.
  wall([cx, y0 - ht, cz], [hx + t, ht, hz + t], [0, -1, 0], ROOM_MAT.floor, '#d4b38c');
  wall([cx, y1 + ht, cz], [hx + t, ht, hz + t], [0, 1, 0], ROOM_MAT.ceiling, '#f2efe9');
  wall([cx, hy, z0 - ht], [hx + t, hy, ht], [0, 0, -1]);
  wall([cx, hy, z1 + ht], [hx + t, hy, ht], [0, 0, 1]);
  wall([x1 + ht, hy, cz], [ht, hy, hz], [1, 0, 0]);
  // The left wall is four pieces around the window, plus a slim cross of glazing bars.
  const lx = x0 - ht;
  wall([lx, wy0 / 2, cz], [ht, wy0 / 2, hz], [-1, 0, 0]);
  wall([lx, (wy1 + y1) / 2, cz], [ht, (y1 - wy1) / 2, hz], [-1, 0, 0]);
  wall([lx, (wy0 + wy1) / 2, (z0 + wz0) / 2], [ht, (wy1 - wy0) / 2, (wz0 - z0) / 2], [-1, 0, 0]);
  wall([lx, (wy0 + wy1) / 2, (wz1 + z1) / 2], [ht, (wy1 - wy0) / 2, (z1 - wz1) / 2], [-1, 0, 0]);
  const frame = '#f4f1ea';
  const midZ = (wz0 + wz1) / 2;
  const midY = (wy0 + wy1) / 2;
  wall([lx, midY, midZ], [0.035, (wy1 - wy0) / 2, 0.03], [-1, 0, 0], ROOM_MAT.plain, frame);
  wall([lx, midY, midZ], [0.035, 0.03, (wz1 - wz0) / 2], [-1, 0, 0], ROOM_MAT.plain, frame);
  // What the camera sees through the window: a thin slab just outside it, shaded as the sky.
  prims.push({
    kind: 'box',
    center: [x0 - t - 0.02, midY, midZ],
    half: [0.005, (wy1 - wy0) / 2 + 0.05, (wz1 - wz0) / 2 + 0.05],
    color: '#ffffff',
    roughness: 1,
    material: ROOM_MAT.sky,
    outward: [-1, 0, 0],
    backdrop: true,
  });

  const add = (p: RoomPrim) => prims.push(p);
  // The rug, laid where the sun lands.
  add({ kind: 'box', center: [-1.2, 0.008, -0.9], half: [1.2, 0.008, 0.95], color: colors.rug, roughness: 0.95, material: ROOM_MAT.rug });

  // Coffee table on the rug: its shadow, and the dark space under it, are the classic leak test.
  const wood = '#7a5236';
  const [tx, tz] = [-0.75, -1.45];
  add({ kind: 'box', center: [tx, 0.43, tz], half: [0.55, 0.025, 0.34], color: wood, roughness: 0.5, material: ROOM_MAT.wood });
  for (const [sx, sz] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
    add({ kind: 'box', center: [tx + sx * 0.49, 0.2, tz + sz * 0.28], half: [0.025, 0.2, 0.025], color: wood, roughness: 0.55, material: ROOM_MAT.wood });
  }

  // A pale sofa against the back wall, facing the sun patch.
  const fabric = '#e9e4da';
  const [sx0, sx1] = [0.3, 2.3];
  const scx = (sx0 + sx1) / 2;
  const shx = (sx1 - sx0) / 2;
  add({ kind: 'box', center: [scx, 0.22, z0 + 0.43], half: [shx - 0.14, 0.22, 0.43], color: fabric, roughness: 0.95, material: ROOM_MAT.fabric });
  add({ kind: 'box', center: [scx, 0.62, z0 + 0.13], half: [shx, 0.42, 0.13], color: fabric, roughness: 0.95, material: ROOM_MAT.fabric });
  for (const s of [-1, 1]) {
    add({ kind: 'box', center: [scx + s * (shx - 0.07), 0.32, z0 + 0.43], half: [0.07, 0.32, 0.43], color: fabric, roughness: 0.95, material: ROOM_MAT.fabric });
  }
  // A screen standing out from the back wall between the sun patch and the sofa: bounce light
  // should not reach the sofa side through it (probes with visibility get this right; RSM and
  // plain trilinear probes leak).
  add({ kind: 'box', center: [0.12, 0.8, z0 + 0.6], half: [0.03, 0.8, 0.6], color: '#d9cbb3', roughness: 0.8, material: ROOM_MAT.wood });
  // A sage ottoman, turned a little.
  add({ kind: 'box', center: [1.55, 0.21, 0.45], half: [0.33, 0.21, 0.33], yaw: 0.35, color: '#6f8f6a', roughness: 0.9, material: ROOM_MAT.fabric });

  // A white sphere in the sun on the rug: its shaded underside, toward the camera, glows red.
  add({ kind: 'sphere', center: [-1.8, 0.3, -0.7], half: [0.3, 0, 0], color: '#f0ede6', roughness: 0.6, material: ROOM_MAT.plain });

  // A tall teal cabinet in the back-left corner, beside the window.
  add({ kind: 'box', center: [x0 + 0.45, 1.05, z0 + 0.19], half: [0.45, 1.05, 0.19], color: '#2c5a63', roughness: 0.55, material: ROOM_MAT.wood });

  // A floor lamp right of the sofa; its shade is emissive (an extra light only the probes can see).
  add({ kind: 'box', center: [2.65, 0.015, z0 + 0.4], half: [0.16, 0.015, 0.16], color: '#2b2826', roughness: 0.4, metallic: 1, material: ROOM_MAT.plain });
  add({ kind: 'box', center: [2.65, 0.78, z0 + 0.4], half: [0.015, 0.76, 0.015], color: '#2b2826', roughness: 0.4, metallic: 1, material: ROOM_MAT.plain });
  add({ kind: 'box', center: [2.65, 1.62, z0 + 0.4], half: [0.19, 0.15, 0.19], color: '#efe2c8', roughness: 0.9, material: ROOM_MAT.lamp, emissive: true });

  // A mustard print above the sofa.
  add({ kind: 'box', center: [scx, 1.78, z0 + 0.015], half: [0.55, 0.36, 0.015], color: '#d9a441', roughness: 0.7, material: ROOM_MAT.art });
  return prims;
}

/** Bytes per `Prim`: four vec4f. */
const PRIM_STRIDE = 64;

/**
 * Prim layout for analytic ray casting (`sun-room-trace.wgsl`). Boxes are rotated by `yaw`
 * about their centre; spheres use `half.x` as the radius.
 */
export const ROOM_PRIM_WGSL = /* wgsl */ `
struct Prim {
  // xyz = centre, w = kind (0 box, 1 sphere)
  center: vec4f,
  // xyz = half extents (or radius in x), w = yaw
  half: vec4f,
  // rgb = linear albedo, a = 1 if emissive (glows with the lamp colour)
  color: vec4f,
  // x = material id, y = roughness
  extra: vec4f,
};
`;

export interface SunRoomOptions {
  colors?: SunRoomColors;
  /** Also upload the primitives for compute-shader ray casting (`primBuffer`). */
  traceable?: boolean;
}

export class SunRoom {
  readonly prims: RoomPrim[];
  readonly props: PropSet;
  /**
   * One `Prim` per solid primitive (the window backdrop is left out), in prop order; only when
   * created with `traceable`. Bind it whole: shaders loop to `arrayLength`.
   */
  readonly primBuffer: GPUBuffer | null;
  private readonly meshes: { box: Mesh; sphere: Mesh };
  private colors: SunRoomColors;
  private readonly hidden: boolean[];

  constructor(
    private readonly device: GPUDevice,
    label: string,
    o: SunRoomOptions = {},
  ) {
    this.colors = { ...(o.colors ?? DEFAULT_ROOM_COLORS) };
    // The backdrop goes last, so solid primitives keep the same index in props and prims.
    const all = buildPrims(this.colors);
    this.prims = [...all.filter((p) => !p.backdrop), ...all.filter((p) => p.backdrop)];
    this.hidden = this.prims.map(() => false);
    this.meshes = {
      box: uploadMesh(device, box(2, 2, 2), `${label}:box`),
      sphere: uploadMesh(device, sphere(1, 48, 32), `${label}:sphere`),
    };
    const props: Prop[] = this.prims.map((p) => ({
      mesh: p.kind === 'box' ? this.meshes.box : this.meshes.sphere,
      color: p.color,
      roughness: p.roughness,
      metallic: p.metallic ?? 0,
      extra: p.material,
      transform: () => primMatrix(p),
    }));
    this.props = new PropSet(device, props, `${label}:props`);
    this.props.update(0);
    this.primBuffer = o.traceable
      ? device.createBuffer({ label: `${label}:prims`, size: this.solidCount * PRIM_STRIDE, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
      : null;
    this.uploadPrims();
  }

  /** Primitives that block light (all but the window backdrop). */
  get solidCount(): number {
    return this.prims.filter((p) => !p.backdrop).length;
  }

  /** Recolours the rug and walls. Returns true if anything changed (a probe bake is then stale). */
  setColors(colors: SunRoomColors): boolean {
    if (colors.rug === this.colors.rug && colors.walls === this.colors.walls) return false;
    this.prims.forEach((p, i) => {
      if (p.material === ROOM_MAT.rug) p.color = colors.rug;
      else if (p.material === ROOM_MAT.wall) p.color = colors.walls;
      this.props.props[i].color = p.color;
    });
    this.colors = { ...colors };
    this.props.update(0);
    this.uploadPrims();
    return true;
  }

  /** Marks the walls standing between `cameraPos` and the interior, for `draw({ camera: true })`. */
  updateCutaway(cameraPos: ArrayLike<number>): void {
    this.prims.forEach((p, i) => {
      const o = p.outward;
      if (!o) return;
      // Distance of the camera beyond the wall's inner face, along its outward normal.
      const axis = o[0] !== 0 ? 0 : o[1] !== 0 ? 1 : 2;
      const face = o[axis] > 0 ? SUN_ROOM.max[axis] : SUN_ROOM.min[axis];
      this.hidden[i] = (cameraPos[axis] - face) * o[axis] > 0;
    });
  }

  /** Is primitive `i` currently cut away? */
  isHidden(i: number): boolean {
    return this.hidden[i];
  }

  /**
   * Draws each primitive as instance `i`. Light passes (the default) draw every solid primitive;
   * `camera` passes skip the cut-away walls and add the window backdrop.
   */
  draw(pass: GPURenderPassEncoder, { camera = false } = {}): void {
    this.props.props.forEach((p, i) => {
      if (camera ? this.hidden[i] : this.prims[i].backdrop) return;
      pass.setVertexBuffer(0, p.mesh.vertexBuffer);
      pass.setIndexBuffer(p.mesh.indexBuffer, 'uint32');
      pass.drawIndexed(p.mesh.indexCount, 1, 0, 0, i);
    });
  }

  private uploadPrims(): void {
    if (!this.primBuffer) return;
    const data = new Float32Array((this.solidCount * PRIM_STRIDE) / 4);
    this.prims.forEach((p, i) => {
      if (p.backdrop) return;
      const o = (i * PRIM_STRIDE) / 4;
      data.set([...p.center, p.kind === 'box' ? 0 : 1, ...p.half, p.yaw ?? 0, ...hexToLinear(p.color), p.emissive ? 1 : 0, p.material, p.roughness, 0, 0], o);
    });
    this.device.queue.writeBuffer(this.primBuffer, 0, data);
  }

  destroy(): void {
    this.props.destroy();
    this.primBuffer?.destroy();
    this.meshes.box.destroy();
    this.meshes.sphere.destroy();
  }
}

/** Model matrix of a primitive (the meshes are a 2×2×2 box and a unit sphere). */
function primMatrix(p: RoomPrim): Mat4 {
  const m = mat4.rotateY(mat4.translation(p.center), p.yaw ?? 0);
  return p.kind === 'box' ? mat4.scale(m, p.half) : mat4.uniformScale(m, p.half[0]);
}
