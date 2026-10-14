/**
 * Cascade fitting for a directional light: split the camera's view depth, then fit one orthographic
 * shadow frustum per slice.
 *
 * "Stable" fitting uses the shared rotation-invariant sphere fit with texel snapping (light-frustum.ts):
 * the map's texel size never changes and the frustum only moves in whole texels, so shadow edges hold
 * still while the camera moves. The "tight" fit is the naive alternative: the light-space bounding box of
 * the slice's eight corners. It wastes fewer texels, but its size and position change every frame, so
 * every edge crawls (shimmering).
 */

import { mat4, vec3, type Mat4 } from 'wgpu-matrix';
import { fitDirectionalShadow, frustumSliceSphere, practicalSplits, type Sphere } from '../../scene/light-frustum';

export interface Cascade {
  viewProj: Mat4;
  /** World size of one shadow-map texel (the larger axis for non-square tight fits). */
  texelWorld: number;
  /** World distance covered by depth 0..1. */
  depthRange: number;
  /** Depth (0..1) where the receivers start; everything nearer the light is caster padding. */
  receiverStart: number;
}

export interface CascadeSetup {
  /** View distances: `count + 1` values from the camera's near plane to the shadow distance. */
  splits: number[];
  /** Width of the blend band at the far end of each cascade (view distance). */
  bands: number[];
  cascades: Cascade[];
}

export interface CascadeOptions {
  /** Camera world matrix (inverse view). */
  cameraWorld: Mat4;
  fovY: number;
  aspect: number;
  near: number;
  shadowDistance: number;
  count: number;
  lambda: number;
  /** Blend band as a fraction of each cascade's depth range. */
  blend: number;
  /** Unit vector toward the light. */
  toLight: ArrayLike<number>;
  mapSize: number;
  stable: boolean;
  /** Everything that can cast: each frustum is extended toward the light until it contains this. */
  casters: Sphere;
}

export function fitCascades(o: CascadeOptions): CascadeSetup {
  const splits = practicalSplits(o.near, o.shadowDistance, o.count, o.lambda);
  const bands = splits.slice(1).map((far, i) => o.blend * (far - splits[i]));
  const toLight = vec3.normalize(vec3.fromValues(o.toLight[0], o.toLight[1], o.toLight[2]));
  const cascades = splits.slice(0, -1).map((start, i) => {
    // Each cascade also covers the previous cascade's blend band, where both are sampled.
    const near = i === 0 ? start : Math.max(o.near, start - bands[i - 1]);
    const far = splits[i + 1];
    return o.stable ? fitStable(o, toLight, near, far) : fitTight(o, toLight, near, far);
  });
  return { splits, bands, cascades };
}

/** How far toward the light (beyond `nearDepth`) a frustum must reach to include every caster. */
function casterPadding(o: CascadeOptions, toLight: Float32Array, nearDepth: number): number {
  // In the light's view (looking down -toLight), distance from the light plane is -dot(p, toLight).
  const castersNear = -vec3.dot(o.casters.center, toLight) - o.casters.radius;
  return Math.max(0, nearDepth - castersNear) + 1;
}

function fitStable(o: CascadeOptions, toLight: Float32Array, near: number, far: number): Cascade {
  const bounds = frustumSliceSphere(o.cameraWorld, o.fovY, o.aspect, near, far);
  const padding = casterPadding(o, toLight, -vec3.dot(bounds.center, toLight) - bounds.radius);
  const fit = fitDirectionalShadow({ toLight, bounds, mapSize: o.mapSize, casterPadding: padding, snap: true });
  return { viewProj: fit.viewProj, texelWorld: fit.texelWorld, depthRange: fit.depthRange, receiverStart: padding / fit.depthRange };
}

function fitTight(o: CascadeOptions, toLight: Float32Array, near: number, far: number): Cascade {
  // Same light view as the stable fit (rotation only), so the two are directly comparable.
  const up = Math.abs(toLight[1]) > 0.99 ? vec3.fromValues(0, 0, 1) : vec3.fromValues(0, 1, 0);
  const view = mat4.lookAt(toLight, vec3.fromValues(0, 0, 0), up);
  const t = Math.tan(o.fovY / 2);
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (const d of [near, far]) {
    for (const [sx, sy] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      const world = vec3.transformMat4(vec3.fromValues(sx * d * t * o.aspect, sy * d * t, -d), o.cameraWorld);
      const p = vec3.transformMat4(world, view);
      for (let c = 0; c < 3; c++) {
        min[c] = Math.min(min[c], p[c]);
        max[c] = Math.max(max[c], p[c]);
      }
    }
  }
  // View space looks down -z: the near (light) side is at max z.
  const nearDepth = -max[2];
  const padding = casterPadding(o, toLight, nearDepth - 1); // the light view's eye sits 1 unit up toLight
  const zNear = nearDepth - padding;
  const zFar = -min[2];
  const proj = mat4.ortho(min[0], max[0], min[1], max[1], zNear, zFar);
  const depthRange = zFar - zNear;
  return {
    viewProj: mat4.multiply(proj, view),
    texelWorld: Math.max(max[0] - min[0], max[1] - min[1]) / o.mapSize,
    depthRange,
    receiverStart: padding / depthRange,
  };
}
