/**
 * Fitting an orthographic shadow frustum for a directional light (E3-S1, E3-S3).
 *
 * The frustum is fitted around a *sphere*, not a box: a sphere's projection does not change as the
 * camera turns, so the shadow map's texel size stays constant and its edges do not swim. The box's
 * position is then snapped to whole shadow-map texels in light space, so translating the camera
 * moves the frustum in texel steps and the rasterised shadow edges stay put (no shimmering).
 * Snapping cannot help when the *light* rotates: every texel then lands somewhere new.
 */

import { mat4, vec3, type Mat4, type Vec3 } from 'wgpu-matrix';

export interface Sphere {
  center: Vec3;
  radius: number;
}

export interface DirectionalShadowFit {
  /** Light view: rotation only, about the world origin, so the snapping grid is fixed in the world. */
  view: Mat4;
  proj: Mat4;
  viewProj: Mat4;
  /** Half the world-space width of the (square) frustum. */
  halfExtent: number;
  /** World-space size of one shadow-map texel. */
  texelWorld: number;
  /** World-space distance covered by depth 0..1. */
  depthRange: number;
}

export interface DirectionalShadowOptions {
  /** Unit vector from the scene toward the light. */
  toLight: ArrayLike<number>;
  /** Region that must be covered by the map. */
  bounds: Sphere;
  /** Shadow-map resolution in texels (square map). */
  mapSize: number;
  /**
   * Extra depth toward the light, so casters outside `bounds` (a tall tree just behind the camera)
   * still land in the map. Defaults to the sphere's diameter.
   */
  casterPadding?: number;
  /** Snap the frustum to whole texels (default true). */
  snap?: boolean;
}

/** Orthographic shadow frustum around `bounds`, as seen from a directional light. */
export function fitDirectionalShadow(o: DirectionalShadowOptions): DirectionalShadowFit {
  const toLight = vec3.normalize(vec3.fromValues(o.toLight[0], o.toLight[1], o.toLight[2]));
  const up = Math.abs(toLight[1]) > 0.99 ? vec3.fromValues(0, 0, 1) : vec3.fromValues(0, 1, 0);
  const view = mat4.lookAt(toLight, vec3.fromValues(0, 0, 0), up);

  const r = o.bounds.radius;
  const halfExtent = r;
  const texelWorld = (2 * halfExtent) / o.mapSize;
  const c = vec3.transformMat4(o.bounds.center, view);
  let cx = c[0];
  let cy = c[1];
  if (o.snap ?? true) {
    cx = Math.round(cx / texelWorld) * texelWorld;
    cy = Math.round(cy / texelWorld) * texelWorld;
  }
  // The light looks down -z in view space; depth runs from the light side of the sphere (plus
  // padding) to the far side.
  const padding = o.casterPadding ?? 2 * r;
  const near = -c[2] - r - padding;
  const far = -c[2] + r;
  const proj = mat4.ortho(cx - halfExtent, cx + halfExtent, cy - halfExtent, cy + halfExtent, near, far);
  return { view, proj, viewProj: mat4.multiply(proj, view), halfExtent, texelWorld, depthRange: far - near };
}

/**
 * The smallest sphere around the slice of a perspective camera's frustum between view distances
 * `near` and `far`. It depends only on the slice and the lens, never on the camera's orientation,
 * which is what keeps the fitted shadow frustum stable.
 *
 * `cameraWorld` is the camera's world matrix (the inverse of its view matrix).
 */
export function frustumSliceSphere(cameraWorld: Mat4, fovY: number, aspect: number, near: number, far: number): Sphere {
  const t = Math.tan(fovY / 2);
  const k2 = t * t * (1 + aspect * aspect);
  let distance: number;
  let radius: number;
  if (k2 >= (far - near) / (far + near)) {
    // A wide lens: the far plane's corners alone decide it.
    distance = far;
    radius = far * Math.sqrt(k2);
  } else {
    distance = 0.5 * (far + near) * (1 + k2);
    radius = 0.5 * Math.sqrt((far - near) ** 2 + 2 * (far * far + near * near) * k2 + (far + near) ** 2 * k2 * k2);
  }
  // View space looks down -z.
  const center = vec3.transformMat4(vec3.fromValues(0, 0, -distance), cameraWorld);
  return { center, radius };
}

/**
 * Cascade split distances by the "practical" scheme (Zhang et al. 2006): a blend of uniform and
 * logarithmic splits. `lambda` 0 is uniform, 1 is logarithmic. Returns `count + 1` distances from
 * `near` to `far`.
 */
export function practicalSplits(near: number, far: number, count: number, lambda: number): number[] {
  const splits = [near];
  for (let i = 1; i < count; i++) {
    const f = i / count;
    const log = near * (far / near) ** f;
    const uniform = near + (far - near) * f;
    splits.push(lambda * log + (1 - lambda) * uniform);
  }
  splits.push(far);
  return splits;
}
