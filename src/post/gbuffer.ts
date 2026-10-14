/**
 * A minimal G-buffer for screen-space effects (AO, indirect light): a single-sampled,
 * sampleable depth32float plus an rgba16float view-space normal (alpha is free for the
 * experiment, e.g. roughness), and a packed "geometry" texture at the effect's resolution.
 *
 *   const gbuffer = await GBuffer.create(device);
 *   // geometry pipeline: fragment targets [GBuffer.normalTarget, ...yours], depthStencil GBuffer.depthStencil
 *   const { normal, depth } = gbuffer.attachments(size);   // + your own colour attachments
 *   ... draw the scene, writing the view-space normal to location 0 ...
 *   const geo = gbuffer.resolve(encoder, 0.5, cam);       // { current, previous } at half resolution
 *
 * `resolve()` packs view normal xyz + linear depth into rgba32float (see screen-space.wgsl for
 * the conventions) and keeps last frame's copy, which temporal passes use to detect disocclusion.
 * Half resolution picks one real sample per 2×2 block (checkerboard nearest/farthest).
 *
 * The geometry pass usually renders every mesh once more; a deferred experiment can make it its
 * only scene pass by adding albedo/material targets and shading in a fullscreen pass afterwards.
 */

import type { Mat4 } from 'wgpu-matrix';
import commonWgsl from '../shaders/common.wgsl?raw';
import fullscreenWgsl from '../shaders/fullscreen.wgsl?raw';
import screenSpaceWgsl from '../shaders/screen-space.wgsl?raw';
import { createShader, wgsl } from '../gpu/shader';
import { TargetPool, type Target } from '../gpu/targets';
import { UniformBlock, uniformStruct } from '../gpu/uniforms';
import gbufferWgsl from './gbuffer.wgsl?raw';

export const GBUFFER_DEPTH_FORMAT: GPUTextureFormat = 'depth32float';
export const GBUFFER_NORMAL_FORMAT: GPUTextureFormat = 'rgba16float';
/** View normal xyz + positive linear view depth w (0 = background). */
export const GEOMETRY_FORMAT: GPUTextureFormat = 'rgba32float';

/** What screen-space passes need to rebuild view-space positions from the geometry texture. */
export interface ScreenCamera {
  near: number;
  far: number;
  /** (P[0][0], P[1][1]) of the camera's (unjittered) perspective projection. */
  projScale: [number, number];
  /** TAA jitter in NDC the G-buffer was rendered with; [0, 0] without TAA. */
  jitter: [number, number];
}

/** A ScreenCamera from a projection matrix and a TAA jitter given in pixels (as `TemporalAA.jitterPixels`). */
export function screenCamera(
  projection: Mat4,
  near: number,
  far: number,
  size: { width: number; height: number },
  jitterPixels: readonly [number, number] = [0, 0],
): ScreenCamera {
  return {
    near,
    far,
    projScale: [projection[0], projection[5]],
    jitter: [(2 * jitterPixels[0]) / size.width, (-2 * jitterPixels[1]) / size.height],
  };
}

const RESOLVE_UNIFORMS = { near: 'f32', far: 'f32', factor: 'u32' } as const;

export class GBuffer {
  /** Fragment target state for the normal attachment (location 0 of a geometry pipeline). */
  static readonly normalTarget: GPUColorTargetState = { format: GBUFFER_NORMAL_FORMAT };
  /** Depth state for a geometry pipeline that writes the G-buffer depth. */
  static readonly depthStencil: GPUDepthStencilState = { format: GBUFFER_DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: 'less' };

  private readonly targets: TargetPool;
  private readonly uniforms: UniformBlock<typeof RESOLVE_UNIFORMS>;
  private depthTarget: Target | null = null;
  private normalTarget: Target | null = null;
  private index = 0;
  /** Size/scale of the geometry texture last written; `previous` is only valid if it still matches. */
  private geometryKey = '';
  private previousValid = false;
  private readonly groups = new WeakMap<GPUTexture, { normal: GPUTexture; group: GPUBindGroup }>();

  private constructor(
    private readonly device: GPUDevice,
    private readonly pipeline: GPURenderPipeline,
    private readonly layout: GPUBindGroupLayout,
  ) {
    this.targets = new TargetPool(device);
    this.uniforms = new UniformBlock(device, 'GBufferResolve', RESOLVE_UNIFORMS);
  }

  static async create(device: GPUDevice): Promise<GBuffer> {
    const code = wgsl(commonWgsl, fullscreenWgsl, screenSpaceWgsl, uniformStruct('GBufferResolve', RESOLVE_UNIFORMS), gbufferWgsl);
    const module = await createShader(device, 'gbuffer-resolve', code);
    const layout = device.createBindGroupLayout({
      label: 'gbuffer-resolve',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
      ],
    });
    const pipeline = await device.createRenderPipelineAsync({
      label: 'gbuffer-resolve',
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      vertex: { module, entryPoint: 'vs_fullscreen' },
      fragment: { module, entryPoint: 'fs_resolve', targets: [{ format: GEOMETRY_FORMAT }] },
    });
    return new GBuffer(device, pipeline, layout);
  }

  /** Full-resolution depth (depth32float) of the current frame; valid after `attachments()`. */
  get depth(): GPUTexture {
    if (!this.depthTarget) throw new Error('GBuffer: call attachments() first');
    return this.depthTarget.texture;
  }

  /** Full-resolution view normal (rgba16float) of the current frame; valid after `attachments()`. */
  get normal(): GPUTexture {
    if (!this.normalTarget) throw new Error('GBuffer: call attachments() first');
    return this.normalTarget.texture;
  }

  /**
   * (Re)sizes the full-resolution targets and returns render-pass attachments that clear them:
   * depth to 1 and the normal to 0. Put `normal` first in `colorAttachments`.
   */
  attachments(size: { width: number; height: number }): {
    normal: GPURenderPassColorAttachment;
    depth: GPURenderPassDepthStencilAttachment;
  } {
    const full = [size.width, size.height] as const;
    this.depthTarget = this.targets.get('depth', { size: full, format: GBUFFER_DEPTH_FORMAT });
    this.normalTarget = this.targets.get('normal', { size: full, format: GBUFFER_NORMAL_FORMAT });
    return {
      normal: { view: this.normalTarget.view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] },
      depth: { view: this.depthTarget.view, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
    };
  }

  /**
   * Packs this frame's G-buffer into a geometry texture at `scale` (1 or 0.5) of full resolution.
   * `previous` is last frame's (same size and scale), or null after a resize, a scale change or `reset()`.
   */
  resolve(encoder: GPUCommandEncoder, scale: 1 | 0.5, cam: ScreenCamera): { current: GPUTexture; previous: GPUTexture | null } {
    const depth = this.depth;
    const full = [depth.width, depth.height] as const;
    const key = `${depth.width}x${depth.height}@${scale}`;
    if (key !== this.geometryKey) {
      this.geometryKey = key;
      this.previousValid = false;
    }
    this.index = 1 - this.index;
    const current = this.targets.get(`geometry${this.index}`, { size: full, scale, format: GEOMETRY_FORMAT });
    const previous = this.targets.get(`geometry${1 - this.index}`, { size: full, scale, format: GEOMETRY_FORMAT });

    this.uniforms.set('near', cam.near).set('far', cam.far).set('factor', scale === 1 ? 1 : 2).upload();
    const pass = encoder.beginRenderPass({
      label: 'gbuffer-resolve',
      colorAttachments: [{ view: current.view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 1, 0] }],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup(depth, this.normal));
    pass.draw(3);
    pass.end();

    const result = { current: current.texture, previous: this.previousValid ? previous.texture : null };
    this.previousValid = true;
    return result;
  }

  /** Forget last frame's geometry (a camera cut). */
  reset(): void {
    this.previousValid = false;
  }

  private bindGroup(depth: GPUTexture, normal: GPUTexture): GPUBindGroup {
    const hit = this.groups.get(depth);
    if (hit?.normal === normal) return hit.group;
    const group = this.device.createBindGroup({
      label: 'gbuffer-resolve',
      layout: this.layout,
      entries: [
        { binding: 0, resource: { buffer: this.uniforms.buffer } },
        { binding: 1, resource: depth.createView() },
        { binding: 2, resource: normal.createView() },
      ],
    });
    this.groups.set(depth, { normal, group });
    return group;
  }

  destroy(): void {
    this.targets.destroy();
    this.uniforms.destroy();
  }
}
