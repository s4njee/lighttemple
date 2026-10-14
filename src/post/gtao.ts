/**
 * Screen-space ambient occlusion pass: GTAO (horizon-based, with bent normals) or classic
 * hemisphere SSAO, at the resolution of a GBuffer geometry texture.
 *
 *   const ao = await GtaoPass.create(device);
 *   const raw = ao.run(encoder, geometry.current, cam, { method: AO_METHOD.GTAO, radius: 1, ... });
 *   // raw: rgba16float, view-space bent normal xyz + visibility w (noisy; accumulate and denoise it)
 *
 * The kernels live in src/shaders/gtao.wgsl. Noise is per pixel and changes with `noiseOffset`,
 * so a temporal pass (TemporalAccumulator) turns a couple of slices per pixel per frame into many.
 * One `run()` per instance per queue submission (the parameters live in one uniform buffer).
 */

import commonWgsl from '../shaders/common.wgsl?raw';
import fullscreenWgsl from '../shaders/fullscreen.wgsl?raw';
import screenSpaceWgsl from '../shaders/screen-space.wgsl?raw';
import gtaoWgsl from '../shaders/gtao.wgsl?raw';
import { createShader, wgsl } from '../gpu/shader';
import { TargetPool } from '../gpu/targets';
import { UniformBlock, uniformStruct } from '../gpu/uniforms';
import type { ScreenCamera } from './gbuffer';
import passWgsl from './gtao-pass.wgsl?raw';

/** Bent normal xyz (view space, unnormalised: average it linearly, normalise to use it) + visibility w. */
export const AO_FORMAT: GPUTextureFormat = 'rgba16float';
export const AO_METHOD = { GTAO: 0, SSAO: 1 } as const;

export interface AoSettings {
  method: (typeof AO_METHOD)[keyof typeof AO_METHOD];
  /** Search radius in world units. */
  radius: number;
  /** GTAO: fraction of the radius over which occluders fade out (0..1). */
  falloff: number;
  /** GTAO: slices per pixel (directions through the pixel). */
  slices: number;
  /** GTAO: steps per side of each slice. SSAO: samples per pixel. */
  steps: number;
  /** SSAO: depth bias in world units. */
  bias?: number;
  /** Cap on the radius in pixels of the geometry texture. */
  maxRadiusPx: number;
  /** Changes the per-pixel noise (e.g. `(frame % 64) * 5.588`); keep it constant for a static pattern. */
  noiseOffset: number;
}

const GTAO_UNIFORMS = {
  projScale: 'vec2f',
  jitter: 'vec2f',
  radius: 'f32',
  falloff: 'f32',
  maxRadiusPx: 'f32',
  slices: 'u32',
  steps: 'u32',
  bias: 'f32',
  noiseOffset: 'f32',
  method: 'u32',
} as const;

export class GtaoPass {
  private readonly targets: TargetPool;
  private readonly uniforms: UniformBlock<typeof GTAO_UNIFORMS>;
  private readonly groups = new WeakMap<GPUTexture, GPUBindGroup>();

  private constructor(
    private readonly device: GPUDevice,
    private readonly pipeline: GPURenderPipeline,
    private readonly layout: GPUBindGroupLayout,
  ) {
    this.targets = new TargetPool(device);
    this.uniforms = new UniformBlock(device, 'Gtao', GTAO_UNIFORMS);
  }

  static async create(device: GPUDevice): Promise<GtaoPass> {
    const code = wgsl(commonWgsl, fullscreenWgsl, screenSpaceWgsl, uniformStruct('Gtao', GTAO_UNIFORMS), passWgsl, gtaoWgsl);
    const module = await createShader(device, 'gtao', code);
    const layout = device.createBindGroupLayout({
      label: 'gtao',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
      ],
    });
    const pipeline = await device.createRenderPipelineAsync({
      label: 'gtao',
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      vertex: { module, entryPoint: 'vs_fullscreen' },
      fragment: { module, entryPoint: 'fs_ao', targets: [{ format: AO_FORMAT }] },
    });
    return new GtaoPass(device, pipeline, layout);
  }

  /** Runs the AO kernel over `geometry` (a GBuffer geometry texture); returns a target of the same size. */
  run(encoder: GPUCommandEncoder, geometry: GPUTexture, cam: ScreenCamera, s: AoSettings): GPUTexture {
    const out = this.targets.get('ao', { size: [geometry.width, geometry.height], format: AO_FORMAT });
    this.uniforms
      .set('projScale', cam.projScale)
      .set('jitter', cam.jitter)
      .set('radius', s.radius)
      .set('falloff', s.falloff)
      .set('maxRadiusPx', s.maxRadiusPx)
      .set('slices', Math.max(1, Math.round(s.slices)))
      .set('steps', Math.max(1, Math.round(s.steps)))
      .set('bias', s.bias ?? 0.02)
      .set('noiseOffset', s.noiseOffset)
      .set('method', s.method)
      .upload();
    let group = this.groups.get(geometry);
    if (!group) {
      group = this.device.createBindGroup({
        label: 'gtao',
        layout: this.layout,
        entries: [
          { binding: 0, resource: { buffer: this.uniforms.buffer } },
          { binding: 1, resource: geometry.createView() },
        ],
      });
      this.groups.set(geometry, group);
    }
    const pass = encoder.beginRenderPass({
      label: 'gtao',
      colorAttachments: [{ view: out.view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 1, 1] }],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, group);
    pass.draw(3);
    pass.end();
    return out.texture;
  }

  /** Frees the output target (it comes back on the next `run()`). */
  trim(): void {
    this.targets.release('ao');
  }

  destroy(): void {
    this.targets.destroy();
    this.uniforms.destroy();
  }
}
