/**
 * Screen-space indirect lighting (one diffuse bounce) with a visibility bitmask, at the resolution
 * of a GBuffer geometry texture. See ssil.wgsl for the technique.
 *
 *   const ssil = await SsilPass.create(device);
 *   const raw = ssil.run(encoder, geometry.current, litWithMips, cam, { radius: 4, thickness: 0.5, ... });
 *   // raw: rgba16float, rgb = incoming indirect light (multiply by albedo), a = visibility
 *
 * `radiance` is any lit colour buffer covering the same screen (usually full resolution, this
 * frame's direct light or last frame's reprojected result) with a mip chain: far samples read
 * coarser mips, which removes most of the sparse-sampling noise for free. Like GtaoPass the
 * output is noisy by design; accumulate and denoise it. One `run()` per instance per submission.
 */

import commonWgsl from '../shaders/common.wgsl?raw';
import fullscreenWgsl from '../shaders/fullscreen.wgsl?raw';
import screenSpaceWgsl from '../shaders/screen-space.wgsl?raw';
import { createShader, wgsl } from '../gpu/shader';
import { TargetPool } from '../gpu/targets';
import { UniformBlock, uniformStruct } from '../gpu/uniforms';
import type { ScreenCamera } from './gbuffer';
import ssilWgsl from './ssil.wgsl?raw';

const OUT_FORMAT: GPUTextureFormat = 'rgba16float';

const SSIL_UNIFORMS = {
  projScale: 'vec2f',
  jitter: 'vec2f',
  radius: 'f32',
  maxRadiusPx: 'f32',
  thickness: 'f32',
  slices: 'f32',
  steps: 'f32',
  noiseOffset: 'f32',
  radianceScale: 'f32',
  maxLod: 'f32',
} as const;

export interface SsilSettings {
  /** World-space gather radius. */
  radius: number;
  /** Cap on the radius in pixels of the geometry texture. */
  maxRadiusPx: number;
  /** Assumed depth of every surface seen on screen, in world units. */
  thickness: number;
  slices: number;
  /** Samples per side of each slice. */
  steps: number;
  /** Changes the per-pixel noise every frame (e.g. `(frame % 64) * 5.588`). */
  noiseOffset: number;
}

export class SsilPass {
  private readonly targets: TargetPool;
  private readonly uniforms: UniformBlock<typeof SSIL_UNIFORMS>;
  private readonly sampler: GPUSampler;

  private constructor(
    private readonly device: GPUDevice,
    private readonly pipeline: GPURenderPipeline,
    private readonly layout: GPUBindGroupLayout,
  ) {
    this.targets = new TargetPool(device);
    this.uniforms = new UniformBlock(device, 'Ssil', SSIL_UNIFORMS);
    this.sampler = device.createSampler({
      label: 'ssil-radiance',
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });
  }

  static async create(device: GPUDevice): Promise<SsilPass> {
    const code = wgsl(commonWgsl, fullscreenWgsl, screenSpaceWgsl, uniformStruct('Ssil', SSIL_UNIFORMS), ssilWgsl);
    const module = await createShader(device, 'ssil', code);
    const layout = device.createBindGroupLayout({
      label: 'ssil',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    });
    const pipeline = await device.createRenderPipelineAsync({
      label: 'ssil',
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      vertex: { module, entryPoint: 'vs_fullscreen' },
      fragment: { module, entryPoint: 'fs_ssil', targets: [{ format: OUT_FORMAT }] },
    });
    return new SsilPass(device, pipeline, layout);
  }

  /** Gathers one bounce from `radiance` (filterable, with mips) over `geometry`; returns a target of its size. */
  run(encoder: GPUCommandEncoder, geometry: GPUTexture, radiance: GPUTexture, cam: ScreenCamera, s: SsilSettings): GPUTexture {
    const out = this.targets.get('ssil', { size: [geometry.width, geometry.height], format: OUT_FORMAT });
    this.uniforms
      .set('projScale', cam.projScale)
      .set('jitter', cam.jitter)
      .set('radius', s.radius)
      .set('maxRadiusPx', s.maxRadiusPx)
      .set('thickness', s.thickness)
      .set('slices', Math.max(1, Math.round(s.slices)))
      .set('steps', Math.max(1, Math.round(s.steps)))
      .set('noiseOffset', s.noiseOffset)
      .set('radianceScale', radiance.width / geometry.width)
      .set('maxLod', radiance.mipLevelCount - 1)
      .upload();
    const group = this.device.createBindGroup({
      label: 'ssil',
      layout: this.layout,
      entries: [
        { binding: 0, resource: { buffer: this.uniforms.buffer } },
        { binding: 1, resource: geometry.createView() },
        { binding: 2, resource: radiance.createView() },
        { binding: 3, resource: this.sampler },
      ],
    });
    const pass = encoder.beginRenderPass({
      label: 'ssil',
      colorAttachments: [{ view: out.view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, group);
    pass.draw(3);
    pass.end();
    return out.texture;
  }

  /** Frees the output target (it comes back on the next `run()`). */
  trim(): void {
    this.targets.release('ssil');
  }

  destroy(): void {
    this.targets.destroy();
    this.uniforms.destroy();
  }
}
