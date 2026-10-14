/**
 * Edge-aware (bilateral) filters for screen-space signals that come with a GBuffer geometry
 * texture (view normal + linear depth per texel):
 *
 *   const bilateral = await BilateralFilter.create(device);
 *   const clean = bilateral.denoise(encoder, signal, geo.current, { radius: 2 });
 *   const full = bilateral.upsample(encoder, clean, geo.current, gbuffer, cam);   // half -> full res
 *
 * `denoise` is a (2r+1)² Gaussian whose taps are weighted by depth and normal similarity, so it
 * smooths within surfaces and stops at silhouettes and creases. `upsample` takes a half-resolution
 * signal to the G-buffer's full resolution with depth/normal-aware bilinear weights (the classic
 * "bilateral upsample"), which keeps half-res AO or GI from bleeding a halo across depth edges.
 *
 * Both write rgba16float targets owned by the filter. One call of each per instance per frame.
 */

import commonWgsl from '../shaders/common.wgsl?raw';
import fullscreenWgsl from '../shaders/fullscreen.wgsl?raw';
import screenSpaceWgsl from '../shaders/screen-space.wgsl?raw';
import { createShader, wgsl } from '../gpu/shader';
import { TargetPool } from '../gpu/targets';
import { UniformBlock, uniformStruct } from '../gpu/uniforms';
import type { ScreenCamera } from './gbuffer';
import bilateralWgsl from './bilateral.wgsl?raw';

const OUT_FORMAT: GPUTextureFormat = 'rgba16float';
/** Largest denoise radius, in texels. */
export const MAX_DENOISE_RADIUS = 4;

const BILATERAL_UNIFORMS = {
  background: 'vec4f',
  radius: 'f32',
  spatialSigma: 'f32',
  depthSigma: 'f32',
  normalPower: 'f32',
  near: 'f32',
  far: 'f32',
} as const;

export interface BilateralOptions {
  /** Relative depth difference (fraction of view depth) at which a tap's weight drops to e⁻¹. */
  depthSigma?: number;
  /** Exponent on the normals' dot product: higher = stricter about creases. */
  normalPower?: number;
}

export class BilateralFilter {
  private readonly targets: TargetPool;
  private readonly denoiseUniforms: UniformBlock<typeof BILATERAL_UNIFORMS>;
  private readonly upsampleUniforms: UniformBlock<typeof BILATERAL_UNIFORMS>;

  private constructor(
    private readonly device: GPUDevice,
    private readonly pipelines: { denoise: GPURenderPipeline; upsample: GPURenderPipeline },
    private readonly layouts: { denoise: GPUBindGroupLayout; upsample: GPUBindGroupLayout },
  ) {
    this.targets = new TargetPool(device);
    this.denoiseUniforms = new UniformBlock(device, 'Bilateral', BILATERAL_UNIFORMS);
    this.upsampleUniforms = new UniformBlock(device, 'Bilateral', BILATERAL_UNIFORMS);
  }

  static async create(device: GPUDevice): Promise<BilateralFilter> {
    const code = wgsl(commonWgsl, fullscreenWgsl, screenSpaceWgsl, uniformStruct('Bilateral', BILATERAL_UNIFORMS), bilateralWgsl);
    const module = await createShader(device, 'bilateral', code);
    const uniform: GPUBindGroupLayoutEntry = { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } };
    const tex = (binding: number, sampleType: GPUTextureSampleType = 'unfilterable-float'): GPUBindGroupLayoutEntry => ({
      binding,
      visibility: GPUShaderStage.FRAGMENT,
      texture: { sampleType },
    });
    const denoise = device.createBindGroupLayout({ label: 'bilateral-denoise', entries: [uniform, tex(1), tex(2)] });
    const upsample = device.createBindGroupLayout({
      label: 'bilateral-upsample',
      entries: [uniform, tex(1), tex(2), tex(3, 'depth'), tex(4)],
    });
    const pipeline = (layout: GPUBindGroupLayout, entryPoint: string) =>
      device.createRenderPipelineAsync({
        label: `bilateral:${entryPoint}`,
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        vertex: { module, entryPoint: 'vs_fullscreen' },
        fragment: { module, entryPoint, targets: [{ format: OUT_FORMAT }] },
      });
    const [denoisePipeline, upsamplePipeline] = await Promise.all([pipeline(denoise, 'fs_denoise'), pipeline(upsample, 'fs_upsample')]);
    return new BilateralFilter(device, { denoise: denoisePipeline, upsample: upsamplePipeline }, { denoise, upsample });
  }

  /** Edge-aware blur of `signal` (any float format) guided by `geometry` of the same size. */
  denoise(
    encoder: GPUCommandEncoder,
    signal: GPUTexture,
    geometry: GPUTexture,
    o: BilateralOptions & { radius: number },
  ): GPUTexture {
    const radius = Math.min(MAX_DENOISE_RADIUS, Math.max(0, Math.round(o.radius)));
    if (radius === 0) return signal;
    const out = this.targets.get('denoise', { size: [signal.width, signal.height], format: OUT_FORMAT });
    this.denoiseUniforms
      .set('radius', radius)
      .set('spatialSigma', Math.max(0.5, radius * 0.75))
      .set('depthSigma', o.depthSigma ?? 0.03)
      .set('normalPower', o.normalPower ?? 8)
      .upload();
    const group = this.device.createBindGroup({
      label: 'bilateral-denoise',
      layout: this.layouts.denoise,
      entries: [
        { binding: 0, resource: { buffer: this.denoiseUniforms.buffer } },
        { binding: 1, resource: signal.createView() },
        { binding: 2, resource: geometry.createView() },
      ],
    });
    this.draw(encoder, this.pipelines.denoise, group, out.view);
    return out.texture;
  }

  /**
   * Upsamples `signal` (with its packed `geometry`) to the size of `full.depth`, guided by the
   * full-resolution depth and view normal. Background pixels get `background` (default: an
   * unoccluded AO texel, normal (0, 0, 1) and visibility 1).
   */
  upsample(
    encoder: GPUCommandEncoder,
    signal: GPUTexture,
    geometry: GPUTexture,
    full: { depth: GPUTexture; normal: GPUTexture },
    cam: ScreenCamera,
    o: BilateralOptions & { background?: readonly [number, number, number, number] } = {},
  ): GPUTexture {
    const out = this.targets.get('upsample', { size: [full.depth.width, full.depth.height], format: OUT_FORMAT });
    this.upsampleUniforms
      .set('background', o.background ?? [0, 0, 1, 1])
      .set('depthSigma', o.depthSigma ?? 0.05)
      .set('normalPower', o.normalPower ?? 4)
      .set('near', cam.near)
      .set('far', cam.far)
      .upload();
    const group = this.device.createBindGroup({
      label: 'bilateral-upsample',
      layout: this.layouts.upsample,
      entries: [
        { binding: 0, resource: { buffer: this.upsampleUniforms.buffer } },
        { binding: 1, resource: signal.createView() },
        { binding: 2, resource: geometry.createView() },
        { binding: 3, resource: full.depth.createView() },
        { binding: 4, resource: full.normal.createView() },
      ],
    });
    this.draw(encoder, this.pipelines.upsample, group, out.view);
    return out.texture;
  }

  private draw(encoder: GPUCommandEncoder, pipeline: GPURenderPipeline, group: GPUBindGroup, view: GPUTextureView): void {
    const pass = encoder.beginRenderPass({
      label: pipeline.label,
      colorAttachments: [{ view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.draw(3);
    pass.end();
  }

  /** Frees the output targets (they come back on the next call). */
  trim(): void {
    this.targets.destroy();
  }

  destroy(): void {
    this.targets.destroy();
    this.denoiseUniforms.destroy();
    this.upsampleUniforms.destroy();
  }
}
