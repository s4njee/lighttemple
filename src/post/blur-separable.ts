/**
 * Separable Gaussian blur of a float texture, in place: horizontal into a scratch target, then
 * vertical back into the source. Uses exact texel loads, so any renderable float format works
 * (including rgba32float without `float32-filterable`). Cost: 2 fullscreen passes, (2r+1) loads
 * per texel per pass, and one scratch texture the size and format of the source.
 *
 *   const blur = await SeparableBlur.create(device);
 *   blur.run(encoder, moments, 3);   // radius in texels, sigma defaults to radius / 2
 *
 * The H/V parameters live in two uniform buffers written at `run()` time, so each instance supports
 * one `run()` per queue submission.
 */

import fullscreenWgsl from '../shaders/fullscreen.wgsl?raw';
import { createShader, wgsl } from '../gpu/shader';
import { TargetPool } from '../gpu/targets';
import { UniformBlock, uniformStruct } from '../gpu/uniforms';
import blurWgsl from './blur-separable.wgsl?raw';

const SCHEMA = { direction: 'vec2f', radius: 'f32', sigma: 'f32' } as const;
/** Largest radius accepted, in texels. */
export const MAX_BLUR_RADIUS = 16;

export class SeparableBlur {
  private readonly pipelines = new Map<GPUTextureFormat, GPURenderPipeline>();
  private readonly targets: TargetPool;
  private readonly passes: [UniformBlock<typeof SCHEMA>, UniformBlock<typeof SCHEMA>];

  private constructor(
    private readonly device: GPUDevice,
    private readonly module: GPUShaderModule,
    private readonly layout: GPUBindGroupLayout,
  ) {
    this.targets = new TargetPool(device);
    this.passes = [new UniformBlock(device, 'Blur', SCHEMA), new UniformBlock(device, 'Blur', SCHEMA)];
  }

  static async create(device: GPUDevice): Promise<SeparableBlur> {
    const module = await createShader(device, 'blur-separable', wgsl(fullscreenWgsl, uniformStruct('Blur', SCHEMA), blurWgsl));
    const layout = device.createBindGroupLayout({
      label: 'blur-separable',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
      ],
    });
    return new SeparableBlur(device, module, layout);
  }

  private pipeline(format: GPUTextureFormat): GPURenderPipeline {
    let p = this.pipelines.get(format);
    if (!p) {
      p = this.device.createRenderPipeline({
        label: `blur-separable:${format}`,
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
        vertex: { module: this.module, entryPoint: 'vs_fullscreen' },
        fragment: { module: this.module, entryPoint: 'fs_blur', targets: [{ format }] },
      });
      this.pipelines.set(format, p);
    }
    return p;
  }

  /**
   * Blurs mip 0 of `texture` (needs RENDER_ATTACHMENT | TEXTURE_BINDING) in place.
   * `radius` is clamped to [0, MAX_BLUR_RADIUS]; 0 does nothing.
   */
  run(encoder: GPUCommandEncoder, texture: GPUTexture, radius: number, sigma = radius / 2): void {
    const r = Math.min(MAX_BLUR_RADIUS, Math.max(0, Math.round(radius)));
    if (r === 0) return;
    const scratch = this.targets.get('scratch', { size: [texture.width, texture.height], format: texture.format });
    const pipeline = this.pipeline(texture.format);
    const steps: [GPUTexture, GPUTexture, [number, number]][] = [
      [texture, scratch.texture, [1, 0]],
      [scratch.texture, texture, [0, 1]],
    ];
    steps.forEach(([src, dst, direction], i) => {
      const u = this.passes[i];
      u.set('direction', direction).set('radius', r).set('sigma', Math.max(0.5, sigma)).upload();
      const pass = encoder.beginRenderPass({
        label: 'blur-separable',
        colorAttachments: [{ view: dst.createView({ baseMipLevel: 0, mipLevelCount: 1 }), loadOp: 'clear', storeOp: 'store' }],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(
        0,
        this.device.createBindGroup({
          layout: this.layout,
          entries: [
            { binding: 0, resource: { buffer: u.buffer } },
            { binding: 1, resource: src.createView({ baseMipLevel: 0, mipLevelCount: 1 }) },
          ],
        }),
      );
      pass.draw(3);
      pass.end();
    });
  }

  /** Frees the scratch texture (it comes back on the next `run()`). */
  trim(): void {
    this.targets.release('scratch');
  }

  destroy(): void {
    this.targets.destroy();
    this.passes.forEach((u) => u.destroy());
  }
}
