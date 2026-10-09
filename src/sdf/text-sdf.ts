/**
 * Runtime text → signed distance field.
 *
 * Text is rasterised with Canvas 2D (so any CSS font works, including the
 * page's own web fonts, and there is no offline MSDF/atlas toolchain), then
 * converted on the GPU with the jump flooding algorithm. The result is an
 * rgba16float texture whose red channel is the signed distance to the glyph
 * outline in *pixels* (negative inside), ready for extrusion, soft shadows,
 * glows and outlines.
 *
 * Trade-off vs. MSDF: single-channel SDFs round off sharp corners when
 * magnified far beyond their raster size. For lighting effects (soft
 * shadows, bevels, glows) that is invisible; for crisp giant glyph edges,
 * rasterise at a larger size.
 */

import jfaWgsl from './jfa.wgsl?raw';

export interface TextSdfOptions {
  text: string;
  /** CSS font-family list, e.g. '"Iowan Old Style", Georgia, serif'. */
  family: string;
  /** CSS font-weight. Default 700. */
  weight?: number | string;
  /** Raster font size in px (texture resolution scales with it). Default 160. */
  size?: number;
  /** Empty margin around the glyphs in px, which bounds how far the field reaches. Default size * 0.35. */
  padding?: number;
}

export interface TextSdf {
  /** rgba16float, r = signed distance in pixels. */
  texture: GPUTexture;
  width: number;
  height: number;
  padding: number;
  destroy(): void;
}

export class TextSdfGenerator {
  private readonly layout: GPUBindGroupLayout;
  private readonly seed: GPUComputePipeline;
  private readonly flood: GPUComputePipeline;
  private readonly resolve: GPUComputePipeline;

  constructor(private readonly device: GPUDevice) {
    const C = GPUShaderStage.COMPUTE;
    this.layout = device.createBindGroupLayout({
      label: 'jfa',
      entries: [
        { binding: 0, visibility: C, texture: { sampleType: 'float' } },
        { binding: 1, visibility: C, texture: { sampleType: 'unfilterable-float' } },
        { binding: 2, visibility: C, storageTexture: { access: 'write-only', format: 'rg32float' } },
        { binding: 3, visibility: C, storageTexture: { access: 'write-only', format: 'rgba16float' } },
        { binding: 4, visibility: C, buffer: { type: 'uniform' } },
      ],
    });
    const module = device.createShaderModule({ label: 'jfa', code: jfaWgsl });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    const make = (entryPoint: string) => device.createComputePipeline({ label: `jfa:${entryPoint}`, layout, compute: { module, entryPoint } });
    this.seed = make('cs_seed');
    this.flood = make('cs_flood');
    this.resolve = make('cs_resolve');
  }

  async generate(opts: TextSdfOptions): Promise<TextSdf> {
    const size = opts.size ?? 160;
    const padding = Math.round(opts.padding ?? size * 0.35);
    const fontSpec = `${opts.weight ?? 700} ${size}px ${opts.family}`;
    try {
      await document.fonts.load(fontSpec, opts.text);
    } catch {
      // Fall back to whatever the browser resolves.
    }

    // 1. Rasterise.
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d')!;
    ctx.font = fontSpec;
    const m = ctx.measureText(opts.text || ' ');
    const left = Math.ceil(m.actualBoundingBoxLeft);
    const glyphW = Math.max(1, Math.ceil(m.actualBoundingBoxLeft + m.actualBoundingBoxRight));
    const glyphH = Math.max(1, Math.ceil(m.actualBoundingBoxAscent + m.actualBoundingBoxDescent));
    canvas.width = glyphW + padding * 2;
    canvas.height = glyphH + padding * 2;
    ctx.font = fontSpec; // resizing resets context state
    ctx.fillStyle = '#fff';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(opts.text, padding + left, padding + Math.ceil(m.actualBoundingBoxAscent));
    const { width, height } = canvas;

    const d = this.device;
    const mask = d.createTexture({
      label: 'text-mask',
      size: [width, height],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    d.queue.copyExternalImageToTexture({ source: canvas }, { texture: mask }, [width, height]);

    // 2. Jump flood.
    const seedTex = () =>
      d.createTexture({ size: [width, height], format: 'rg32float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING });
    const a = seedTex();
    const b = seedTex();
    const sdf = d.createTexture({
      label: 'text-sdf',
      size: [width, height],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });

    const steps: number[] = [];
    for (let s = 2 ** Math.floor(Math.log2(Math.max(width, height))); s >= 1; s >>= 1) steps.push(s);
    // One extra 1-step pass (JFA+1) fixes most of the algorithm's rare misses.
    steps.push(1);
    const buffers = [0, ...steps].map((step) => {
      const buf = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      d.queue.writeBuffer(buf, 0, new Int32Array([step, 0, 0, 0]));
      return buf;
    });
    const group = (src: GPUTexture, dst: GPUTexture, uniform: GPUBuffer) =>
      d.createBindGroup({
        layout: this.layout,
        entries: [
          { binding: 0, resource: mask.createView() },
          { binding: 1, resource: src.createView() },
          { binding: 2, resource: dst.createView() },
          { binding: 3, resource: sdf.createView() },
          { binding: 4, resource: { buffer: uniform } },
        ],
      });

    const encoder = d.createCommandEncoder({ label: 'text-sdf' });
    const wx = Math.ceil(width / 8);
    const wy = Math.ceil(height / 8);
    const dispatch = (pipeline: GPUComputePipeline, bg: GPUBindGroup) => {
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(wx, wy);
      pass.end();
    };
    dispatch(this.seed, group(b, a, buffers[0]));
    let src = a;
    let dst = b;
    steps.forEach((_, i) => {
      dispatch(this.flood, group(src, dst, buffers[i + 1]));
      [src, dst] = [dst, src];
    });
    dispatch(this.resolve, group(src, dst, buffers[0]));
    d.queue.submit([encoder.finish()]);
    await d.queue.onSubmittedWorkDone();

    mask.destroy();
    a.destroy();
    b.destroy();
    for (const buf of buffers) buf.destroy();
    return { texture: sdf, width, height, padding, destroy: () => sdf.destroy() };
  }
}
