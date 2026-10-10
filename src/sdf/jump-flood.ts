/**
 * Signed distance fields from any coverage mask, on the GPU (jump flooding).
 *
 *   const jfa = new JumpFlood(device);
 *   jfa.encode(encoder, maskTexture, sdfTexture);   // sdf.r = signed distance in texels
 *
 * `mask` is any float-sampleable texture whose *alpha* is coverage (inside >= 0.5).
 * `out` must be an rgba16float texture of the same size with STORAGE_BINDING usage.
 * Scratch textures and uniforms are cached per size, so re-encoding every time a
 * mask changes (painting, editing text) costs only the dispatches.
 */

import jfaWgsl from './jfa.wgsl?raw';

interface Scratch {
  a: GPUTexture;
  b: GPUTexture;
  steps: GPUBuffer[];
}

export class JumpFlood {
  private readonly layout: GPUBindGroupLayout;
  private readonly seed: GPUComputePipeline;
  private readonly flood: GPUComputePipeline;
  private readonly resolve: GPUComputePipeline;
  private scratch = new Map<string, Scratch>();

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

  private scratchFor(width: number, height: number): Scratch {
    const key = `${width}x${height}`;
    let s = this.scratch.get(key);
    if (!s) {
      const d = this.device;
      const tex = () =>
        d.createTexture({ label: 'jfa-seeds', size: [width, height], format: 'rg32float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING });
      const stepSizes = [0];
      for (let step = 2 ** Math.floor(Math.log2(Math.max(width, height))); step >= 1; step >>= 1) stepSizes.push(step);
      // One extra 1-step pass (JFA+1) fixes most of the algorithm's rare misses.
      stepSizes.push(1);
      const steps = stepSizes.map((step) => {
        const buf = d.createBuffer({ label: 'jfa-step', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        d.queue.writeBuffer(buf, 0, new Int32Array([step, 0, 0, 0]));
        return buf;
      });
      s = { a: tex(), b: tex(), steps };
      this.scratch.set(key, s);
    }
    return s;
  }

  encode(encoder: GPUCommandEncoder, mask: GPUTexture, out: GPUTexture): void {
    const { width, height } = mask;
    const s = this.scratchFor(width, height);
    const maskView = mask.createView();
    const outView = out.createView();
    const group = (src: GPUTexture, dst: GPUTexture, uniform: GPUBuffer) =>
      this.device.createBindGroup({
        layout: this.layout,
        entries: [
          { binding: 0, resource: maskView },
          { binding: 1, resource: src.createView() },
          { binding: 2, resource: dst.createView() },
          { binding: 3, resource: outView },
          { binding: 4, resource: { buffer: uniform } },
        ],
      });
    const wx = Math.ceil(width / 8);
    const wy = Math.ceil(height / 8);
    const dispatch = (pipeline: GPUComputePipeline, bg: GPUBindGroup) => {
      const pass = encoder.beginComputePass({ label: pipeline.label });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(wx, wy);
      pass.end();
    };
    dispatch(this.seed, group(s.b, s.a, s.steps[0]));
    let src = s.a;
    let dst = s.b;
    for (let i = 1; i < s.steps.length; i++) {
      dispatch(this.flood, group(src, dst, s.steps[i]));
      [src, dst] = [dst, src];
    }
    dispatch(this.resolve, group(src, dst, s.steps[0]));
  }

  /** Frees cached scratch textures (they are recreated on demand). */
  trim(): void {
    for (const s of this.scratch.values()) {
      s.a.destroy();
      s.b.destroy();
      for (const b of s.steps) b.destroy();
    }
    this.scratch.clear();
  }

  destroy(): void {
    this.trim();
  }
}
