/**
 * Temporal anti-aliasing / accumulation ("TAA-lite").
 *
 * Usage per frame:
 *   const proj = taa.jitter(camera.viewProjection, size);  // render the scene with this
 *   ... render HDR scene + a sampleable depth buffer ...
 *   const resolved = taa.resolve(encoder, { color, depth, viewProj: camera.viewProjection });
 *   post.render(encoder, out, { time, source: resolved });
 *
 * Beyond smoothing edges, TAA turns per-pixel noise into extra samples over
 * time: stochastic shadows, AO and volumetrics can use a fraction of the taps
 * if they rotate their noise every frame (see `frameNoiseOffset`).
 */

import { mat4, type Mat4 } from 'wgpu-matrix';
import commonWgsl from '../shaders/common.wgsl?raw';
import fullscreenWgsl from '../shaders/fullscreen.wgsl?raw';
import { createShader, wgsl } from '../gpu/shader';
import { TargetPool } from '../gpu/targets';
import { UniformBlock } from '../gpu/uniforms';
import { HDR_FORMAT } from './post';
import taaWgsl from './taa.wgsl?raw';

const TAA_UNIFORMS = {
  invViewProj: 'mat4x4f',
  prevViewProj: 'mat4x4f',
  feedback: 'f32',
  gamma: 'f32',
  historyValid: 'u32',
} as const;

/** Radical inverse in `base`: the Halton low-discrepancy sequence. */
function halton(index: number, base: number): number {
  let f = 1;
  let r = 0;
  for (let i = index; i > 0; i = Math.floor(i / base)) {
    f /= base;
    r += f * (i % base);
  }
  return r;
}

const JITTER_SAMPLES = 16;

export interface TaaResolveInput {
  /** Current frame, HDR, rendered with the jittered matrix. */
  color: GPUTexture;
  /** Single-sampled depth of the current frame (needs TEXTURE_BINDING). */
  depth: GPUTexture;
  /** Current *unjittered* view-projection. */
  viewProj: Mat4;
}

export class TemporalAA {
  /** History weight, 0..1. Higher = smoother and less noisy, but more ghosting. */
  feedback = 0.9;
  /** Variance-clip box size in standard deviations. Lower = less ghosting, more flicker. */
  gamma = 1.25;

  private frame = 0;
  private prevViewProj: Mat4 | null = null;
  private historyIndex = 0;
  private historyKey = '';
  private readonly targets: TargetPool;
  private readonly uniforms: UniformBlock<typeof TAA_UNIFORMS>;
  private readonly sampler: GPUSampler;
  private readonly bindGroups = new WeakMap<GPUTexture, Map<GPUTexture, { depth: GPUTexture; group: GPUBindGroup }>>();

  private constructor(
    private readonly device: GPUDevice,
    private readonly pipeline: GPURenderPipeline,
    uniforms: UniformBlock<typeof TAA_UNIFORMS>,
  ) {
    this.targets = new TargetPool(device);
    this.uniforms = uniforms;
    this.sampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });
  }

  static async create(device: GPUDevice): Promise<TemporalAA> {
    const uniforms = new UniformBlock(device, 'Taa', TAA_UNIFORMS);
    const module = await createShader(device, 'taa', wgsl(commonWgsl, fullscreenWgsl, uniforms.wgslStruct, taaWgsl));
    const pipeline = await device.createRenderPipelineAsync({
      label: 'taa-resolve',
      layout: 'auto',
      vertex: { module, entryPoint: 'vs_fullscreen' },
      fragment: { module, entryPoint: 'fs_resolve', targets: [{ format: HDR_FORMAT }] },
    });
    return new TemporalAA(device, pipeline, uniforms);
  }

  /** This frame's sub-pixel offset in pixels, each component in (-0.5, 0.5). */
  get jitterPixels(): [number, number] {
    const i = (this.frame % JITTER_SAMPLES) + 1;
    return [halton(i, 2) - 0.5, halton(i, 3) - 0.5];
  }

  /** A per-frame offset for screen-space noise (e.g. IGN), so noise decorrelates over time. */
  get frameNoiseOffset(): number {
    return (this.frame % 64) * 5.588238;
  }

  /** Returns `viewProj` with this frame's sub-pixel jitter applied in clip space. */
  jitter(viewProj: Mat4, size: { width: number; height: number }): Mat4 {
    const [jx, jy] = this.jitterPixels;
    const offset = mat4.translation([(2 * jx) / size.width, (-2 * jy) / size.height, 0]);
    return mat4.multiply(offset, viewProj);
  }

  /** Drop history (e.g. after a cut or a big parameter change). */
  reset(): void {
    this.prevViewProj = null;
  }

  resolve(encoder: GPUCommandEncoder, input: TaaResolveInput): GPUTexture {
    const { color, depth, viewProj } = input;
    const size = [color.width, color.height] as const;
    const key = `${color.width}x${color.height}`;
    if (key !== this.historyKey) {
      this.historyKey = key;
      this.prevViewProj = null;
    }
    const read = this.targets.get(`history${this.historyIndex}`, { size, format: HDR_FORMAT });
    const write = this.targets.get(`history${1 - this.historyIndex}`, { size, format: HDR_FORMAT });

    this.uniforms
      .set('invViewProj', mat4.inverse(viewProj))
      .set('prevViewProj', this.prevViewProj ?? viewProj)
      .set('feedback', this.feedback)
      .set('gamma', this.gamma)
      .set('historyValid', this.prevViewProj ? 1 : 0)
      .upload();

    const pass = encoder.beginRenderPass({
      label: 'taa-resolve',
      colorAttachments: [{ view: write.view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup(color, read.texture, depth));
    pass.draw(3);
    pass.end();

    this.prevViewProj = mat4.clone(viewProj) as Mat4;
    this.historyIndex = 1 - this.historyIndex;
    this.frame++;
    return write.texture;
  }

  private bindGroup(color: GPUTexture, history: GPUTexture, depth: GPUTexture): GPUBindGroup {
    let byHistory = this.bindGroups.get(color);
    if (!byHistory) this.bindGroups.set(color, (byHistory = new Map()));
    const hit = byHistory.get(history);
    if (hit?.depth === depth) return hit.group;
    const group = this.device.createBindGroup({
      label: 'taa-resolve',
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: color.createView() },
        { binding: 1, resource: history.createView() },
        { binding: 2, resource: depth.createView() },
        { binding: 3, resource: this.sampler },
        { binding: 4, resource: { buffer: this.uniforms.buffer } },
      ],
    });
    byHistory.set(history, { depth, group });
    return group;
  }

  destroy(): void {
    this.targets.destroy();
    this.uniforms.destroy();
  }
}
