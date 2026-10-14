/**
 * Temporal accumulation for noisy screen-space signals (AO, indirect light), at the signal's own
 * resolution, separate from TAA:
 *
 *   const history = await TemporalAccumulator.create(device);
 *   const geo = gbuffer.resolve(encoder, 0.5, cam);
 *   const raw = gtao.run(encoder, geo.current, cam, settings);
 *   const smooth = history.run(encoder, { signal: raw, geometry: geo.current, previousGeometry: geo.previous,
 *                                         cam, view: camera.view, viewProj: jitteredViewProj, maxFrames: 16 });
 *
 * Why not just TAA? TAA's colour clamp is built for aliasing: it clips history to the current
 * neighbourhood's spread, and per-pixel AO noise is wide enough that the clip either keeps the
 * noise or ghosts. This pass instead trusts history wherever reprojection lands on the *same
 * surface* (last frame's linear depth matches) and resets it where it doesn't (disocclusion), so it
 * can average 16–32 frames of a static signal. Each pixel blends with 1 / n, n = its history
 * length, so freshly revealed pixels converge as fast as a plain average. Signals whose value
 * changes over time (a moving light's bounce) can opt into a neighbourhood clip (`clampSigma`).
 *
 * The camera's jittered matrix is fine: geometry and history are both in jittered screen space.
 * There is no velocity buffer, so moving objects leave a short trail of their old AO.
 * One `run()` per instance per frame.
 */

import { mat4, type Mat4 } from 'wgpu-matrix';
import commonWgsl from '../shaders/common.wgsl?raw';
import fullscreenWgsl from '../shaders/fullscreen.wgsl?raw';
import screenSpaceWgsl from '../shaders/screen-space.wgsl?raw';
import { createShader, wgsl } from '../gpu/shader';
import { TargetPool } from '../gpu/targets';
import { UniformBlock, uniformStruct } from '../gpu/uniforms';
import type { ScreenCamera } from './gbuffer';
import accumulateWgsl from './temporal-accumulate.wgsl?raw';

const SIGNAL_FORMAT: GPUTextureFormat = 'rgba16float';
const COUNT_FORMAT: GPUTextureFormat = 'r16float';

const ACCUMULATE_UNIFORMS = {
  viewToPrevClip: 'mat4x4f',
  projScale: 'vec2f',
  jitter: 'vec2f',
  maxFrames: 'f32',
  clampSigma: 'f32',
  depthTolerance: 'f32',
  historyValid: 'u32',
} as const;

export interface AccumulateInput {
  /** This frame's signal (rgba16float), the same size as `geometry`. */
  signal: GPUTexture;
  /** This frame's packed geometry (GBuffer.resolve().current). */
  geometry: GPUTexture;
  /** Last frame's packed geometry (GBuffer.resolve().previous); null restarts the history. */
  previousGeometry: GPUTexture | null;
  cam: ScreenCamera;
  /** This frame's view matrix. */
  view: Mat4;
  /** This frame's view-projection, *as rendered* (jittered under TAA). */
  viewProj: Mat4;
  /** History length cap: the blend weight never drops below 1 / maxFrames. */
  maxFrames: number;
  /** Clip history to mean ± k·σ of the current 3×3 neighbourhood; 0 (default) disables the clip. */
  clampSigma?: number;
  /** Relative depth difference above which a history texel counts as another surface (default 0.04). */
  depthTolerance?: number;
}

export class TemporalAccumulator {
  private readonly targets: TargetPool;
  private readonly uniforms: UniformBlock<typeof ACCUMULATE_UNIFORMS>;
  private prevViewProj: Mat4 | null = null;
  private index = 0;
  private key = '';
  private lastCount: GPUTexture | null = null;

  private constructor(
    private readonly device: GPUDevice,
    private readonly pipeline: GPURenderPipeline,
    private readonly layout: GPUBindGroupLayout,
  ) {
    this.targets = new TargetPool(device);
    this.uniforms = new UniformBlock(device, 'Accumulate', ACCUMULATE_UNIFORMS);
  }

  static async create(device: GPUDevice): Promise<TemporalAccumulator> {
    const code = wgsl(commonWgsl, fullscreenWgsl, screenSpaceWgsl, uniformStruct('Accumulate', ACCUMULATE_UNIFORMS), accumulateWgsl);
    const module = await createShader(device, 'temporal-accumulate', code);
    const tex = (binding: number): GPUBindGroupLayoutEntry => ({
      binding,
      visibility: GPUShaderStage.FRAGMENT,
      texture: { sampleType: 'unfilterable-float' },
    });
    const layout = device.createBindGroupLayout({
      label: 'temporal-accumulate',
      entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }, tex(1), tex(2), tex(3), tex(4), tex(5)],
    });
    const pipeline = await device.createRenderPipelineAsync({
      label: 'temporal-accumulate',
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      vertex: { module, entryPoint: 'vs_fullscreen' },
      fragment: { module, entryPoint: 'fs_accumulate', targets: [{ format: SIGNAL_FORMAT }, { format: COUNT_FORMAT }] },
    });
    return new TemporalAccumulator(device, pipeline, layout);
  }

  /** Per-pixel history length (r16float, frames) written by the last `run()`: a disocclusion map. */
  get historyLength(): GPUTexture | null {
    return this.lastCount;
  }

  /** Drop the history (camera cut, a parameter that changes the signal everywhere). */
  reset(): void {
    this.prevViewProj = null;
  }

  run(encoder: GPUCommandEncoder, input: AccumulateInput): GPUTexture {
    const { signal, geometry } = input;
    const size = [signal.width, signal.height] as const;
    const key = `${signal.width}x${signal.height}`;
    if (key !== this.key) {
      this.key = key;
      this.prevViewProj = null;
    }
    const valid = this.prevViewProj !== null && input.previousGeometry !== null;
    const read = this.index;
    const write = 1 - this.index;
    const histSignal = this.targets.get(`signal${read}`, { size, format: SIGNAL_FORMAT });
    const histCount = this.targets.get(`count${read}`, { size, format: COUNT_FORMAT });
    const outSignal = this.targets.get(`signal${write}`, { size, format: SIGNAL_FORMAT });
    const outCount = this.targets.get(`count${write}`, { size, format: COUNT_FORMAT });

    const viewToPrevClip = mat4.multiply(this.prevViewProj ?? input.viewProj, mat4.inverse(input.view));
    this.uniforms
      .set('viewToPrevClip', viewToPrevClip)
      .set('projScale', input.cam.projScale)
      .set('jitter', input.cam.jitter)
      .set('maxFrames', Math.max(1, input.maxFrames))
      .set('clampSigma', input.clampSigma ?? 0)
      .set('depthTolerance', input.depthTolerance ?? 0.04)
      .set('historyValid', valid ? 1 : 0)
      .upload();

    // Bind groups are cheap next to this pass; the inputs ping-pong, so build one per frame.
    const group = this.device.createBindGroup({
      label: 'temporal-accumulate',
      layout: this.layout,
      entries: [
        { binding: 0, resource: { buffer: this.uniforms.buffer } },
        { binding: 1, resource: signal.createView() },
        { binding: 2, resource: geometry.createView() },
        { binding: 3, resource: (input.previousGeometry ?? geometry).createView() },
        { binding: 4, resource: histSignal.view },
        { binding: 5, resource: histCount.view },
      ],
    });
    const pass = encoder.beginRenderPass({
      label: 'temporal-accumulate',
      colorAttachments: [
        { view: outSignal.view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] },
        { view: outCount.view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] },
      ],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, group);
    pass.draw(3);
    pass.end();

    this.prevViewProj = mat4.clone(input.viewProj) as Mat4;
    this.index = write;
    this.lastCount = outCount.texture;
    return outSignal.texture;
  }

  /** Frees the history (it restarts on the next `run()`). */
  trim(): void {
    this.targets.destroy();
    this.prevViewProj = null;
    this.lastCount = null;
  }

  destroy(): void {
    this.targets.destroy();
    this.uniforms.destroy();
  }
}
