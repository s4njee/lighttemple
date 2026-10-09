/**
 * Shared HDR post-processing: scene target → bloom → exposure/tonemap →
 * vignette/grain → dithered sRGB on the swapchain.
 *
 * Experiments render premultiplied linear HDR into `scene(size)` (alpha =
 * coverage), then call `render()` to finish the frame. The standard "Post"
 * params group is added to the panel, so every experiment gets the same
 * exposure/tonemap/bloom controls.
 *
 *   const post = await PostProcess.create(ctx, { bloom: 0.05 });
 *   // per frame:
 *   const hdr = post.scene(size);            // render into hdr.view
 *   post.render(encoder, swapchainView, { time });
 */

import commonWgsl from '../shaders/common.wgsl?raw';
import fullscreenWgsl from '../shaders/fullscreen.wgsl?raw';
import type { ExperimentContext } from '../experiments/types';
import { createShader, wgsl } from '../gpu/shader';
import { TargetPool, type Target } from '../gpu/targets';
import { UniformBlock } from '../gpu/uniforms';
import postWgsl from './post.wgsl?raw';

export const HDR_FORMAT: GPUTextureFormat = 'rgba16float';
const MAX_BLOOM_LEVELS = 6;
/** Stop downsampling once the smaller side would drop below this. */
const MIN_BLOOM_SIZE = 8;

export const TONEMAP = { None: 0, ACES: 1, AgX: 2 } as const;

export interface PostDefaults {
  exposure?: number;
  tonemap?: (typeof TONEMAP)[keyof typeof TONEMAP];
  bloom?: number;
  bloomRadius?: number;
  vignette?: number;
  grain?: number;
}

export interface PostRenderOptions {
  /** Seconds; animates grain. */
  time: number;
  /** Skip exposure, bloom, tonemapping, vignette and grain (for debug views). */
  raw?: boolean;
  /**
   * Display-space linear colour shown where scene alpha < 1. Lets a scene
   * dissolve into a flat page colour with no tonemapping mismatch. Default black.
   */
  background?: readonly [number, number, number];
}

export class PostProcess {
  private readonly targets: TargetPool;
  private readonly uniforms: UniformBlock<typeof POST_UNIFORMS>;
  private readonly sampler: GPUSampler;
  private readonly black: GPUTexture;
  private readonly bindGroups = new WeakMap<
    GPURenderPipeline,
    WeakMap<GPUTexture, { bloom: GPUTexture | undefined; group: GPUBindGroup }>
  >();
  private sceneTarget: Target | null = null;

  private constructor(
    private readonly device: GPUDevice,
    private readonly settings: ReturnType<typeof addParams>,
    uniforms: UniformBlock<typeof POST_UNIFORMS>,
    private readonly pipelines: {
      down: GPURenderPipeline;
      downKaris: GPURenderPipeline;
      up: GPURenderPipeline;
      composite: GPURenderPipeline;
    },
  ) {
    this.targets = new TargetPool(device);
    this.uniforms = uniforms;
    this.sampler = device.createSampler({
      label: 'post-linear',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });
    this.black = device.createTexture({
      label: 'post-black',
      size: [1, 1],
      format: HDR_FORMAT,
      usage: GPUTextureUsage.TEXTURE_BINDING,
    });
  }

  static async create(ctx: ExperimentContext, defaults: PostDefaults = {}): Promise<PostProcess> {
    const { device, format } = ctx;
    const settings = addParams(ctx, defaults);
    const uniforms = new UniformBlock(device, 'Post', POST_UNIFORMS);
    const module = await createShader(device, 'post', wgsl(commonWgsl, fullscreenWgsl, uniforms.wgslStruct, postWgsl));
    const make = (entryPoint: string, target: GPUTextureFormat, blend?: GPUBlendState) =>
      device.createRenderPipelineAsync({
        label: `post:${entryPoint}`,
        layout: 'auto',
        vertex: { module, entryPoint: 'vs_fullscreen' },
        fragment: { module, entryPoint, targets: [{ format: target, blend }] },
      });
    const additive: GPUBlendState = {
      color: { operation: 'add', srcFactor: 'one', dstFactor: 'one' },
      alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one' },
    };
    const [down, downKaris, up, composite] = await Promise.all([
      make('fs_down', HDR_FORMAT),
      make('fs_down_karis', HDR_FORMAT),
      make('fs_up', HDR_FORMAT, additive),
      make('fs_composite', format),
    ]);
    return new PostProcess(device, settings, uniforms, { down, downKaris, up, composite });
  }

  /** The HDR scene target for this frame size (premultiplied rgba16float). */
  scene(size: { width: number; height: number }): Target {
    this.sceneTarget = this.targets.get('scene', { size: [size.width, size.height], format: HDR_FORMAT });
    return this.sceneTarget;
  }

  /** Encodes bloom + composite from the scene target into `output`. */
  render(encoder: GPUCommandEncoder, output: GPUTextureView, opts: PostRenderOptions): void {
    const scene = this.sceneTarget;
    if (!scene) throw new Error('PostProcess.render() called before scene()');
    const s = this.settings;
    const raw = opts.raw ?? false;
    const doBloom = !raw && s.bloom > 0;

    // Bloom pyramid: level i is (1/2)^(i+1) of full resolution.
    const levels: Target[] = [];
    if (doBloom) {
      for (let i = 0; i < MAX_BLOOM_LEVELS; i++) {
        const scale = 1 / 2 ** (i + 1);
        if (Math.min(scene.width, scene.height) * scale < MIN_BLOOM_SIZE) break;
        levels.push(this.targets.get(`bloom${i}`, { size: [scene.width, scene.height], scale, format: HDR_FORMAT }));
      }
    }

    this.uniforms
      .set('exposure', s.exposure)
      .set('tonemap', s.tonemap)
      .set('bloomStrength', levels.length > 0 ? s.bloom : 0)
      .set('bloomRadius', s.bloomRadius)
      .set('bloomLevels', Math.max(1, levels.length))
      .set('vignette', s.vignette)
      .set('grain', s.grain)
      .set('time', opts.time)
      .set('aspect', scene.width / scene.height)
      .set('raw', raw ? 1 : 0)
      .set('background', opts.background ?? [0, 0, 0])
      .upload();

    if (levels.length > 0) {
      levels.forEach((dst, i) => {
        const src = i === 0 ? scene.texture : levels[i - 1].texture;
        this.fullscreen(encoder, i === 0 ? this.pipelines.downKaris : this.pipelines.down, src, dst.view, 'clear');
      });
      for (let i = levels.length - 1; i > 0; i--) {
        this.fullscreen(encoder, this.pipelines.up, levels[i].texture, levels[i - 1].view, 'load');
      }
    }

    this.fullscreen(encoder, this.pipelines.composite, scene.texture, output, 'clear', levels[0]?.texture ?? this.black);
  }

  private fullscreen(
    encoder: GPUCommandEncoder,
    pipeline: GPURenderPipeline,
    src: GPUTexture,
    dst: GPUTextureView,
    loadOp: GPULoadOp,
    bloom?: GPUTexture,
  ): void {
    const pass = encoder.beginRenderPass({
      label: pipeline.label,
      colorAttachments: [{ view: dst, loadOp, storeOp: 'store', clearValue: [0, 0, 0, 0] }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.bindGroup(pipeline, src, bloom));
    pass.draw(3);
    pass.end();
  }

  /** Bind groups are cached per (pipeline, source texture); textures only change on resize. */
  private bindGroup(pipeline: GPURenderPipeline, src: GPUTexture, bloom?: GPUTexture): GPUBindGroup {
    let perPipeline = this.bindGroups.get(pipeline);
    if (!perPipeline) this.bindGroups.set(pipeline, (perPipeline = new WeakMap()));
    const hit = perPipeline.get(src);
    if (hit && hit.bloom === bloom) return hit.group;

    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: src.createView() },
      { binding: 2, resource: this.sampler },
    ];
    // 'auto' layouts only contain the bindings each entry point actually uses.
    if (pipeline !== this.pipelines.down && pipeline !== this.pipelines.downKaris) {
      entries.push({ binding: 3, resource: { buffer: this.uniforms.buffer } });
    }
    if (bloom) entries.push({ binding: 1, resource: bloom.createView() });
    const group = this.device.createBindGroup({ label: pipeline.label, layout: pipeline.getBindGroupLayout(0), entries });
    perPipeline.set(src, { bloom, group });
    return group;
  }

  destroy(): void {
    this.targets.destroy();
    this.uniforms.destroy();
    this.black.destroy();
  }
}

const POST_UNIFORMS = {
  exposure: 'f32',
  tonemap: 'u32',
  bloomStrength: 'f32',
  bloomRadius: 'f32',
  bloomLevels: 'f32',
  vignette: 'f32',
  grain: 'f32',
  time: 'f32',
  aspect: 'f32',
  raw: 'u32',
  background: 'vec3f',
} as const;

function addParams(ctx: ExperimentContext, d: PostDefaults) {
  return ctx.params.add(
    {
      exposure: { value: d.exposure ?? 1, min: 0.05, max: 4, step: 0.01 },
      tonemap: { value: d.tonemap ?? TONEMAP.AgX, options: TONEMAP },
      bloom: {
        value: d.bloom ?? 0.04,
        min: 0,
        max: 0.4,
        step: 0.005,
        hint: 'Energy-conserving mix of a 6-level blur pyramid. No threshold, so only genuinely bright pixels glow noticeably.',
      },
      bloomRadius: { value: d.bloomRadius ?? 1, min: 0.25, max: 3, step: 0.05, label: 'Bloom radius' },
      vignette: { value: d.vignette ?? 0.25, min: 0, max: 1, step: 0.01 },
      grain: { value: d.grain ?? 0.03, min: 0, max: 0.2, step: 0.005 },
    },
    'Post',
  );
}
