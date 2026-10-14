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
 *
 * On a transparent canvas (`ctx.transparent`, `alphaMode: 'premultiplied'`) the
 * final pass writes premultiplied alpha instead of compositing over a colour,
 * so the scene's alpha lets the HTML behind the canvas show through.
 */

import commonWgsl from '../shaders/common.wgsl?raw';
import fullscreenWgsl from '../shaders/fullscreen.wgsl?raw';
import type { ParamSink } from '../ui/params';
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
  /**
   * Read from this HDR texture instead of the scene target (same size), e.g.
   * the output of TemporalAA.resolve().
   */
  source?: GPUTexture;
  /**
   * 'opaque' composites over `background` and writes alpha 1. 'premultiplied' writes
   * premultiplied alpha for a canvas configured with `alphaMode: 'premultiplied'` over live
   * HTML (`background` is then ignored): opaque content stays opaque, a black shadow becomes
   * alpha = darkness, and colour never exceeds alpha. Defaults to 'premultiplied' when the
   * context passed to `create()` is transparent, else 'opaque'.
   */
  output?: 'opaque' | 'premultiplied';
  /**
   * How 'premultiplied' alpha is written. Browsers blend canvases in sRGB space. 'linear'
   * (default) remaps the alpha of dark pixels as 1 - (1 - a)^(1/2.2), so semi-transparent black
   * (a shadow) darkens the page by what the opaque path's linear-light blend would; bright pixels
   * keep their coverage. 'coverage' writes the scene alpha unchanged, the plain CSS behaviour
   * (shadows come out darker, with harder-looking edges).
   */
  alphaBlend?: 'linear' | 'coverage';
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
  private premultiplied: GPURenderPipeline | null;

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
    private readonly module: GPUShaderModule,
    private readonly format: GPUTextureFormat,
    premultiplied: GPURenderPipeline | null,
    private readonly transparent: boolean,
  ) {
    this.premultiplied = premultiplied;
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

  /**
   * `params` is optional: inside the gallery the standard "Post" group is added to
   * the panel; standalone pages (e.g. demos/) get fixed settings from `defaults`.
   */
  static async create(ctx: PostContext, defaults: PostDefaults = {}): Promise<PostProcess> {
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
    const transparent = ctx.transparent ?? false;
    const [down, downKaris, up, composite, premultiplied] = await Promise.all([
      make('fs_down', HDR_FORMAT),
      make('fs_down_karis', HDR_FORMAT),
      make('fs_up', HDR_FORMAT, additive),
      make('fs_composite', format),
      // Only transparent hosts compile this up front; anyone else asking for it gets it lazily.
      transparent ? make('fs_composite_premultiplied', format) : null,
    ]);
    return new PostProcess(device, settings, uniforms, { down, downKaris, up, composite }, module, format, premultiplied, transparent);
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
    const input = opts.source ?? scene.texture;
    const doBloom = !raw && s.bloom > 0;
    const premultiplied = (opts.output ?? (this.transparent ? 'premultiplied' : 'opaque')) === 'premultiplied';

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
      .set('alphaMode', opts.alphaBlend === 'coverage' ? 0 : 1)
      .upload();

    if (levels.length > 0) {
      levels.forEach((dst, i) => {
        const src = i === 0 ? input : levels[i - 1].texture;
        this.fullscreen(encoder, i === 0 ? this.pipelines.downKaris : this.pipelines.down, src, dst.view, 'clear');
      });
      for (let i = levels.length - 1; i > 0; i--) {
        this.fullscreen(encoder, this.pipelines.up, levels[i].texture, levels[i - 1].view, 'load');
      }
    }

    const composite = premultiplied ? this.premultipliedPipeline() : this.pipelines.composite;
    this.fullscreen(encoder, composite, input, output, 'clear', levels[0]?.texture ?? this.black);
  }

  private premultipliedPipeline(): GPURenderPipeline {
    this.premultiplied ??= this.device.createRenderPipeline({
      label: 'post:fs_composite_premultiplied',
      layout: 'auto',
      vertex: { module: this.module, entryPoint: 'vs_fullscreen' },
      fragment: { module: this.module, entryPoint: 'fs_composite_premultiplied', targets: [{ format: this.format }] },
    });
    return this.premultiplied;
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
  /** Premultiplied output only: 0 = coverage, 1 = linear-light alpha. */
  alphaMode: 'u32',
} as const;

export interface PostContext {
  device: GPUDevice;
  format: GPUTextureFormat;
  params?: ParamSink;
  /** The output is a transparent (premultiplied-alpha) canvas: `render()` defaults to premultiplied output. */
  transparent?: boolean;
}

function addParams(ctx: PostContext, d: PostDefaults) {
  if (!ctx.params) {
    return {
      exposure: d.exposure ?? 1,
      tonemap: d.tonemap ?? TONEMAP.AgX,
      bloom: d.bloom ?? 0.04,
      bloomRadius: d.bloomRadius ?? 1,
      vignette: d.vignette ?? 0.25,
      grain: d.grain ?? 0.03,
    };
  }
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
