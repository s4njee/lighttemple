/**
 * Image-based lighting: bake an environment once, shade with it every frame.
 *
 *   const ibl = await Ibl.create(device);
 *   await ibl.setProcedural({ kind: 'studio' });   // or setEquirect(parseHdr(...))
 *   // pipeline layout: [yourGroup0, ibl.bindGroupLayout]
 *   // shader: wgsl(common, brdf, ibl.wgslStruct, iblWgsl, ...) then ibl_shade(...)
 *   pass.setBindGroup(1, ibl.bindGroup);
 *
 * Baking produces: an environment cube with a full mip chain (backgrounds,
 * filtered importance sampling), a GGX-prefiltered specular cube with one mip
 * per roughness step, 9 SH coefficients for diffuse irradiance (also read back
 * to the CPU to estimate the dominant light direction for shadows), and the
 * split-sum DFG LUT.
 */

import commonWgsl from '../shaders/common.wgsl?raw';
import brdfWgsl from '../shaders/brdf.wgsl?raw';
import { createShader, wgsl } from '../gpu/shader';
import { UniformBlock, uniformStruct } from '../gpu/uniforms';
import bakeWgsl from './bake.wgsl?raw';
import { uploadHalfFloatTexture, type HdrImage } from './hdr';
import iblWgsl from './ibl.wgsl?raw';

export { iblWgsl };

const ENV_FORMAT: GPUTextureFormat = 'rgba16float';
const ENV_SIZE = 512;
const SPEC_SIZE = 256;
const SPEC_MIPS = 7; // 256 .. 4, roughness 0, 1/6, .. 1
const LUT_SIZE = 128;
const PREFILTER_SAMPLES = 96;
const SH_SIZE = 32;

export type ProceduralEnvironment =
  | { kind: 'studio'; exposure?: number }
  | { kind: 'sky'; sunAzimuth: number; sunElevation: number; exposure?: number };

const IBL_PARAMS = {
  rotation: 'vec2f',
  intensity: 'f32',
  specMips: 'f32',
  envMips: 'f32',
} as const;

const BAKE_PARAMS = {
  sunDir: 'vec3f',
  exposure: 'f32',
  envSize: 'f32',
  mipCount: 'f32',
  sampleCount: 'u32',
  shSize: 'u32',
  shLod: 'f32',
} as const;

const mipCount = (size: number) => Math.floor(Math.log2(size)) + 1;

export class Ibl {
  /** Environment yaw in radians. */
  rotation = 0;
  /** Multiplier on all environment light. */
  intensity = 1;
  /** Brightest direction of the *unrotated* environment (from SH band 1). See `lightDirection()`. */
  dominantDirection: [number, number, number] = [0, 1, 0];
  /** Linear RGB irradiance coefficients (9 × rgb), CPU copy. */
  sh: Float32Array = new Float32Array(27);

  readonly bindGroupLayout: GPUBindGroupLayout;
  readonly wgslStruct: string;
  private group: GPUBindGroup | null = null;
  private env: GPUTexture | null = null;
  private spec: GPUTexture | null = null;
  private readonly lut: GPUTexture;
  private readonly shBuffer: GPUBuffer;
  private readonly params: UniformBlock<typeof IBL_PARAMS>;
  private readonly bake: UniformBlock<typeof BAKE_PARAMS>;
  private readonly linear: GPUSampler;
  private readonly repeatU: GPUSampler;
  private version = 0;

  private constructor(
    private readonly device: GPUDevice,
    private readonly pipelines: Record<'equirect' | 'studio' | 'sky' | 'down' | 'prefilter', GPURenderPipeline> & {
      sh: GPUComputePipeline;
      lut: GPURenderPipeline;
    },
  ) {
    this.params = new UniformBlock(device, 'IblParams', IBL_PARAMS);
    this.bake = new UniformBlock(device, 'Bake', BAKE_PARAMS);
    this.wgslStruct = this.params.wgslStruct;
    this.linear = device.createSampler({
      label: 'ibl-linear',
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });
    this.repeatU = device.createSampler({ label: 'ibl-equirect', magFilter: 'linear', minFilter: 'linear', addressModeU: 'repeat' });
    this.shBuffer = device.createBuffer({
      label: 'ibl-sh',
      size: 9 * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.lut = device.createTexture({
      label: 'ibl-dfg-lut',
      size: [LUT_SIZE, LUT_SIZE],
      format: 'rg16float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.bindGroupLayout = device.createBindGroupLayout({
      label: 'ibl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: 'cube' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: 'cube' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });

    const encoder = device.createCommandEncoder({ label: 'ibl-lut' });
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: this.lut.createView(), loadOp: 'clear', storeOp: 'store' }],
    });
    pass.setPipeline(pipelines.lut);
    pass.draw(3);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  static async create(device: GPUDevice): Promise<Ibl> {
    const code = wgsl(commonWgsl, brdfWgsl, uniformStruct('Bake', BAKE_PARAMS), bakeWgsl);
    const module = await createShader(device, 'ibl-bake', code);
    const frag = (entryPoint: string, format: GPUTextureFormat = ENV_FORMAT) =>
      device.createRenderPipelineAsync({
        label: `ibl:${entryPoint}`,
        layout: 'auto',
        vertex: { module, entryPoint: 'vs_bake' },
        fragment: { module, entryPoint, targets: [{ format }] },
      });
    const [equirect, studio, sky, down, prefilter, lut, sh] = await Promise.all([
      frag('fs_equirect'),
      frag('fs_studio'),
      frag('fs_sky'),
      frag('fs_downsample'),
      frag('fs_prefilter'),
      frag('fs_brdf_lut', 'rg16float'),
      device.createComputePipelineAsync({ label: 'ibl:cs_sh', layout: 'auto', compute: { module, entryPoint: 'cs_sh' } }),
    ]);
    return new Ibl(device, { equirect, studio, sky, down, prefilter, lut, sh });
  }

  /** The current bind group (changes whenever a new environment is baked). */
  get bindGroup(): GPUBindGroup {
    if (!this.group) throw new Error('Ibl: no environment baked yet');
    return this.group;
  }

  /** Bumps every time a new environment finishes baking. */
  get environmentVersion(): number {
    return this.version;
  }

  /** World-space direction *towards* the environment's dominant light, honouring `rotation`. */
  lightDirection(): [number, number, number] {
    // Inverse of ibl_rotate() in ibl.wgsl.
    const [x, y, z] = this.dominantDirection;
    const c = Math.cos(this.rotation);
    const s = Math.sin(this.rotation);
    return [c * x - s * z, y, s * x + c * z];
  }

  /** Uploads rotation/intensity; call once per frame before drawing. */
  update(): void {
    this.params
      .set('rotation', [Math.cos(this.rotation), Math.sin(this.rotation)])
      .set('intensity', this.intensity)
      .set('specMips', SPEC_MIPS)
      .set('envMips', mipCount(ENV_SIZE))
      .upload();
  }

  async setEquirect(image: HdrImage): Promise<void> {
    const source = uploadHalfFloatTexture(this.device, image, 'ibl-equirect');
    const group = this.device.createBindGroup({
      layout: this.pipelines.equirect.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: source.createView() },
        { binding: 1, resource: this.repeatU },
      ],
    });
    try {
      await this.bakeFrom(this.pipelines.equirect, group);
    } finally {
      source.destroy();
    }
  }

  async setProcedural(env: ProceduralEnvironment): Promise<void> {
    const sun =
      env.kind === 'sky'
        ? [Math.cos(env.sunElevation) * Math.sin(env.sunAzimuth), Math.sin(env.sunElevation), Math.cos(env.sunElevation) * Math.cos(env.sunAzimuth)]
        : [0, 1, 0];
    this.bake.set('sunDir', sun).set('exposure', env.exposure ?? 1).upload();
    const pipeline = env.kind === 'sky' ? this.pipelines.sky : this.pipelines.studio;
    const group = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 2, resource: { buffer: this.bake.buffer } }],
    });
    await this.bakeFrom(pipeline, group);
  }

  private async bakeFrom(source: GPURenderPipeline, sourceGroup: GPUBindGroup): Promise<void> {
    const d = this.device;
    const envMips = mipCount(ENV_SIZE);
    const env = d.createTexture({
      label: 'ibl-env',
      size: [ENV_SIZE, ENV_SIZE, 6],
      mipLevelCount: envMips,
      format: ENV_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    const spec = d.createTexture({
      label: 'ibl-specular',
      size: [SPEC_SIZE, SPEC_SIZE, 6],
      mipLevelCount: SPEC_MIPS,
      format: ENV_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.bake
      .set('envSize', ENV_SIZE)
      .set('mipCount', SPEC_MIPS)
      .set('sampleCount', PREFILTER_SAMPLES)
      .set('shSize', SH_SIZE)
      .set('shLod', Math.log2(ENV_SIZE / SH_SIZE))
      .upload();

    const encoder = d.createCommandEncoder({ label: 'ibl-bake' });
    const faceView = (tex: GPUTexture, face: number, mip: number) =>
      tex.createView({ dimension: '2d', baseArrayLayer: face, arrayLayerCount: 1, baseMipLevel: mip, mipLevelCount: 1 });
    const draw = (pipeline: GPURenderPipeline, group: GPUBindGroup, target: GPUTextureView, instance: number) => {
      const pass = encoder.beginRenderPass({ colorAttachments: [{ view: target, loadOp: 'clear', storeOp: 'store' }] });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.draw(3, 1, 0, instance);
      pass.end();
    };

    // 1. Source -> env mip 0.
    for (let f = 0; f < 6; f++) draw(source, sourceGroup, faceView(env, f, 0), f);

    // 2. Env mip chain: each level samples only the previous one.
    for (let m = 1; m < envMips; m++) {
      const group = d.createBindGroup({
        layout: this.pipelines.down.getBindGroupLayout(0),
        entries: [
          { binding: 1, resource: this.linear },
          { binding: 3, resource: env.createView({ dimension: 'cube', baseMipLevel: m - 1, mipLevelCount: 1 }) },
        ],
      });
      for (let f = 0; f < 6; f++) draw(this.pipelines.down, group, faceView(env, f, m), f);
    }

    // 3. Specular prefilter (reads the whole env chain).
    const envCube = env.createView({ dimension: 'cube' });
    const prefilterGroup = d.createBindGroup({
      layout: this.pipelines.prefilter.getBindGroupLayout(0),
      entries: [
        { binding: 1, resource: this.linear },
        { binding: 2, resource: { buffer: this.bake.buffer } },
        { binding: 3, resource: envCube },
      ],
    });
    for (let m = 0; m < SPEC_MIPS; m++) {
      for (let f = 0; f < 6; f++) draw(this.pipelines.prefilter, prefilterGroup, faceView(spec, f, m), m * 6 + f);
    }

    // 4. SH9 irradiance.
    const shGroup = d.createBindGroup({
      layout: this.pipelines.sh.getBindGroupLayout(0),
      entries: [
        { binding: 1, resource: this.linear },
        { binding: 2, resource: { buffer: this.bake.buffer } },
        { binding: 3, resource: envCube },
        { binding: 4, resource: { buffer: this.shBuffer } },
      ],
    });
    const cp = encoder.beginComputePass({ label: 'ibl-sh' });
    cp.setPipeline(this.pipelines.sh);
    cp.setBindGroup(0, shGroup);
    cp.dispatchWorkgroups(1);
    cp.end();

    const readback = d.createBuffer({ size: 9 * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    encoder.copyBufferToBuffer(this.shBuffer, 0, readback, 0, 9 * 16);
    d.queue.submit([encoder.finish()]);

    this.env?.destroy();
    this.spec?.destroy();
    this.env = env;
    this.spec = spec;
    this.group = d.createBindGroup({
      label: 'ibl',
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: envCube },
        { binding: 1, resource: spec.createView({ dimension: 'cube' }) },
        { binding: 2, resource: this.lut.createView() },
        { binding: 3, resource: this.linear },
        { binding: 4, resource: { buffer: this.shBuffer } },
        { binding: 5, resource: { buffer: this.params.buffer } },
      ],
    });
    this.version++;

    await readback.mapAsync(GPUMapMode.READ);
    const raw = new Float32Array(readback.getMappedRange().slice(0));
    readback.destroy();
    for (let k = 0; k < 9; k++) this.sh.set(raw.subarray(k * 4, k * 4 + 3), k * 3);
    // Band-1 coefficients are (y, z, x) in the basis order used by the shader.
    const lum = (k: number) => 0.2126 * this.sh[k * 3] + 0.7152 * this.sh[k * 3 + 1] + 0.0722 * this.sh[k * 3 + 2];
    const dir = [lum(3), lum(1), lum(2)];
    const len = Math.hypot(...dir) || 1;
    this.dominantDirection = [dir[0] / len, dir[1] / len, dir[2] / len];
  }

  destroy(): void {
    this.env?.destroy();
    this.spec?.destroy();
    this.lut.destroy();
    this.shBuffer.destroy();
    this.params.destroy();
    this.bake.destroy();
  }
}
