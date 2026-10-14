import { mat4, vec3 } from 'wgpu-matrix';
import brdfWgsl from '../../shaders/brdf.wgsl?raw';
import commonWgsl from '../../shaders/common.wgsl?raw';
import fullscreenWgsl from '../../shaders/fullscreen.wgsl?raw';
import shadowWgsl from '../../shaders/shadow.wgsl?raw';
import { hexToLinear } from '../../gpu/color';
import { MESH_VERTEX_LAYOUT, MESH_VERTEX_WGSL, box, plane, sphere, torus, uploadMesh, type Mesh } from '../../gpu/geometry';
import { createShader, wgsl } from '../../gpu/shader';
import { TargetPool } from '../../gpu/targets';
import { UniformBlock, uniformStruct } from '../../gpu/uniforms';
import { HDR_FORMAT, PostProcess } from '../../post/post';
import { fitDirectionalShadow } from '../../scene/light-frustum';
import { OrbitCamera } from '../../scene/orbit-camera';
import { PROP_WGSL, PropSet, type Prop } from '../../scene/props';
import type { ExperimentFactory } from '../types';
import shaderWgsl from './shader.wgsl?raw';

const MSAA = 4;
const DEPTH_FORMAT: GPUTextureFormat = 'depth24plus';
const SHADOW_FORMAT: GPUTextureFormat = 'depth32float';
const REALTIME_MAP_SIZE = 2048;
const BAKE_MAP_SIZE = 1024;
/** Half-size of the baked ground region (world units, centred on the origin). */
const BAKE_EXTENT = 6;
/** Bake samples encoded per frame (the bake is progressive so it never stalls a frame). */
const BAKE_PER_FRAME = 16;
const BAKE_STRIDE = 256; // dynamic-offset alignment
const BAKE_FIELDS = { viewProj: 'mat4x4f', kind: 'vec4f', extent: 'f32' } as const;
const BLOB_FLOATS = 8;

const MODE = { Blob: 0, Baked: 1, Realtime: 2 } as const;

/** What a blob needs to know about an object: where it is, its footprint and its vertical span. */
interface BlobSource {
  /** Ground position (x, z) of the object's centre at time t. */
  at: (t: number) => [number, number];
  /** Footprint radius on the ground. */
  radius: number;
  /** Lowest and highest point above the ground at time t. */
  span: (t: number) => [number, number];
}

interface SceneObject extends Prop {
  blob: BlobSource;
}

const create: ExperimentFactory = async (ctx) => {
  const { device, surface, params, reducedMotion } = ctx;

  const look = params.add(
    {
      mode: {
        value: MODE.Baked,
        options: { 'Blob decals': MODE.Blob, 'Baked + blobs': MODE.Baked, 'Real-time shadow map': MODE.Realtime },
        hint: 'Blob: one instanced quad per object, no shadow pass, no memory. Baked: one lightmap lookup per ground pixel (512² rgba16f = 2 MB), no per-frame shadow work; moving objects fall back to blobs. Real-time: a 2048² depth pass every frame (16 MB) plus 16 compare taps per pixel.',
      },
      animate: { value: !reducedMotion, label: 'Animate', hint: 'The bouncing ball and the orbiting orb are "dynamic": never baked.' },
      view: { value: 0, options: { Lit: 0, 'Shadow term': 1 }, hint: 'Shadow term: white ground darkened only by the active technique.' },
      showLightmap: { value: false, label: 'Show lightmap', hint: 'Corner overlay. Red = baked sun visibility, cyan = baked sky visibility (AO).' },
      background: { value: '#d8d0c4', type: 'color', hint: 'Page colour the floor fades into.' },
    },
    'Technique',
  );
  const sun = params.add(
    {
      azimuth: { value: -112, min: -180, max: 180, step: 1 },
      elevation: { value: 42, min: 10, max: 89, step: 1, hint: 'Changing the sun restarts the bake (progressive, a few frames).' },
      intensity: { value: 3.2, min: 0, max: 8, step: 0.05 },
      color: { value: '#fff0dc', type: 'color' },
      ambient: { value: 0.45, min: 0, max: 1.5, step: 0.01, hint: 'Sky light. In baked mode it is multiplied by the baked sky visibility, i.e. ambient occlusion.' },
    },
    'Sun',
  );
  const blob = params.add(
    {
      blobOpacity: { value: 0.75, min: 0, max: 1, step: 0.01, label: 'Blob opacity' },
      blobSoftness: { value: 0.55, min: 0.05, max: 1, step: 0.01, label: 'Blob softness', hint: 'Fraction of the radius that fades out. Grows with height.' },
      orient: { value: 1, min: 0, max: 1, step: 0.01, label: 'Follow light', hint: '0: round blobs straight below. 1: offset and stretched along the light by each object\'s height.' },
      fadeHeight: { value: 2.6, min: 0.5, max: 6, step: 0.05, label: 'Fade height', hint: 'Height above the ground at which a blob has faded out entirely.' },
    },
    'Blob decals',
  );
  const baking = params.add(
    {
      samples: { value: 128, options: { '32': 32, '64': 64, '128': 128, '256': 256 }, label: 'Bake samples', hint: 'Half are jittered sun directions (soft shadows), half cosine-distributed sky directions (AO). Each is one depth pass + one accumulate pass, once.' },
      lightmapSize: { value: 512, options: { '256': 256, '512': 512, '1024': 1024 }, label: 'Lightmap size' },
      sunSize: { value: 4, min: 0, max: 15, step: 0.1, label: 'Sun size (°)', hint: 'Angular diameter the bake jitters the sun over. Free for the bake; expensive in real time.' },
    },
    'Bake',
  );
  const post = await PostProcess.create(ctx, { bloom: 0.02, vignette: 0.12, grain: 0.02 });

  // --- Scene -----------------------------------------------------------------------------
  const meshes: Record<string, Mesh> = {
    ground: uploadMesh(device, plane(60, 1), 'ground'),
    pillar: uploadMesh(device, box(0.38, 1.3, 0.38), 'pillar'),
    lintel: uploadMesh(device, box(1.7, 0.32, 0.5), 'lintel'),
    cube: uploadMesh(device, box(0.9, 0.9, 0.9), 'cube'),
    small: uploadMesh(device, box(0.6, 0.6, 0.6), 'small-cube'),
    trunk: uploadMesh(device, box(0.16, 0.9, 0.16), 'trunk'),
    sphere: uploadMesh(device, sphere(1, 48, 32), 'sphere'),
    ring: uploadMesh(device, torus(0.5, 0.16, 24, 72), 'ring'),
  };
  const T = (x: number, y: number, z: number, yaw = 0) => mat4.rotateY(mat4.translation([x, y, z]), yaw);
  const ball = (x: number, y: number, z: number, r: number) => mat4.scale(mat4.translation([x, y, z]), [r, r, r]);
  const still = (x: number, z: number) => () => [x, z] as [number, number];
  const fixed = (lo: number, hi: number) => () => [lo, hi] as [number, number];

  const statics: SceneObject[] = [
    // An arch: the blob under the lintel is the classic failure (a blob cannot have a hole).
    { mesh: meshes.pillar, color: '#e8dccb', roughness: 0.8, transform: () => T(-2.8, 0.65, -0.6), blob: { at: still(-2.8, -0.6), radius: 0.3, span: fixed(0, 1.3) } },
    { mesh: meshes.pillar, color: '#e8dccb', roughness: 0.8, transform: () => T(-1.6, 0.65, -0.6), blob: { at: still(-1.6, -0.6), radius: 0.3, span: fixed(0, 1.3) } },
    { mesh: meshes.lintel, color: '#e98a6b', roughness: 0.6, transform: () => T(-2.2, 1.46, -0.6), blob: { at: still(-2.2, -0.6), radius: 0.75, span: fixed(1.3, 1.62) } },
    // A stack of blocks.
    { mesh: meshes.cube, color: '#6aa5a0', roughness: 0.55, transform: () => T(1.5, 0.45, -1.3, 0.3), blob: { at: still(1.5, -1.3), radius: 0.62, span: fixed(0, 0.9) } },
    { mesh: meshes.small, color: '#f2c14e', roughness: 0.5, transform: () => T(1.55, 1.2, -1.25, 0.95), blob: { at: still(1.55, -1.25), radius: 0.42, span: fixed(0.9, 1.5) } },
    // A ball resting on the floor and a ring lying flat.
    { mesh: meshes.sphere, color: '#9b8ec4', roughness: 0.35, transform: () => ball(-0.6, 0.55, 1.0, 0.55), blob: { at: still(-0.6, 1.0), radius: 0.5, span: fixed(0, 1.1) } },
    { mesh: meshes.ring, color: '#f4efe6', roughness: 0.4, transform: () => T(-2.4, 0.16, 2.0), blob: { at: still(-2.4, 2.0), radius: 0.62, span: fixed(0, 0.32) } },
    // A tree: the canopy's blob drifts away from the trunk with the sun, as the real shadow does.
    { mesh: meshes.trunk, color: '#8a6a52', roughness: 0.8, transform: () => T(2.4, 0.45, 1.4), blob: { at: still(2.4, 1.4), radius: 0.12, span: fixed(0, 0.9) } },
    { mesh: meshes.sphere, color: '#8fb573', roughness: 0.7, transform: () => ball(2.4, 1.4, 1.4, 0.62), blob: { at: still(2.4, 1.4), radius: 0.6, span: fixed(0.78, 2.02) } },
  ];
  const bounce = (t: number) => 0.32 + Math.abs(Math.sin(t * 2.4)) * 1.5;
  const orbAt = (t: number): [number, number] => [Math.cos(t * 0.5) * 3.4, Math.sin(t * 0.5) * 3.4];
  const orbY = (t: number) => 1.7 + 0.35 * Math.sin(t * 1.3);
  const dynamics: SceneObject[] = [
    { mesh: meshes.sphere, color: '#e85d75', roughness: 0.3, transform: (t) => ball(0.4, bounce(t), -0.1, 0.32), blob: { at: still(0.4, -0.1), radius: 0.3, span: (t) => [bounce(t) - 0.32, bounce(t) + 0.32] } },
    {
      mesh: meshes.sphere,
      color: '#fff5e0',
      roughness: 0.2,
      metallic: 1,
      transform: (t) => ball(orbAt(t)[0], orbY(t), orbAt(t)[1], 0.22),
      blob: { at: orbAt, radius: 0.22, span: (t) => [orbY(t) - 0.22, orbY(t) + 0.22] },
    },
  ];
  const staticSet = new PropSet(device, statics, 'blob-statics');
  const dynamicSet = new PropSet(device, dynamics, 'blob-dynamics');
  staticSet.update(0);
  const blobSources = [...statics, ...dynamics].map((o) => o.blob);
  const blobData = new Float32Array(blobSources.length * BLOB_FLOATS);
  const blobBuffer = device.createBuffer({ label: 'blobs', size: blobData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

  const frame = new UniformBlock(device, 'Frame', {
    viewProj: 'mat4x4f',
    lightViewProj: 'mat4x4f',
    cameraPos: 'vec3f',
    mode: 'u32',
    toLight: 'vec3f',
    debugView: 'u32',
    sunColor: 'vec3f',
    fadeStart: 'f32',
    skyColor: 'vec3f',
    fadeEnd: 'f32',
    groundBounce: 'vec3f',
    bakeExtent: 'f32',
    groundColor: 'vec3f',
    shadowTexelWorld: 'f32',
  });

  const module = await createShader(
    device,
    'blob-shadows',
    wgsl(commonWgsl, fullscreenWgsl, MESH_VERTEX_WGSL, PROP_WGSL, brdfWgsl, frame.wgslStruct, uniformStruct('Bake', BAKE_FIELDS), shadowWgsl, shaderWgsl),
  );

  // --- Layouts & pipelines -------------------------------------------------------------
  const VF = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
  const mainLayout = device.createBindGroupLayout({
    label: 'blob-main',
    entries: [
      { binding: 0, visibility: VF, buffer: { type: 'uniform' } },
      { binding: 1, visibility: VF, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      { binding: 6, visibility: VF, buffer: { type: 'read-only-storage' } },
    ],
  });
  const shadowLayout = device.createBindGroupLayout({
    label: 'blob-shadow',
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
    ],
  });
  const bakeObjectsLayout = device.createBindGroupLayout({
    label: 'bake-objects',
    entries: [{ binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }],
  });
  const bakeUniform: GPUBindGroupLayoutEntry = {
    binding: 0,
    visibility: VF,
    buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 96 },
  };
  const bakeDepthLayout = device.createBindGroupLayout({ label: 'bake-depth', entries: [bakeUniform] });
  const bakeAccumLayout = device.createBindGroupLayout({
    label: 'bake-accumulate',
    entries: [
      bakeUniform,
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
    ],
  });
  const emptyLayout = device.createBindGroupLayout({ label: 'empty', entries: [] });
  const layout = (...g: GPUBindGroupLayout[]) => device.createPipelineLayout({ bindGroupLayouts: g });

  const msaaDepth = (write: boolean, compare: GPUCompareFunction = 'less'): GPUDepthStencilState => ({
    format: DEPTH_FORMAT,
    depthWriteEnabled: write,
    depthCompare: compare,
  });
  const over: GPUBlendState = {
    color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
    alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
  };
  const add: GPUBlendState = {
    color: { srcFactor: 'one', dstFactor: 'one' },
    alpha: { srcFactor: 'one', dstFactor: 'one' },
  };
  const [shadowPipeline, bakeDepthPipeline, bakeAccumPipeline, objectPipeline, groundPipeline, blobPipeline, overlayPipeline] = await Promise.all([
    device.createRenderPipelineAsync({
      label: 'blob:shadow',
      layout: layout(shadowLayout),
      vertex: { module, entryPoint: 'vs_shadow', buffers: [MESH_VERTEX_LAYOUT] },
      primitive: { cullMode: 'back' },
      depthStencil: { format: SHADOW_FORMAT, depthWriteEnabled: true, depthCompare: 'less' },
    }),
    device.createRenderPipelineAsync({
      label: 'blob:bake-depth',
      layout: layout(bakeObjectsLayout, bakeDepthLayout),
      vertex: { module, entryPoint: 'vs_bake_depth', buffers: [MESH_VERTEX_LAYOUT] },
      primitive: { cullMode: 'none' },
      depthStencil: { format: SHADOW_FORMAT, depthWriteEnabled: true, depthCompare: 'less' },
    }),
    device.createRenderPipelineAsync({
      label: 'blob:bake-accumulate',
      layout: layout(emptyLayout, bakeAccumLayout),
      vertex: { module, entryPoint: 'vs_fullscreen' },
      fragment: { module, entryPoint: 'fs_bake_accumulate', targets: [{ format: 'rgba16float', blend: add }] },
    }),
    device.createRenderPipelineAsync({
      label: 'blob:object',
      layout: layout(mainLayout),
      vertex: { module, entryPoint: 'vs_object', buffers: [MESH_VERTEX_LAYOUT] },
      fragment: { module, entryPoint: 'fs_object', targets: [{ format: HDR_FORMAT }] },
      primitive: { cullMode: 'back' },
      depthStencil: msaaDepth(true),
      multisample: { count: MSAA },
    }),
    device.createRenderPipelineAsync({
      label: 'blob:ground',
      layout: layout(mainLayout),
      vertex: { module, entryPoint: 'vs_ground', buffers: [MESH_VERTEX_LAYOUT] },
      fragment: { module, entryPoint: 'fs_ground', targets: [{ format: HDR_FORMAT }] },
      primitive: { cullMode: 'back' },
      depthStencil: msaaDepth(true),
      multisample: { count: MSAA },
    }),
    device.createRenderPipelineAsync({
      label: 'blob:decals',
      layout: layout(mainLayout),
      vertex: { module, entryPoint: 'vs_blob' },
      fragment: { module, entryPoint: 'fs_blob', targets: [{ format: HDR_FORMAT, blend: over }] },
      depthStencil: msaaDepth(false, 'less-equal'),
      multisample: { count: MSAA },
    }),
    device.createRenderPipelineAsync({
      label: 'blob:overlay',
      layout: layout(mainLayout),
      vertex: { module, entryPoint: 'vs_fullscreen' },
      fragment: { module, entryPoint: 'fs_overlay', targets: [{ format: HDR_FORMAT }] },
      depthStencil: msaaDepth(false, 'always'),
      multisample: { count: MSAA },
    }),
  ]);

  // --- Resources -------------------------------------------------------------------------
  const cmp = device.createSampler({ compare: 'less-equal', magFilter: 'linear', minFilter: 'linear' });
  const linear = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  const depthTexture = (size: number, label: string) =>
    device.createTexture({ label, size: [size, size], format: SHADOW_FORMAT, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
  const targets = new TargetPool(device);
  const bakeBuffer = device.createBuffer({ label: 'bake-samples', size: 256 * BAKE_STRIDE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const bakeObjectsGroup = device.createBindGroup({ layout: bakeObjectsLayout, entries: [{ binding: 1, resource: { buffer: staticSet.buffer } }] });
  const bakeDepthGroup = device.createBindGroup({ layout: bakeDepthLayout, entries: [{ binding: 0, resource: { buffer: bakeBuffer, size: 96 } }] });
  const emptyGroup = device.createBindGroup({ layout: emptyLayout, entries: [] });
  const shadowGroups = [staticSet, dynamicSet].map((set) =>
    device.createBindGroup({
      layout: shadowLayout,
      entries: [
        { binding: 0, resource: { buffer: frame.buffer } },
        { binding: 1, resource: { buffer: set.buffer } },
      ],
    }),
  );

  // The real-time map only exists while the real-time mode is on (16 MB at 2048²); the bake's depth
  // map only while a bake is running. Both are swapped for a 1×1 stand-in otherwise.
  let shadowMap = depthTexture(1, 'shadow-map-idle');
  let lightmap: GPUTexture | null = null;
  let mainGroups: GPUBindGroup[] = [];
  const rebuildMainGroups = () => {
    mainGroups = [staticSet, dynamicSet].map((set) =>
      device.createBindGroup({
        layout: mainLayout,
        entries: [
          { binding: 0, resource: { buffer: frame.buffer } },
          { binding: 1, resource: { buffer: set.buffer } },
          { binding: 2, resource: shadowMap.createView() },
          { binding: 3, resource: cmp },
          { binding: 4, resource: lightmap!.createView() },
          { binding: 5, resource: linear },
          { binding: 6, resource: { buffer: blobBuffer } },
        ],
      }),
    );
  };
  const ensureShadowMap = (size: number) => {
    if (shadowMap.width === size) return;
    shadowMap.destroy();
    shadowMap = depthTexture(size, size > 1 ? 'shadow-map' : 'shadow-map-idle');
    rebuildMainGroups();
  };

  // --- Bake state --------------------------------------------------------------------------
  let bakeKey = '';
  let bakeDone = 0;
  let bakeTotal = 0;
  let bakeDepth: GPUTexture | null = null;
  let bakeAccumGroup: GPUBindGroup | null = null;
  const bakeData = new Float32Array((256 * BAKE_STRIDE) / 4);

  /** Interleaved sun / sky sample directions (unit vectors toward the light). */
  const bakeDirections = (count: number, toSun: Float32Array, sunSizeDeg: number): Float32Array[] => {
    const dirs: Float32Array[] = [];
    const up = Math.abs(toSun[1]) > 0.99 ? vec3.fromValues(1, 0, 0) : vec3.fromValues(0, 1, 0);
    const t1 = vec3.normalize(vec3.cross(toSun, up));
    const t2 = vec3.cross(toSun, t1);
    const cone = Math.tan(((sunSizeDeg / 2) * Math.PI) / 180);
    const half = count / 2;
    for (let i = 0; i < half; i++) {
      // Sun: a Vogel disc across the sun's angular radius.
      const r = Math.sqrt((i + 0.5) / half) * cone;
      const a = i * 2.39996323;
      dirs.push(vec3.normalize(vec3.add(toSun, vec3.add(vec3.scale(t1, r * Math.cos(a)), vec3.scale(t2, r * Math.sin(a))))) as Float32Array);
      // Sky: cosine-weighted hemisphere (Hammersley), kept above ~8° where ortho maps get stretched.
      const u = Math.min((i + 0.5) / half, Math.cos((8 * Math.PI) / 180) ** 2);
      const phi = i * 2.39996323 + 0.7;
      const sinT = Math.sqrt(u);
      dirs.push(vec3.fromValues(sinT * Math.cos(phi), Math.sqrt(1 - u), sinT * Math.sin(phi)) as Float32Array);
    }
    return dirs;
  };

  const startBake = (toSun: Float32Array) => {
    bakeTotal = baking.samples;
    bakeDone = 0;
    const lm = targets.get('lightmap', { size: [baking.lightmapSize, baking.lightmapSize], format: 'rgba16float' });
    if (lm.texture !== lightmap) {
      lightmap = lm.texture;
      rebuildMainGroups();
    }
    bakeDepth ??= depthTexture(BAKE_MAP_SIZE, 'bake-depth');
    bakeAccumGroup = device.createBindGroup({
      layout: bakeAccumLayout,
      entries: [
        { binding: 0, resource: { buffer: bakeBuffer, size: 96 } },
        { binding: 1, resource: bakeDepth.createView() },
        { binding: 2, resource: cmp },
      ],
    });
    // One ortho map per direction, fitted around the whole baked square so far-reaching grazing
    // shadows still land inside it.
    const bounds = { center: vec3.fromValues(0, 0, 0), radius: BAKE_EXTENT * Math.SQRT2 };
    bakeDirections(bakeTotal, toSun, baking.sunSize).forEach((dir, i) => {
      const fit = fitDirectionalShadow({ toLight: dir, bounds, mapSize: BAKE_MAP_SIZE, snap: false, casterPadding: 4 });
      const o = (i * BAKE_STRIDE) / 4;
      bakeData.set(fit.viewProj, o);
      bakeData.set(i % 2 === 0 ? [1, 0, 0, 0] : [0, 1, 0, 0], o + 16);
      bakeData[o + 20] = BAKE_EXTENT;
    });
    device.queue.writeBuffer(bakeBuffer, 0, bakeData, 0, (bakeTotal * BAKE_STRIDE) / 4);
  };

  const encodeBake = (encoder: GPUCommandEncoder) => {
    const end = Math.min(bakeTotal, bakeDone + BAKE_PER_FRAME);
    for (let i = bakeDone; i < end; i++) {
      const offset = [i * BAKE_STRIDE];
      const dp = encoder.beginRenderPass({
        label: 'bake-depth',
        colorAttachments: [],
        depthStencilAttachment: { view: bakeDepth!.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
      });
      dp.setPipeline(bakeDepthPipeline);
      dp.setBindGroup(0, bakeObjectsGroup);
      dp.setBindGroup(1, bakeDepthGroup, offset);
      staticSet.draw(dp);
      dp.end();
      const ap = encoder.beginRenderPass({
        label: 'bake-accumulate',
        colorAttachments: [{ view: lightmap!.createView(), loadOp: i === 0 ? 'clear' : 'load', storeOp: 'store', clearValue: [0, 0, 0, 0] }],
      });
      ap.setPipeline(bakeAccumPipeline);
      ap.setBindGroup(0, emptyGroup);
      ap.setBindGroup(1, bakeAccumGroup!, offset);
      ap.draw(3);
      ap.end();
    }
    bakeDone = end;
  };

  const camera = new OrbitCamera(surface.canvas, { target: [0, 0.6, 0.2], distance: 10.5, yaw: 0.35, pitch: 0.55, maxDistance: 25 });
  let animTime = 1.1;

  return {
    frame({ time, dt, size }) {
      if (look.animate) animTime += dt;
      camera.update(dt, size.width / size.height);
      const mode = look.mode;

      const az = (sun.azimuth * Math.PI) / 180;
      const el = (sun.elevation * Math.PI) / 180;
      const toSun = vec3.fromValues(Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az)) as Float32Array;

      // (Re)start the bake whenever what it depends on changes; it then runs a few frames.
      const key = `${sun.azimuth}:${sun.elevation}:${baking.sunSize}:${baking.samples}:${baking.lightmapSize}`;
      if (key !== bakeKey) {
        bakeKey = key;
        startBake(toSun);
      }
      ensureShadowMap(mode === MODE.Realtime ? REALTIME_MAP_SIZE : 1);

      dynamicSet.update(animTime);

      // Blobs: offset along the light by the object's height, stretched by its vertical extent,
      // larger, softer and fainter the higher the object floats.
      const dirLen = Math.hypot(toSun[0], toSun[2]);
      const axis = dirLen > 1e-3 ? [-toSun[0] / dirLen, -toSun[2] / dirLen] : [1, 0];
      const cot = Math.min(dirLen / Math.max(toSun[1], 1e-3), 3) * blob.orient;
      blobSources.forEach((b, i) => {
        const [x, z] = b.at(animTime);
        const [lo, hi] = b.span(animTime);
        const grow = 1 + lo * 0.35;
        const shift = 0.5 * (lo + hi) * cot;
        const along = (b.radius + 0.5 * (hi - lo) * cot) * grow;
        const across = b.radius * grow;
        const fade = Math.max(0, 1 - lo / blob.fadeHeight);
        const soft = Math.min(1, blob.blobSoftness + lo * 0.2);
        blobData.set([x + axis[0] * shift, z + axis[1] * shift, axis[0], axis[1], along, across, blob.blobOpacity * fade, soft], i * BLOB_FLOATS);
      });
      device.queue.writeBuffer(blobBuffer, 0, blobData);

      const fit = fitDirectionalShadow({ toLight: toSun, bounds: { center: vec3.fromValues(0, 0.8, 0.2), radius: 4.6 }, mapSize: REALTIME_MAP_SIZE });
      const lc = hexToLinear(sun.color);
      const ground = hexToLinear('#e6dccf');
      frame
        .set('viewProj', camera.viewProjection)
        .set('lightViewProj', fit.viewProj)
        .set('cameraPos', camera.position)
        .set('mode', mode)
        .set('toLight', toSun)
        .set('debugView', look.view)
        .set('sunColor', lc.map((c) => c * sun.intensity))
        .set('skyColor', [0.62 * sun.ambient, 0.7 * sun.ambient, 0.85 * sun.ambient])
        .set('groundBounce', ground.map((c) => c * sun.ambient * 0.5))
        .set('groundColor', ground)
        .set('fadeStart', 6)
        .set('fadeEnd', 16)
        .set('bakeExtent', BAKE_EXTENT)
        .set('shadowTexelWorld', fit.texelWorld)
        .upload();

      const encoder = device.createCommandEncoder({ label: 'blob-shadows' });
      const bakeRunning = bakeDone < bakeTotal;
      if (bakeRunning) encodeBake(encoder);

      if (mode === MODE.Realtime) {
        const sp = encoder.beginRenderPass({
          label: 'shadow-pass',
          colorAttachments: [],
          depthStencilAttachment: { view: shadowMap.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
        });
        sp.setPipeline(shadowPipeline);
        [staticSet, dynamicSet].forEach((set, i) => {
          sp.setBindGroup(0, shadowGroups[i]);
          set.draw(sp);
        });
        sp.end();
      }

      const full = [size.width, size.height] as const;
      const color = targets.get('msaa', { size: full, format: HDR_FORMAT, sampleCount: MSAA });
      const depth = targets.get('depth', { size: full, format: DEPTH_FORMAT, sampleCount: MSAA });
      const scene = post.scene(size);
      const pass = encoder.beginRenderPass({
        label: 'main-pass',
        colorAttachments: [{ view: color.view, resolveTarget: scene.view, loadOp: 'clear', storeOp: 'discard', clearValue: [0, 0, 0, 0] }],
        depthStencilAttachment: { view: depth.view, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard' },
      });
      pass.setBindGroup(0, mainGroups[0]);
      pass.setPipeline(groundPipeline);
      pass.setVertexBuffer(0, meshes.ground.vertexBuffer);
      pass.setIndexBuffer(meshes.ground.indexBuffer, 'uint32');
      pass.drawIndexed(meshes.ground.indexCount);
      pass.setPipeline(objectPipeline);
      [staticSet, dynamicSet].forEach((set, i) => {
        pass.setBindGroup(0, mainGroups[i]);
        set.draw(pass);
      });
      // Decals: every object in blob mode, only the dynamic ones on top of the bake.
      if (mode !== MODE.Realtime) {
        const first = mode === MODE.Baked ? statics.length : 0;
        pass.setBindGroup(0, mainGroups[0]);
        pass.setPipeline(blobPipeline);
        pass.draw(6, blobSources.length - first, 0, first);
      }
      if (look.showLightmap) {
        const s = Math.round(Math.min(size.width, size.height) * 0.3);
        const m = Math.round(16 * size.pixelRatio);
        pass.setViewport(m, size.height - s - m, s, s, 0, 1);
        pass.setPipeline(overlayPipeline);
        pass.draw(3);
      }
      pass.end();

      post.render(encoder, surface.currentTexture().createView(), { time, raw: look.view !== 0, background: hexToLinear(look.background) });
      device.queue.submit([encoder.finish()]);
      if (bakeRunning && bakeDone === bakeTotal) {
        // Bake finished: only the lightmap is kept (destroy waits for the submitted work).
        bakeDepth?.destroy();
        bakeDepth = null;
        bakeAccumGroup = null;
      }
    },
    dispose() {
      camera.destroy();
      shadowMap.destroy();
      bakeDepth?.destroy();
      bakeBuffer.destroy();
      blobBuffer.destroy();
      staticSet.destroy();
      dynamicSet.destroy();
      Object.values(meshes).forEach((m) => m.destroy());
      targets.destroy();
      frame.destroy();
      post.destroy();
    },
  };
};

export default create;
