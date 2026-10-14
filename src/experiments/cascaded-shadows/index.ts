import { mat4, vec3, type Mat4 } from 'wgpu-matrix';
import commonWgsl from '../../shaders/common.wgsl?raw';
import noiseWgsl from '../../shaders/noise.wgsl?raw';
import fullscreenWgsl from '../../shaders/fullscreen.wgsl?raw';
import brdfWgsl from '../../shaders/brdf.wgsl?raw';
import { MESH_VERTEX_LAYOUT, MESH_VERTEX_WGSL, uploadMesh, type Mesh } from '../../gpu/geometry';
import { createShader, wgsl } from '../../gpu/shader';
import { TargetPool } from '../../gpu/targets';
import { HDR_FORMAT, PostProcess } from '../../post/post';
import { TemporalAA } from '../../post/taa';
import { UniformBlock } from '../../gpu/uniforms';
import { OrbitCamera } from '../../scene/orbit-camera';
import { PROP_WGSL } from '../../scene/props';
import { hexToLinear } from '../../gpu/color';
import type { ExperimentFactory } from '../types';
import { fitCascades } from './cascades';
import { FLY_RADIUS, buildWorld, flyPoint, terrainHeight, worldBounds, type Chunk } from './world';
import shaderWgsl from './shader.wgsl?raw';

const MSAA_SAMPLES = 4;
const DEPTH_FORMAT: GPUTextureFormat = 'depth24plus';
const SHADOW_FORMAT: GPUTextureFormat = 'depth32float';
const MAX_CASCADES = 4;
const FOV_Y = (45 * Math.PI) / 180;
const CAMERA_NEAR = 0.3;
const CAMERA_FAR = 2000;
/** Distances (m) over which the far terrain dissolves into the horizon haze. */
const FADE_START = 380;
const FADE_END = 650;
/** Height of the fly-through camera above the (smoothed) ground. */
const FLY_HEIGHT = 8;

const DETAIL = [
  { grid: 128, density: 0.45 },
  { grid: 256, density: 1 },
  { grid: 384, density: 1.8 },
];

interface GpuWorld {
  terrain: Mesh;
  terrainChunks: Chunk[];
  groups: { mesh: Mesh; chunks: Chunk[] }[];
  objects: GPUBuffer;
}

const create: ExperimentFactory = async (ctx) => {
  const { device, surface, params, reducedMotion } = ctx;
  const coarse = window.matchMedia('(pointer: coarse)').matches;

  const cam = params.add(
    {
      cameraMode: {
        value: 'orbit',
        options: { Orbit: 'orbit', 'Fly-through': 'fly' },
        label: 'Camera',
        hint: 'Orbit: drag to orbit, wheel to zoom (it drifts slowly unless you prefer reduced motion). Fly-through glides around the valley so you can watch shadow edges for shimmering; drag to look around.',
      },
      flySpeed: { value: reducedMotion ? 0 : 4, min: 0, max: 30, step: 0.5, label: 'Fly speed (m/s)' },
    },
    'Camera',
  );
  const sun = params.add(
    {
      azimuth: { value: 205, min: 0, max: 360, step: 1 },
      elevation: { value: 16, min: 2, max: 80, step: 0.5, hint: 'A low sun makes long shadows that reach far across the terrain.' },
      sunIntensity: { value: 5, min: 0, max: 12, step: 0.05, label: 'Intensity' },
      sunColor: { value: '#ffc996', type: 'color', label: 'Colour' },
    },
    'Sun',
  );
  const csm = params.add(
    {
      cascades: {
        value: coarse ? 3 : 4,
        options: { '1 (single map)': 1, '2': 2, '3': 3, '4': 4 },
        hint: 'One shadow map per cascade. "1" spreads a single map over the whole shadow distance, for comparison.',
      },
      lambda: {
        value: 0.75,
        min: 0,
        max: 1,
        step: 0.01,
        label: 'Split λ',
        hint: 'Practical split scheme: 0 = uniform splits, 1 = logarithmic (tiny near cascades). In between gives the near cascades most of the resolution.',
      },
      shadowDistance: { value: 320, min: 30, max: 700, step: 5, label: 'Shadow distance (m)' },
      mapSize: {
        value: 1024,
        options: { '512': 512, '1024': 1024, '2048': 2048 },
        label: 'Map size',
        hint: 'Per cascade. Four 1024² depth32 layers are 16 MB; 2048² is 64 MB.',
      },
      blend: {
        value: 0.15,
        min: 0,
        max: 0.5,
        step: 0.01,
        label: 'Blend band',
        hint: 'Fraction of each cascade, at its far end, that blends into the next one (and, for the last cascade, fades shadows out).',
      },
      blendMode: {
        value: 0,
        options: { Lerp: 0, Dither: 1, None: 2 },
        label: 'Blend mode',
        hint: 'Lerp samples both cascades in the band (2× taps there). Dither picks one per pixel with rising probability: half the cost, but noisy without TAA.',
      },
      stabilize: {
        value: true,
        label: 'Stabilise',
        hint: 'Bounding-sphere fit + texel snapping: shadow edges hold still. Off = tight box fit: a little sharper, but every edge shimmers as the camera moves.',
      },
      freeze: { value: false, label: 'Freeze cascades', hint: 'Stop refitting to the camera. Switch to Orbit and zoom out to see where each cascade lands.' },
    },
    'Cascades',
  );
  const filt = params.add(
    {
      filter: { value: 1, options: { 'Hard (2×2 HW)': 0, 'PCF 3×3': 1, 'PCF 5×5': 2, 'Poisson (world radius)': 3 } },
      softness: {
        value: 0.15,
        min: 0.01,
        max: 1,
        step: 0.01,
        label: 'Poisson radius (m)',
        hint: 'Poisson only. A world-space radius keeps the penumbra the same width in every cascade; PCF kernels are in texels, so they widen at each seam.',
      },
      depthBias: {
        value: 0.6,
        min: 0,
        max: 4,
        step: 0.05,
        label: 'Depth bias',
        hint: "Slope-scaled, in each cascade's texels and multiplied by the filter's radius, so one value works for every cascade and filter.",
      },
      normalOffset: { value: 1, min: 0, max: 6, step: 0.1, label: 'Normal offset', hint: "In each cascade's texels." },
    },
    'Filtering',
  );
  const look = params.add(
    {
      background: { value: '#e6d9c8', type: 'color', hint: 'Page colour the horizon dissolves into (display space, matched exactly).' },
      haze: { value: '#e3b98e', type: 'color' },
      fog: { value: 0.0009, min: 0, max: 0.012, step: 0.0001, label: 'Haze density' },
      grass: { value: '#8c8a3c', type: 'color' },
      sky: { value: '#9db3d4', type: 'color', label: 'Sky (ambient)' },
      ambient: { value: 0.32, min: 0, max: 1.5, step: 0.01 },
      detail: {
        value: coarse ? 0 : 1,
        options: { Low: 0, Medium: 1, High: 2 },
        label: 'Scene detail',
        hint: 'Terrain resolution and prop count: about 3k / 6.5k / 12k instances, 0.2 / 0.5 / 1 million triangles.',
      },
      antialiasing: {
        value: 'msaa',
        options: { 'MSAA 4×': 'msaa', TAA: 'taa', Off: 'off' },
        hint: 'TAA also smooths dithered cascade blending and Poisson grain.',
      },
      view: { value: 0, options: { Lit: 0, 'Cascade colours': 1, Visibility: 2, 'Shadow texels': 3 } },
      showMaps: { value: false, label: 'Show cascade maps' },
    },
    'Look & debug',
  );
  const post = await PostProcess.create(ctx, { bloom: 0.03, vignette: 0.14, grain: 0.02 });
  const taa = await TemporalAA.create(device);

  // --- Uniforms -----------------------------------------------------------------
  const frame = new UniformBlock(device, 'Frame', {
    viewProj: 'mat4x4f',
    invViewProj: 'mat4x4f',
    cascade0: 'mat4x4f',
    cascade1: 'mat4x4f',
    cascade2: 'mat4x4f',
    cascade3: 'mat4x4f',
    cameraPos: 'vec3f',
    fogDensity: 'f32',
    cascadeOrigin: 'vec3f',
    cascadeCount: 'u32',
    cascadeAxis: 'vec3f',
    filterMode: 'u32',
    toSun: 'vec3f',
    blendMode: 'u32',
    sunColor: 'vec4f',
    skyColor: 'vec4f',
    groundColor: 'vec4f',
    // a = coverage of the haze band at the horizon
    hazeColor: 'vec4f',
    grassColor: 'vec4f',
    /** Far end of each cascade (view depth). */
    splits: 'vec4f',
    bands: 'vec4f',
    texelWorld: 'vec4f',
    depthRange: 'vec4f',
    receiverStart: 'vec4f',
    mapSize: 'f32',
    depthBias: 'f32',
    normalOffset: 'f32',
    softness: 'f32',
    debugView: 'u32',
    noiseOffset: 'f32',
    fadeStart: 'f32',
    fadeEnd: 'f32',
  });
  // One tiny block per cascade, so each shadow pass binds its own matrix.
  const shadowPasses = Array.from({ length: MAX_CASCADES }, () => new UniformBlock(device, 'ShadowPass', { viewProj: 'mat4x4f' }));

  // --- Pipelines ----------------------------------------------------------------
  const module = await createShader(
    device,
    'cascaded-shadows',
    wgsl(commonWgsl, noiseWgsl, fullscreenWgsl, MESH_VERTEX_WGSL, PROP_WGSL, frame.wgslStruct, shadowPasses[0].wgslStruct, brdfWgsl, shaderWgsl),
  );
  const mainLayout = device.createBindGroupLayout({
    label: 'cascaded-shadows',
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth', viewDimension: '2d-array' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
    ],
  });
  // The shadow passes can't bind the array they render into, so they get their own layout.
  const shadowLayout = device.createBindGroupLayout({
    label: 'cascaded-shadows:shadow-pass',
    entries: [
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
    ],
  });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [mainLayout] });
  const overlayDepth: GPUDepthStencilState = { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: 'always' };

  // Scene pipelines in a multisampled and a single-sampled flavour (TAA needs sampleable depth).
  const makeScenePipelines = (count: number) =>
    Promise.all([
      device.createRenderPipelineAsync({
        label: `csm-main-x${count}`,
        layout,
        vertex: { module, entryPoint: 'vs_main', buffers: [MESH_VERTEX_LAYOUT] },
        fragment: { module, entryPoint: 'fs_main', targets: [{ format: HDR_FORMAT }] },
        primitive: { topology: 'triangle-list', cullMode: 'back' },
        depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: 'less' },
        multisample: { count },
      }),
      device.createRenderPipelineAsync({
        label: `csm-sky-x${count}`,
        layout,
        vertex: { module, entryPoint: 'vs_fullscreen' },
        fragment: { module, entryPoint: 'fs_sky', targets: [{ format: HDR_FORMAT }] },
        depthStencil: overlayDepth,
        multisample: { count },
      }),
      device.createRenderPipelineAsync({
        label: `csm-thumbs-x${count}`,
        layout,
        vertex: { module, entryPoint: 'vs_thumb' },
        fragment: { module, entryPoint: 'fs_thumb', targets: [{ format: HDR_FORMAT }] },
        depthStencil: overlayDepth,
        multisample: { count },
      }),
    ]);
  const [shadowPipeline, msaaPipelines, singlePipelines] = await Promise.all([
    device.createRenderPipelineAsync({
      label: 'csm-shadow',
      layout: device.createPipelineLayout({ bindGroupLayouts: [shadowLayout] }),
      vertex: { module, entryPoint: 'vs_shadow', buffers: [MESH_VERTEX_LAYOUT] },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { format: SHADOW_FORMAT, depthWriteEnabled: true, depthCompare: 'less' },
    }),
    makeScenePipelines(MSAA_SAMPLES),
    makeScenePipelines(1),
  ]);

  const shadowSampler = device.createSampler({
    label: 'csm-compare',
    compare: 'less-equal',
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  });

  // --- World (rebuilt when the detail level changes) -----------------------------
  let world: GpuWorld | null = null;
  let worldDetail = -1;
  const buildGpuWorld = (detail: number): GpuWorld => {
    const w = buildWorld(DETAIL[detail]);
    const objects = device.createBuffer({
      label: 'csm-objects',
      size: w.objects.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(objects, 0, w.objects);
    return {
      terrain: uploadMesh(device, w.terrain.mesh, 'csm-terrain'),
      terrainChunks: w.terrain.chunks,
      groups: w.groups.map((g) => ({ mesh: uploadMesh(device, g.mesh, `csm-${g.kind}`), chunks: g.chunks })),
      objects,
    };
  };
  const destroyWorld = (w: GpuWorld) => {
    w.terrain.destroy();
    w.groups.forEach((g) => g.mesh.destroy());
    w.objects.destroy();
  };

  // --- Shadow map array (recreated when its size or layer count changes) ------------
  let shadowMap: GPUTexture | null = null;
  let layerViews: GPUTextureView[] = [];
  let mainBindGroup: GPUBindGroup | null = null;
  let shadowBindGroups: GPUBindGroup[] = [];
  const rebind = () => {
    if (!shadowMap || !world) return;
    const objects = world.objects;
    mainBindGroup = device.createBindGroup({
      layout: mainLayout,
      entries: [
        { binding: 0, resource: { buffer: frame.buffer } },
        { binding: 1, resource: { buffer: objects } },
        { binding: 2, resource: shadowMap.createView({ dimension: '2d-array' }) },
        { binding: 3, resource: shadowSampler },
      ],
    });
    shadowBindGroups = shadowPasses.map((block) =>
      device.createBindGroup({
        layout: shadowLayout,
        entries: [
          { binding: 1, resource: { buffer: objects } },
          { binding: 4, resource: { buffer: block.buffer } },
        ],
      }),
    );
  };
  const ensureShadowMap = (size: number, layers: number) => {
    if (shadowMap?.width === size && shadowMap.depthOrArrayLayers === layers) return;
    shadowMap?.destroy();
    shadowMap = device.createTexture({
      label: 'csm-shadow-maps',
      size: [size, size, layers],
      format: SHADOW_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    const tex = shadowMap;
    layerViews = Array.from({ length: layers }, (_, i) =>
      tex.createView({ label: `csm-layer-${i}`, dimension: '2d', baseArrayLayer: i, arrayLayerCount: 1 }),
    );
    rebind();
  };

  // --- Culling: each pass draws only the chunks inside its frustum ------------------
  const planes = new Float32Array(24);
  let planeCount = 0;
  /** Gribb–Hartmann planes of a WebGPU (z in 0..1) clip matrix. Shadow passes skip the near plane. */
  const setFrustum = (m: Mat4, withNear: boolean) => {
    const row = (i: number) => [m[i], m[4 + i], m[8 + i], m[12 + i]];
    const [r0, r1, r2, r3] = [row(0), row(1), row(2), row(3)];
    const list = [
      r3.map((v, i) => v + r0[i]),
      r3.map((v, i) => v - r0[i]),
      r3.map((v, i) => v + r1[i]),
      r3.map((v, i) => v - r1[i]),
      r3.map((v, i) => v - r2[i]),
    ];
    if (withNear) list.push(r2);
    list.forEach((p, i) => planes.set(p, i * 4));
    planeCount = list.length;
  };
  const visible = (c: Chunk) => {
    for (let o = 0; o < planeCount * 4; o += 4) {
      // The box corner furthest along the plane normal.
      const x = planes[o] >= 0 ? c.max[0] : c.min[0];
      const y = planes[o + 1] >= 0 ? c.max[1] : c.min[1];
      const z = planes[o + 2] >= 0 ? c.max[2] : c.min[2];
      if (planes[o] * x + planes[o + 1] * y + planes[o + 2] * z + planes[o + 3] < 0) return false;
    }
    return true;
  };
  /** Merges consecutive visible chunks (contiguous in their buffer) into single draws. */
  const drawRuns = (chunks: Chunk[], draw: (first: number, count: number) => void) => {
    let first = -1;
    let count = 0;
    for (const c of chunks) {
      if (!visible(c)) continue;
      if (first >= 0 && first + count === c.first) {
        count += c.count;
      } else {
        if (first >= 0) draw(first, count);
        first = c.first;
        count = c.count;
      }
    }
    if (first >= 0) draw(first, count);
  };
  const drawWorld = (pass: GPURenderPassEncoder, w: GpuWorld, viewProj: Mat4, isCamera: boolean) => {
    setFrustum(viewProj, isCamera);
    pass.setVertexBuffer(0, w.terrain.vertexBuffer);
    pass.setIndexBuffer(w.terrain.indexBuffer, 'uint32');
    drawRuns(w.terrainChunks, (first, count) => pass.drawIndexed(count, 1, first, 0, 0));
    for (const g of w.groups) {
      pass.setVertexBuffer(0, g.mesh.vertexBuffer);
      pass.setIndexBuffer(g.mesh.indexBuffer, 'uint32');
      drawRuns(g.chunks, (first, count) => pass.drawIndexed(g.mesh.indexCount, count, 0, 0, first));
    }
  };

  // --- Camera ---------------------------------------------------------------------
  const moundTop = terrainHeight(0, 0);
  const orbit = new OrbitCamera(surface.canvas, {
    target: [0, moundTop + 2, 0],
    distance: 36,
    yaw: 2.1,
    pitch: 0.15,
    minPitch: 0.03,
    maxPitch: 1.35,
    minDistance: 6,
    maxDistance: 260,
    fovY: FOV_Y,
    near: CAMERA_NEAR,
    far: CAMERA_FAR,
  });
  let flyAngle = 2.75;
  let lookRef: { yaw: number; pitch: number } | null = null;
  const view = mat4.identity();
  const proj = mat4.identity();
  const viewProj = mat4.identity();
  let frozen: { world: Mat4; aspect: number } | null = null;

  const targets = new TargetPool(device);
  const bounds = worldBounds();
  const casters = { center: vec3.fromValues(...bounds.center), radius: bounds.radius };

  return {
    frame({ time, dt, size }) {
      if (look.detail !== worldDetail) {
        if (world) destroyWorld(world);
        world = buildGpuWorld(look.detail);
        worldDetail = look.detail;
        rebind();
      }
      const count = csm.cascades;
      ensureShadowMap(csm.mapSize, count);
      const w = world!;
      const aspect = size.width / size.height;

      // Camera: orbit, or a slow glide around the valley (drag offsets the look direction).
      const fly = cam.cameraMode === 'fly';
      orbit.autoRotate = !fly && !reducedMotion ? 0.03 : 0;
      orbit.update(dt, aspect);
      let eye: Float32Array;
      if (fly) {
        lookRef ??= { yaw: orbit.yaw, pitch: orbit.pitch };
        flyAngle += (dt * cam.flySpeed) / FLY_RADIUS;
        const p = flyPoint(flyAngle);
        // Ride over a smoothed ground so the camera doesn't bob with every bump.
        let ground = 0;
        for (let k = -3; k <= 5; k++) {
          const q = flyPoint(flyAngle + k * 0.07);
          ground += terrainHeight(q.x, q.z) / 9;
        }
        const y = Math.max(ground, terrainHeight(p.x, p.z) + 1) + FLY_HEIGHT;
        // Look across the valley, past the stone circle.
        const inward = 1.2;
        const lx = p.dirX * Math.cos(inward) - (p.x / FLY_RADIUS) * Math.sin(inward);
        const lz = p.dirZ * Math.cos(inward) - (p.z / FLY_RADIUS) * Math.sin(inward);
        const yaw = Math.atan2(-lx, -lz) + (orbit.yaw - lookRef.yaw);
        const pitch = 0.12 + (orbit.pitch - lookRef.pitch);
        eye = vec3.fromValues(p.x, y, p.z);
        const dir = [-Math.sin(yaw) * Math.cos(pitch), -Math.sin(pitch), -Math.cos(yaw) * Math.cos(pitch)];
        mat4.lookAt(eye, vec3.add(eye, dir), [0, 1, 0], view);
        mat4.perspective(FOV_Y, aspect, CAMERA_NEAR, CAMERA_FAR, proj);
      } else {
        lookRef = null;
        eye = vec3.clone(orbit.position);
        mat4.copy(orbit.view, view);
        mat4.copy(orbit.projection, proj);
      }
      mat4.multiply(proj, view, viewProj);
      const cameraWorld = mat4.inverse(view);

      // Sun.
      const az = (sun.azimuth * Math.PI) / 180;
      const el = (sun.elevation * Math.PI) / 180;
      const toSun = vec3.fromValues(Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az));

      // Cascades: fitted to the live camera, or to the one captured when "Freeze" was switched on.
      if (csm.freeze) frozen ??= { world: mat4.clone(cameraWorld), aspect };
      else frozen = null;
      const fitCamera = frozen ?? { world: cameraWorld, aspect };
      const setup = fitCascades({
        cameraWorld: fitCamera.world,
        fovY: FOV_Y,
        aspect: fitCamera.aspect,
        near: CAMERA_NEAR,
        shadowDistance: csm.shadowDistance,
        count,
        lambda: csm.lambda,
        blend: csm.blend,
        toLight: toSun,
        mapSize: csm.mapSize,
        stable: csm.stabilize,
        casters,
      });
      const perCascade = (f: (i: number) => number) => Array.from({ length: MAX_CASCADES }, (_, i) => (i < count ? f(i) : 0));

      const aa = look.antialiasing;
      const samples = aa === 'msaa' ? MSAA_SAMPLES : 1;
      const [mainPipeline, skyPipeline, thumbPipeline] = samples > 1 ? msaaPipelines : singlePipelines;
      if (aa !== 'taa') taa.reset();
      const drawViewProj = aa === 'taa' ? taa.jitter(viewProj, size) : viewProj;

      const sc = hexToLinear(sun.sunColor);
      const sky = hexToLinear(look.sky);
      const grass = hexToLinear(look.grass);
      const haze = hexToLinear(look.haze);
      const ambient = look.ambient;
      frame
        .set('viewProj', drawViewProj)
        .set('invViewProj', mat4.inverse(viewProj))
        .set('cameraPos', eye)
        .set('fogDensity', look.fog)
        .set('cascadeOrigin', fitCamera.world.slice(12, 15))
        .set('cascadeCount', count)
        .set('cascadeAxis', vec3.negate(fitCamera.world.slice(8, 11)))
        .set('filterMode', filt.filter)
        .set('toSun', toSun)
        .set('blendMode', csm.blendMode)
        .set('sunColor', [sc[0] * sun.sunIntensity, sc[1] * sun.sunIntensity, sc[2] * sun.sunIntensity, 0])
        .set('skyColor', [sky[0] * ambient, sky[1] * ambient, sky[2] * ambient, 0])
        // Bounce light off sunlit grass.
        .set('groundColor', [grass[0] * ambient * 0.6, grass[1] * ambient * 0.6, grass[2] * ambient * 0.6, 0])
        .set('hazeColor', [haze[0], haze[1], haze[2], 0.9])
        .set('grassColor', [...grass, 0])
        .set('splits', perCascade((i) => setup.splits[i + 1]))
        .set('bands', perCascade((i) => setup.bands[i]))
        .set('texelWorld', perCascade((i) => setup.cascades[i].texelWorld))
        .set('depthRange', perCascade((i) => setup.cascades[i].depthRange))
        .set('receiverStart', perCascade((i) => setup.cascades[i].receiverStart))
        .set('mapSize', csm.mapSize)
        .set('depthBias', filt.depthBias)
        .set('normalOffset', filt.normalOffset)
        .set('softness', filt.softness)
        .set('debugView', look.view)
        .set('noiseOffset', aa === 'taa' ? taa.frameNoiseOffset : 0)
        .set('fadeStart', FADE_START)
        .set('fadeEnd', FADE_END);
      (['cascade0', 'cascade1', 'cascade2', 'cascade3'] as const).forEach((key, i) => {
        if (i < count) frame.set(key, setup.cascades[i].viewProj);
      });
      frame.upload();
      setup.cascades.forEach((c, i) => shadowPasses[i].set('viewProj', c.viewProj).upload());

      const encoder = device.createCommandEncoder({ label: 'cascaded-shadows' });

      // 1. One depth pass per cascade, each into its own array layer, culled to that cascade.
      setup.cascades.forEach((c, i) => {
        const sp = encoder.beginRenderPass({
          label: `csm-shadow-${i}`,
          colorAttachments: [],
          depthStencilAttachment: { view: layerViews[i], depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
        });
        sp.setPipeline(shadowPipeline);
        sp.setBindGroup(0, shadowBindGroups[i]);
        drawWorld(sp, w, c.viewProj, false);
        sp.end();
      });

      // 2. Main pass: horizon haze, then the world.
      const fullSize = [size.width, size.height] as const;
      const scene = post.scene(size);
      const msaaColor = samples > 1 ? targets.get('msaa', { size: fullSize, format: HDR_FORMAT, sampleCount: samples }) : null;
      const depth = targets.get(`depth-x${samples}`, {
        size: fullSize,
        format: DEPTH_FORMAT,
        sampleCount: samples,
        usage: samples > 1 ? GPUTextureUsage.RENDER_ATTACHMENT : GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      const mp = encoder.beginRenderPass({
        label: 'csm-main',
        colorAttachments: [
          msaaColor
            ? { view: msaaColor.view, resolveTarget: scene.view, loadOp: 'clear', storeOp: 'discard', clearValue: [0, 0, 0, 0] }
            : { view: scene.view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] },
        ],
        depthStencilAttachment: {
          view: depth.view,
          depthClearValue: 1,
          depthLoadOp: 'clear',
          depthStoreOp: aa === 'taa' ? 'store' : 'discard',
        },
      });
      mp.setBindGroup(0, mainBindGroup!);
      mp.setPipeline(skyPipeline);
      mp.draw(3);
      mp.setPipeline(mainPipeline);
      drawWorld(mp, w, viewProj, true);

      if (look.showMaps) {
        // A row of thumbnails along the bottom-left, one per cascade (the instance index is the layer).
        const m = Math.round(12 * size.pixelRatio);
        const s = Math.round(Math.min(Math.min(size.width, size.height) * 0.24, (size.width - m * (count + 1)) / count));
        mp.setPipeline(thumbPipeline);
        for (let i = 0; i < count; i++) {
          mp.setViewport(m + i * (s + m), size.height - s - m, s, s, 0, 1);
          mp.draw(3, 1, 0, i);
        }
      }
      mp.end();

      const source = aa === 'taa' ? taa.resolve(encoder, { color: scene.texture, depth: depth.texture, viewProj }) : undefined;
      post.render(encoder, surface.currentTexture().createView(), {
        time,
        raw: look.view !== 0,
        background: hexToLinear(look.background),
        source,
      });
      device.queue.submit([encoder.finish()]);
    },
    dispose() {
      orbit.destroy();
      shadowMap?.destroy();
      if (world) destroyWorld(world);
      targets.destroy();
      post.destroy();
      taa.destroy();
      frame.destroy();
      shadowPasses.forEach((b) => b.destroy());
    },
  };
};

export default create;
