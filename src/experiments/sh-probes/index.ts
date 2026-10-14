import commonWgsl from '../../shaders/common.wgsl?raw';
import noiseWgsl from '../../shaders/noise.wgsl?raw';
import shWgsl from '../../shaders/sh.wgsl?raw';
import brdfWgsl from '../../shaders/brdf.wgsl?raw';
import shadowWgsl from '../../shaders/shadow.wgsl?raw';
import sunRoomWgsl from '../../shaders/sun-room.wgsl?raw';
import sunRoomTraceWgsl from '../../shaders/sun-room-trace.wgsl?raw';
import { MESH_VERTEX_LAYOUT, MESH_VERTEX_WGSL, sphere, uploadMesh } from '../../gpu/geometry';
import { createShader, wgsl } from '../../gpu/shader';
import { TargetPool } from '../../gpu/targets';
import { UniformBlock } from '../../gpu/uniforms';
import { hexToLinear } from '../../gpu/color';
import { HDR_FORMAT, PostProcess } from '../../post/post';
import { OrbitCamera } from '../../scene/orbit-camera';
import { fitDirectionalShadow } from '../../scene/light-frustum';
import { PROP_WGSL } from '../../scene/props';
import {
  DEFAULT_ROOM_COLORS,
  DEFAULT_SUN,
  ROOM_PRIM_WGSL,
  SUN_ROOM,
  SUN_ROOM_BOUNDS,
  SUN_ROOM_CAMERA,
  SunRoom,
  outdoorColors,
  sunDirection,
} from '../../scene/sun-room';
import type { ExperimentFactory } from '../types';
import shaderWgsl from './shader.wgsl?raw';

const MSAA_SAMPLES = 4;
const DEPTH_FORMAT: GPUTextureFormat = 'depth24plus';
const SHADOW_FORMAT: GPUTextureFormat = 'depth32float';
/** Probe visibility tiles: 8×8 octahedral texels plus a one-texel border (matches TILE in the shader). */
const TILE = 10;
/** Bounce volumes kept on the GPU (the "Bounces" param picks how many are used). */
const MAX_BOUNCES = 3;
const COEFFS = 9;
const VIEW_TRUST = 4;
/** Sweeps averaged before the bake stops (each with freshly rotated rays). */
const CONVERGED_SWEEPS = 48;
/**
 * After the light changes, the volumes keep blending with this many sweeps' worth of history:
 * enough to hide ray noise while the sun sways, short enough (α = 1/5) to follow it.
 */
const MOVING_SWEEPS = 4;
const LEAK = { off: 0, normal: 1, visibility: 2 } as const;

const create: ExperimentFactory = async (ctx) => {
  const { device, surface, params, reducedMotion } = ctx;
  const coarse = window.matchMedia('(pointer: coarse)').matches;

  const sun = params.add(
    {
      azimuth: { value: DEFAULT_SUN.azimuth, min: -150, max: -30, step: 1, hint: 'The window faces −x; the sun has to be on that side to get in.' },
      elevation: { value: DEFAULT_SUN.elevation, min: 12, max: 70, step: 1 },
      sunIntensity: { value: 9, min: 0, max: 20, step: 0.05, label: 'Intensity' },
      sunColor: { value: '#fff0dc', type: 'color', label: 'Colour' },
      sky: { value: '#9cc0ee', type: 'color', hint: 'Seen by the probes through the window: a cool fill that only reaches the room as probe light.' },
      skyIntensity: { value: 1.5, min: 0, max: 4, step: 0.05, label: 'Sky intensity' },
      lamp: {
        value: 0.6,
        min: 0,
        max: 8,
        step: 0.05,
        label: 'Lamp shade',
        hint: 'Radiance of the emissive lamp shade. It has no light source of its own: it lights the room only through the probes.',
      },
      sway: { value: !reducedMotion, label: 'Sway sun', hint: 'Swings the sun slowly, so the probes re-bake progressively every frame.' },
    },
    'Light',
  );
  const bake = params.add(
    {
      grid: {
        value: '8x4x7',
        options: { '4 × 2 × 4 (32)': '4x2x4', '6 × 3 × 5 (90)': '6x3x5', '8 × 4 × 7 (224)': '8x4x7', '12 × 6 × 10 (720)': '12x6x10', '16 × 8 × 13 (1664)': '16x8x13' },
        label: 'Probe grid',
        hint: 'Probes across × up × deep, spread evenly through the room. Finer grids follow the light better and leak less, at more bake work and memory.',
      },
      rays: {
        value: coarse ? 64 : 128,
        options: { '64': 64, '128': 128, '256': 256 },
        label: 'Rays / probe',
        hint: 'Per probe, bounce and sweep. Each sweep turns the directions randomly and is averaged in, so the bake converges over 48 sweeps, then stops.',
      },
      bounces: {
        value: 2,
        options: { '1': 1, '2': 2, '3': 3 },
        hint: 'Bake passes. Pass 1 sees sunlit surfaces and the sky; each further pass also sees the previous pass\'s probe light on those surfaces.',
      },
      budget: {
        value: 64,
        min: 8,
        max: 1024,
        step: 8,
        label: 'Probes / frame',
        hint: 'Probes re-baked per frame (for every bounce); one pass over the grid is a sweep. After a light change, old light fades out over a few sweeps.',
      },
      order: { value: 2, options: { 'L1 (4 coefficients)': 1, 'L2 (9 coefficients)': 2 }, label: 'SH order' },
      dering: {
        value: 0,
        min: 0,
        max: 1,
        step: 0.01,
        label: 'Deringing',
        hint: 'Fades in a Hann window over the SH bands: softer light, less overshoot opposite bright sources.',
      },
    },
    'Probes',
  );
  const sampling = params.add(
    {
      leak: {
        value: LEAK.visibility,
        options: { 'Off (plain trilinear)': LEAK.off, 'Normal bias + back-face': LEAK.normal, 'Visibility (DDGI-lite)': LEAK.visibility },
        label: 'Leak reduction',
        hint: 'Off blends the 8 surrounding probes by distance alone, so light from probes behind walls and furniture leaks in.',
      },
      bias: {
        value: 0.25,
        min: 0,
        max: 0.6,
        step: 0.01,
        label: 'Normal bias',
        hint: 'How far lookups move off the surface (along the normal, leaning toward the camera), as a fraction of the probe spacing.',
      },
      indirect: { value: 1, min: 0, max: 3, step: 0.01, label: 'Indirect intensity' },
    },
    'Sampling',
  );
  const look = params.add(
    {
      rug: { value: DEFAULT_ROOM_COLORS.rug, type: 'color', hint: 'Recolouring the room re-bakes the probes.' },
      walls: { value: DEFAULT_ROOM_COLORS.walls, type: 'color' },
      view: {
        value: 0,
        options: { Lit: 0, 'Indirect only': 1, 'Probe irradiance': 2, 'Direct only': 3, 'Leak weights': VIEW_TRUST },
        hint: 'Probe irradiance: probe light on a white surface. Leak weights: how much of the trilinear weight survived the visibility test (red = mostly rejected as hidden).',
      },
      showProbes: { value: false, label: 'Show probes', hint: 'Each probe as a white ball lit by its own SH. Dark red: switched off (inside furniture).' },
      showGrid: { value: false, label: 'Show grid' },
      shadowSize: { value: coarse ? 1024 : 2048, options: { '1024': 1024, '2048': 2048, '4096': 4096 }, label: 'Sun map size' },
      antialiasing: { value: 'msaa', options: { 'MSAA 4×': 'msaa', Off: 'off' } },
    },
    'Look & debug',
  );
  const post = await PostProcess.create(ctx, { exposure: 2.2, bloom: 0.04, vignette: 0.25, grain: 0.015 });

  const room = new SunRoom(device, 'sh-probes', { traceable: true });
  const probeMesh = uploadMesh(device, sphere(1, 16, 12), 'sh-probes:probe');

  const frame = new UniformBlock(device, 'Frame', {
    viewProj: 'mat4x4f',
    lightViewProj: 'mat4x4f',
    cameraPos: 'vec3f',
    shadowTexelWorld: 'f32',
    toSun: 'vec3f',
    indirect: 'f32',
    sunColor: 'vec4f',
    skyZenith: 'vec4f',
    skyHorizon: 'vec4f',
    outdoorGround: 'vec4f',
    lampColor: 'vec4f',
    gridMin: 'vec3f',
    normalBias: 'f32',
    gridCell: 'vec3f',
    shOrder: 'u32',
    gridDims: 'vec4u',
    atlasSize: 'vec2f',
    leakMode: 'u32',
    debugView: 'u32',
    dering: 'f32',
    probeRadius: 'f32',
    normalOffset: 'f32',
  });
  const STEP = { offset: 'u32', rays: 'u32', bounce: 'u32', seed: 'u32', blend: 'f32' } as const;
  const steps = Array.from({ length: MAX_BOUNCES }, () => new UniformBlock(device, 'Step', STEP));
  const visStep = new UniformBlock(device, 'Step', STEP);

  const module = await createShader(
    device,
    'sh-probes',
    wgsl(
      commonWgsl,
      noiseWgsl,
      shWgsl,
      brdfWgsl,
      shadowWgsl,
      sunRoomWgsl,
      MESH_VERTEX_WGSL,
      PROP_WGSL,
      ROOM_PRIM_WGSL,
      sunRoomTraceWgsl,
      frame.wgslStruct,
      visStep.wgslStruct,
      shaderWgsl,
    ),
  );

  // --- Layouts: scene (0), probe volume (1), then per-pass resources (2) -------------------
  const ALL = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE;
  const sceneLayout = device.createBindGroupLayout({
    label: 'sh-probes:scene',
    entries: [
      { binding: 0, visibility: ALL, buffer: { type: 'uniform' } },
      { binding: 1, visibility: ALL, buffer: { type: 'read-only-storage' } },
    ],
  });
  const volumeLayout = device.createBindGroupLayout({
    label: 'sh-probes:volume',
    entries: [
      { binding: 0, visibility: ALL, texture: { sampleType: 'float' } },
      { binding: 1, visibility: ALL, sampler: { type: 'filtering' } },
      { binding: 2, visibility: ALL, buffer: { type: 'read-only-storage' } },
    ],
  });
  const emptyLayout = device.createBindGroupLayout({ label: 'sh-probes:empty', entries: [] });
  const bakeOutLayout = device.createBindGroupLayout({
    label: 'sh-probes:bake-out',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });
  const visOutLayout = device.createBindGroupLayout({
    label: 'sh-probes:vis-out',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
    ],
  });
  const shadowPassLayout = device.createBindGroupLayout({
    label: 'sh-probes:shadow-pass',
    entries: [{ binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }],
  });
  const cameraLayout = device.createBindGroupLayout({
    label: 'sh-probes:camera',
    entries: [
      { binding: 3, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
      { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
    ],
  });
  const pipelineLayout = (...layouts: GPUBindGroupLayout[]) => device.createPipelineLayout({ bindGroupLayouts: layouts });
  const cameraPipelineLayout = pipelineLayout(sceneLayout, volumeLayout, cameraLayout);

  const scenePipelines = (count: number) => {
    const depthStencil: GPUDepthStencilState = { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: 'less' };
    const multisample = { count };
    return Promise.all([
      device.createRenderPipelineAsync({
        label: `sh-probes:main-x${count}`,
        layout: cameraPipelineLayout,
        vertex: { module, entryPoint: 'vs_main', buffers: [MESH_VERTEX_LAYOUT] },
        fragment: { module, entryPoint: 'fs_main', targets: [{ format: HDR_FORMAT }] },
        primitive: { topology: 'triangle-list', cullMode: 'back' },
        depthStencil,
        multisample,
      }),
      device.createRenderPipelineAsync({
        label: `sh-probes:probes-x${count}`,
        layout: cameraPipelineLayout,
        vertex: { module, entryPoint: 'vs_probe', buffers: [MESH_VERTEX_LAYOUT] },
        fragment: { module, entryPoint: 'fs_probe', targets: [{ format: HDR_FORMAT }] },
        primitive: { topology: 'triangle-list', cullMode: 'back' },
        depthStencil,
        multisample,
      }),
      device.createRenderPipelineAsync({
        label: `sh-probes:grid-x${count}`,
        layout: cameraPipelineLayout,
        vertex: { module, entryPoint: 'vs_grid' },
        fragment: { module, entryPoint: 'fs_grid', targets: [{ format: HDR_FORMAT }] },
        primitive: { topology: 'line-list' },
        depthStencil: { ...depthStencil, depthWriteEnabled: false },
        multisample,
      }),
    ]);
  };
  const [probePipeline, visPipeline, shadowPipeline, msaaPipelines, singlePipelines] = await Promise.all([
    device.createComputePipelineAsync({
      label: 'sh-probes:bake',
      layout: pipelineLayout(sceneLayout, volumeLayout, bakeOutLayout),
      compute: { module, entryPoint: 'cs_probe' },
    }),
    device.createComputePipelineAsync({
      label: 'sh-probes:visibility',
      layout: pipelineLayout(sceneLayout, emptyLayout, visOutLayout),
      compute: { module, entryPoint: 'cs_visibility' },
    }),
    device.createRenderPipelineAsync({
      label: 'sh-probes:shadow',
      layout: pipelineLayout(sceneLayout, emptyLayout, shadowPassLayout),
      vertex: { module, entryPoint: 'vs_shadow', buffers: [MESH_VERTEX_LAYOUT] },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { format: SHADOW_FORMAT, depthWriteEnabled: true, depthCompare: 'less', depthBias: 2, depthBiasSlopeScale: 1.5 },
    }),
    scenePipelines(MSAA_SAMPLES),
    scenePipelines(1),
  ]);

  const linearSampler = device.createSampler({ label: 'sh-probes:linear', magFilter: 'linear', minFilter: 'linear' });
  const shadowSampler = device.createSampler({
    label: 'sh-probes:shadow-compare',
    compare: 'less-equal',
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  });
  const sceneGroup = device.createBindGroup({
    layout: sceneLayout,
    entries: [
      { binding: 0, resource: { buffer: frame.buffer } },
      { binding: 1, resource: { buffer: room.primBuffer! } },
    ],
  });
  const emptyGroup = device.createBindGroup({ layout: emptyLayout, entries: [] });
  const shadowPassGroup = device.createBindGroup({ layout: shadowPassLayout, entries: [{ binding: 3, resource: { buffer: room.props.buffer } }] });

  // Bake schedule: probes are re-baked `budget` at a time, every bounce in turn, each sweep with
  // differently rotated rays, blended into the volume as a running average (weight 1/(n+1) for
  // the n-th sweep). Once CONVERGED_SWEEPS have been averaged the bake idles; a light change
  // rewinds n to MOVING_SWEEPS, so old light fades out over a few sweeps.
  let needVisibility = true;
  let sweeps = 0;
  let sweepIndex = 0;
  let cursor = 0;
  let lightKey = '';

  // --- Probe grid resources (rebuilt when the grid changes) ---------------------------------
  interface Grid {
    key: string;
    dims: [number, number, number];
    count: number;
    volumes: GPUBuffer[];
    atlas: GPUTexture;
    /** volumeGroups[i]: the atlas + volume i, read-only (the display, or a bake's source). */
    volumeGroups: GPUBindGroup[];
    /** outGroups[i]: step i + volume i, written by the bake of bounce i + 1. */
    outGroups: GPUBindGroup[];
    visGroup: GPUBindGroup;
  }
  let grid: Grid | null = null;
  const destroyGrid = (g: Grid) => {
    g.volumes.forEach((b) => b.destroy());
    g.atlas.destroy();
  };
  const ensureGrid = (key: string): Grid => {
    if (grid?.key === key) return grid;
    if (grid) destroyGrid(grid);
    const dims = key.split('x').map(Number) as [number, number, number];
    const count = dims[0] * dims[1] * dims[2];
    // New buffers start zeroed: no probe light until the first pass of the bake reaches them.
    const volumes = Array.from({ length: MAX_BOUNCES }, (_, i) =>
      device.createBuffer({ label: `sh-probes:volume${i + 1}`, size: count * COEFFS * 16, usage: GPUBufferUsage.STORAGE }),
    );
    const atlas = device.createTexture({
      label: 'sh-probes:visibility',
      size: [dims[0] * dims[2] * TILE, dims[1] * TILE],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    const atlasView = atlas.createView();
    grid = {
      key,
      dims,
      count,
      volumes,
      atlas,
      volumeGroups: volumes.map((v) =>
        device.createBindGroup({
          layout: volumeLayout,
          entries: [
            { binding: 0, resource: atlasView },
            { binding: 1, resource: linearSampler },
            { binding: 2, resource: { buffer: v } },
          ],
        }),
      ),
      outGroups: volumes.map((v, i) =>
        device.createBindGroup({
          layout: bakeOutLayout,
          entries: [
            { binding: 0, resource: { buffer: steps[i].buffer } },
            { binding: 1, resource: { buffer: v } },
          ],
        }),
      ),
      visGroup: device.createBindGroup({
        layout: visOutLayout,
        entries: [
          { binding: 0, resource: { buffer: visStep.buffer } },
          { binding: 2, resource: atlasView },
        ],
      }),
    };
    needVisibility = true;
    sweeps = 0;
    cursor = 0;
    return grid;
  };

  let cameraGroup: GPUBindGroup | null = null;
  let boundShadowMap: GPUTexture | null = null;
  const cameraBindings = (shadowMap: GPUTexture) => {
    if (cameraGroup && boundShadowMap === shadowMap) return cameraGroup;
    boundShadowMap = shadowMap;
    cameraGroup = device.createBindGroup({
      layout: cameraLayout,
      entries: [
        { binding: 3, resource: { buffer: room.props.buffer } },
        { binding: 4, resource: shadowMap.createView() },
        { binding: 5, resource: shadowSampler },
      ],
    });
    return cameraGroup;
  };

  const targets = new TargetPool(device);
  const camera = new OrbitCamera(surface.canvas, SUN_ROOM_CAMERA);
  let swayTime = 0;
  const background = hexToLinear('#1b1917');

  return {
    frame({ time, dt, size }) {
      if (sun.sway) swayTime += dt;
      camera.update(dt, size.width / size.height);
      room.updateCutaway(camera.position);
      const g = ensureGrid(bake.grid);
      const recoloured = room.setColors({ rug: look.rug, walls: look.walls });

      // Sun: a slow sway around the chosen direction.
      const toSun = sunDirection(sun.azimuth + 12 * Math.sin(swayTime * 0.11), sun.elevation + 3 * Math.sin(swayTime * 0.07 + 1));
      const sc = hexToLinear(sun.sunColor).map((c) => c * sun.sunIntensity);
      const sky = hexToLinear(sun.sky).map((c) => c * sun.skyIntensity);
      const outdoor = outdoorColors(sc, sky, toSun, sun.skyIntensity);
      const lamp = hexToLinear('#ffc98a').map((c) => c * sun.lamp);
      const key = [...toSun, ...sc, ...sky, ...lamp, sampling.leak, sampling.bias, bake.order, bake.dering, bake.bounces, bake.rays].map((v) => v.toFixed(5)).join();
      if (key !== lightKey || recoloured) {
        lightKey = key;
        sweeps = Math.min(sweeps, MOVING_SWEEPS);
      }

      const [nx, ny, nz] = g.dims;
      const cell = [0, 1, 2].map((i) => (SUN_ROOM.max[i] - SUN_ROOM.min[i]) / g.dims[i]);
      const shadowMapSize = look.shadowSize;
      const fit = fitDirectionalShadow({ toLight: toSun, bounds: SUN_ROOM_BOUNDS, mapSize: shadowMapSize });
      const samples = look.antialiasing === 'msaa' ? MSAA_SAMPLES : 1;
      const [mainPipeline, spherePipeline, gridPipeline] = samples > 1 ? msaaPipelines : singlePipelines;

      frame
        .set('viewProj', camera.viewProjection)
        .set('lightViewProj', fit.viewProj)
        .set('cameraPos', camera.position)
        .set('shadowTexelWorld', fit.texelWorld)
        .set('toSun', toSun)
        .set('indirect', sampling.indirect)
        .set('sunColor', [...sc, 0])
        .set('skyZenith', [...outdoor.zenith, 0])
        .set('skyHorizon', [...outdoor.horizon, 0])
        .set('outdoorGround', [...outdoor.ground, 0])
        .set('lampColor', [...lamp, 0])
        .set('gridMin', SUN_ROOM.min)
        .set('normalBias', sampling.bias * Math.min(...cell))
        .set('gridCell', cell)
        .set('shOrder', bake.order)
        .set('gridDims', [nx, ny, nz, g.count])
        .set('atlasSize', [g.atlas.width, g.atlas.height])
        .set('leakMode', sampling.leak)
        .set('debugView', look.view)
        .set('dering', bake.dering)
        .set('probeRadius', Math.min(...cell) * 0.12)
        .set('normalOffset', 1.5)
        .upload();

      const encoder = device.createCommandEncoder({ label: 'sh-probes' });

      // 1. Bake: visibility once per grid, then a chunk of probes for every bounce.
      const bakeNow = needVisibility || sweeps < CONVERGED_SWEEPS;
      if (bakeNow) {
        const cp = encoder.beginComputePass({ label: 'sh-probes:bake' });
        cp.setBindGroup(0, sceneGroup);
        if (needVisibility) {
          visStep.set('offset', 0).set('rays', 0).set('bounce', 0).upload();
          cp.setPipeline(visPipeline);
          cp.setBindGroup(1, emptyGroup);
          cp.setBindGroup(2, g.visGroup);
          cp.dispatchWorkgroups(g.count);
          needVisibility = false;
        }
        if (sweeps < CONVERGED_SWEEPS) {
          const chunk = Math.min(bake.budget, g.count - cursor);
          cp.setPipeline(probePipeline);
          for (let b = 1; b <= bake.bounces; b++) {
            steps[b - 1].set('offset', cursor).set('rays', bake.rays).set('bounce', b).set('seed', sweepIndex).set('blend', 1 / (sweeps + 1)).upload();
            // Bounce b reads volume b − 1. The first bounce reads nothing, but needs some other volume bound.
            cp.setBindGroup(1, g.volumeGroups[b === 1 ? 1 : b - 2]);
            cp.setBindGroup(2, g.outGroups[b - 1]);
            cp.dispatchWorkgroups(chunk);
          }
          cursor += chunk;
          if (cursor >= g.count) {
            cursor = 0;
            sweeps++;
            sweepIndex++;
          }
        }
        cp.end();
      }

      // 2. Sun shadow map: every primitive, cut-away walls included.
      const shadowMap = targets.get('shadow', { size: [shadowMapSize, shadowMapSize], format: SHADOW_FORMAT });
      const sp = encoder.beginRenderPass({
        label: 'sh-probes:shadow',
        colorAttachments: [],
        depthStencilAttachment: { view: shadowMap.view, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
      });
      sp.setPipeline(shadowPipeline);
      sp.setBindGroup(0, sceneGroup);
      sp.setBindGroup(1, emptyGroup);
      sp.setBindGroup(2, shadowPassGroup);
      room.draw(sp);
      sp.end();

      // 3. Camera pass.
      const fullSize = [size.width, size.height] as const;
      const scene = post.scene(size);
      const msaaColor = samples > 1 ? targets.get('msaa', { size: fullSize, format: HDR_FORMAT, sampleCount: samples }) : null;
      const depth = targets.get(`depth-x${samples}`, { size: fullSize, format: DEPTH_FORMAT, sampleCount: samples, usage: GPUTextureUsage.RENDER_ATTACHMENT });
      const mp = encoder.beginRenderPass({
        label: 'sh-probes:main',
        colorAttachments: [
          msaaColor
            ? { view: msaaColor.view, resolveTarget: scene.view, loadOp: 'clear', storeOp: 'discard', clearValue: [0, 0, 0, 0] }
            : { view: scene.view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] },
        ],
        depthStencilAttachment: { view: depth.view, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard' },
      });
      mp.setBindGroup(0, sceneGroup);
      mp.setBindGroup(1, g.volumeGroups[bake.bounces - 1]);
      mp.setBindGroup(2, cameraBindings(shadowMap.texture));
      mp.setPipeline(mainPipeline);
      room.draw(mp, { camera: true });
      if (look.showProbes) {
        mp.setPipeline(spherePipeline);
        mp.setVertexBuffer(0, probeMesh.vertexBuffer);
        mp.setIndexBuffer(probeMesh.indexBuffer, 'uint32');
        mp.drawIndexed(probeMesh.indexCount, g.count);
      }
      if (look.showGrid) {
        mp.setPipeline(gridPipeline);
        mp.draw(2, g.count * 3);
      }
      mp.end();

      post.render(encoder, surface.currentTexture().createView(), { time, raw: look.view === VIEW_TRUST, background });
      device.queue.submit([encoder.finish()]);
    },
    dispose() {
      camera.destroy();
      if (grid) destroyGrid(grid);
      targets.destroy();
      post.destroy();
      room.destroy();
      probeMesh.destroy();
      frame.destroy();
      steps.forEach((s) => s.destroy());
      visStep.destroy();
    },
  };
};

export default create;
