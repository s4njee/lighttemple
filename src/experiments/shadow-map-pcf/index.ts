import { mat4, vec3 } from 'wgpu-matrix';
import commonWgsl from '../../shaders/common.wgsl?raw';
import fullscreenWgsl from '../../shaders/fullscreen.wgsl?raw';
import brdfWgsl from '../../shaders/brdf.wgsl?raw';
import shadowWgsl from '../../shaders/shadow.wgsl?raw';
import shadowMomentsWgsl from '../../shaders/shadow-moments.wgsl?raw';
import contactShadowWgsl from '../../shaders/contact-shadow.wgsl?raw';
import { MESH_VERTEX_LAYOUT, MESH_VERTEX_WGSL, box, plane, sphere, torus, uploadMesh } from '../../gpu/geometry';
import { createShader, wgsl } from '../../gpu/shader';
import { TargetPool } from '../../gpu/targets';
import { HDR_FORMAT, PostProcess } from '../../post/post';
import { TemporalAA } from '../../post/taa';
import { SeparableBlur } from '../../post/blur-separable';
import { UniformBlock } from '../../gpu/uniforms';
import { OrbitCamera } from '../../scene/orbit-camera';
import { fitDirectionalShadow, frustumSliceSphere, type Sphere } from '../../scene/light-frustum';
import { PROP_WGSL, PropSet } from '../../scene/props';
import { hexToLinear } from '../../gpu/color';
import type { ExperimentFactory } from '../types';
import shaderWgsl from './shader.wgsl?raw';

const MSAA_SAMPLES = 4;
const DEPTH_FORMAT: GPUTextureFormat = 'depth24plus';
const SHADOW_FORMAT: GPUTextureFormat = 'depth32float';
/** Camera depth prepass for contact shadows: 32-bit, single-sampled, sampleable. */
const PREPASS_FORMAT: GPUTextureFormat = 'depth32float';
/** "Scene bounds" fit: a sphere around the props (the floor beyond it fades out anyway). */
const SCENE_BOUNDS: Sphere = { center: vec3.fromValues(0, 1, 0), radius: 5.5 };
const MODE = { VSM: 5, EVSM: 6 } as const;
/** Moment maps are pre-blurred, so they don't need the sharpest resolutions; this also caps their memory. */
const MAX_MOMENT_SIZE = 1024;
/** e^(2c) must stay finite in the moment format: ln(65504) / 2 for half floats, ~88.7 / 2 for floats. */
const EVSM_MAX_EXP = { 16: 5.54, 32: 42 } as const;
/** Smallest variance Chebyshev's bound may use (depth units²): hides the format's rounding noise. */
const MIN_VARIANCE = { 16: 1e-4, 32: 2e-6 } as const;

const create: ExperimentFactory = async (ctx) => {
  const { device, surface, params, reducedMotion } = ctx;
  const float32Filterable = ctx.gpu.caps.float32Filterable;
  const light = params.add(
    {
      azimuth: { value: 35, min: -180, max: 180, step: 1 },
      elevation: { value: 42, min: 5, max: 89, step: 1 },
      intensity: { value: 3.2, min: 0, max: 10, step: 0.05 },
      color: { value: '#fff1de', type: 'color' },
      orbitLight: { value: !reducedMotion, label: 'Orbit light' },
    },
    'Sun',
  );
  const shadow = params.add(
    {
      mode: {
        value: 4,
        options: { Off: 0, 'Hard (2×2 HW)': 1, 'PCF grid': 2, 'Poisson disc': 3, PCSS: 4, VSM: MODE.VSM, EVSM: MODE.EVSM },
      },
      mapSize: {
        value: 2048,
        options: { '512': 512, '1024': 1024, '2048': 2048, '4096': 4096 },
        label: 'Map size',
        hint: 'VSM and EVSM use at most 1024²: their moments are blurred anyway, and they cost up to 4 floats per texel.',
      },
      fit: {
        value: 'scene',
        options: { 'Scene bounds (fixed)': 'scene', 'Camera frustum (tight)': 'camera' },
        label: 'Frustum fit',
        hint: 'Scene bounds: one fixed sphere around the props. Camera frustum: a sphere around the visible slice out to "Shadow distance", which follows the camera.',
      },
      shadowDistance: {
        value: 12,
        min: 3,
        max: 40,
        step: 0.5,
        label: 'Shadow distance',
        hint: 'Camera fit only: how far from the camera shadows are drawn (nothing beyond is shadowed). Shorter = smaller texels.',
      },
      snap: {
        value: true,
        label: 'Texel snapping',
        hint: 'Moves the light frustum in whole shadow-map texels, so edges stay put while the camera moves (camera fit). It cannot help while the light itself turns.',
      },
      pcfRadius: { value: 2, min: 0, max: 6, step: 1, label: 'PCF radius', hint: 'Kernel radius in texels (PCF grid & Poisson).' },
      lightSize: { value: 2.5, min: 0.1, max: 12, step: 0.1, label: 'Sun size (°)', hint: 'Angular diameter used by PCSS. The real sun is ~0.53°.' },
      depthBias: { value: 0.0006, min: 0, max: 0.01, step: 0.0001, label: 'Depth bias' },
      normalOffset: { value: 1.5, min: 0, max: 6, step: 0.1, label: 'Normal offset', hint: 'In shadow-map texels.' },
    },
    'Shadows',
  );
  const vsm = params.add(
    {
      blurRadius: {
        value: 4,
        min: 0,
        max: 8,
        step: 1,
        label: 'Blur radius',
        hint: 'Separable Gaussian over the moments, in texels: 2 passes of 2r+1 taps, paid per map texel instead of per screen pixel.',
      },
      bleedReduce: {
        value: 0.2,
        min: 0,
        max: 0.95,
        step: 0.01,
        label: 'Bleed reduction',
        hint: 'Clips the low end of Chebyshev\'s bound. Removes light bleeding where shadows overlap, but over-darkens and hardens penumbrae.',
      },
      evsmExponent: {
        value: 40,
        min: 1,
        max: 42,
        step: 0.5,
        label: 'EVSM exponent',
        hint: 'Higher = less bleeding. Clamped to 5.54 with 16-bit moments, where e^2c would overflow.',
      },
      precision: {
        value: 32,
        options: { 'Float32 (if filterable)': 32, Float16: 16 },
        label: 'Moment format',
        hint: float32Filterable
          ? 'This device can filter 32-bit float textures.'
          : 'This device cannot filter 32-bit float textures, so moments are always 16-bit.',
      },
    },
    'Moments (VSM / EVSM)',
  );
  const contact = params.add(
    {
      contact: { value: false, label: 'Contact shadows', hint: 'Adds a depth prepass and a short screen-space march toward the sun.' },
      contactLength: { value: 0.35, min: 0.05, max: 1.5, step: 0.01, label: 'Ray length', hint: 'World units.' },
      contactSteps: { value: 12, min: 4, max: 32, step: 1, label: 'Steps' },
      contactThickness: {
        value: 0.2,
        min: 0.02,
        max: 1,
        step: 0.01,
        label: 'Thickness',
        hint: 'How deep each on-screen surface is assumed to be. Too thin misses hits; too thick casts shadows from silhouettes.',
      },
    },
    'Contact shadows',
  );
  const look = params.add(
    {
      background: { value: '#c9c5bf', type: 'color', hint: 'Page colour the floor fades into (display space, matched exactly).' },
      sky: { value: '#a9bcd6', type: 'color' },
      ground: { value: '#6b5d52', type: 'color' },
      ambient: { value: 0.28, min: 0, max: 1.5, step: 0.01 },
      antialiasing: {
        value: 'msaa',
        options: { 'MSAA 4×': 'msaa', TAA: 'taa', Off: 'off' },
        hint: 'TAA also accumulates the per-pixel Poisson/PCSS noise over frames, so soft shadows come out smooth.',
      },
      view: {
        value: 0,
        options: { Lit: 0, Visibility: 1, 'Light depth': 2, 'Shadow texels': 3, 'Contact mask': 4 },
        hint: 'Shadow texels: the map\'s texel grid on every surface (easiest to read at 512). Contact mask: the contact-shadow term alone.',
      },
      showMap: { value: false, label: 'Show shadow map' },
      animate: { value: !reducedMotion },
      camDrift: {
        value: false,
        label: 'Drift camera',
        hint: 'Slowly orbits the camera. With the camera fit and snapping off, watch shadow edges shimmer.',
      },
    },
    'Look & debug',
  );
  const post = await PostProcess.create(ctx, { bloom: 0.02, vignette: 0.12, grain: 0.02 });
  const taa = await TemporalAA.create(device);
  const blur = await SeparableBlur.create(device);

  // --- Scene ------------------------------------------------------------------
  const meshes = {
    ground: uploadMesh(device, plane(60, 1), 'ground'),
    box: uploadMesh(device, box(1, 1, 1), 'box'),
    sphere: uploadMesh(device, sphere(1, 64, 40), 'sphere'),
    torus: uploadMesh(device, torus(0.8, 0.26, 40, 128), 'torus'),
  };
  const T = (x: number, y: number, z: number) => mat4.translation([x, y, z]);
  const props = new PropSet(
    device,
    [
      { mesh: meshes.ground, color: '#d8cfc4', roughness: 0.9 },
      {
        mesh: meshes.torus,
        // Polished copper.
        color: '#f2b08a',
        roughness: 0.32,
        metallic: 1,
        transform: (t) => mat4.rotateX(mat4.rotateY(T(0, 1.15, 0), t * 0.35), Math.PI / 2),
      },
      { mesh: meshes.sphere, color: '#c95f4b', roughness: 0.45, transform: () => mat4.scale(T(-1.9, 0.6, 0.9), [0.6, 0.6, 0.6]) },
      { mesh: meshes.sphere, color: '#e8e2d6', roughness: 0.65, transform: () => mat4.scale(T(-0.7, 0.35, 2.1), [0.35, 0.35, 0.35]) },
      {
        mesh: meshes.sphere,
        color: '#7e9cc4',
        roughness: 0.15,
        transform: (t) => mat4.scale(T(0.4, 2.4 + 0.25 * Math.sin(t * 0.8), -1.9), [0.38, 0.38, 0.38]),
      },
      { mesh: meshes.box, color: '#a9b49f', roughness: 0.8, transform: () => mat4.scale(T(1.9, 1.1, -0.8), [0.6, 2.2, 0.6]) },
      {
        mesh: meshes.box,
        color: '#3f4a5c',
        roughness: 0.5,
        transform: () => mat4.scale(mat4.rotateY(T(1.7, 0.2, 1.5), 0.6), [1.2, 0.4, 1.2]),
      },
      // A thin pole: the textbook demo of contact-hardening (sharp at the base, soft at the tip).
      { mesh: meshes.box, color: '#9a9ca3', roughness: 0.35, metallic: 1, transform: () => mat4.scale(T(-1.7, 1.5, -1.6), [0.07, 3, 0.07]) },
      // A few small cubes: detail a low-resolution shadow map loses, which contact shadows recover.
      ...[
        [0.55, 2.6, 0.16, 0.3],
        [0.95, 2.35, 0.12, 1.1],
        [0.8, 2.95, 0.1, 0.7],
      ].map(([x, z, s, r]) => ({
        mesh: meshes.box,
        color: '#e3d6c1',
        roughness: 0.55,
        transform: () => mat4.scale(mat4.rotateY(T(x, s / 2, z), r), [s, s, s]),
      })),
    ],
    'shadow-map-pcf:props',
  );

  const frame = new UniformBlock(device, 'Frame', {
    viewProj: 'mat4x4f',
    lightViewProj: 'mat4x4f',
    cameraPos: 'vec3f',
    fadeStart: 'f32',
    lightDir: 'vec3f',
    shadowMode: 'u32',
    lightColor: 'vec4f',
    skyColor: 'vec4f',
    groundColor: 'vec4f',
    evsmExps: 'vec2f',
    screenSize: 'vec2f',
    fadeEnd: 'f32',
    pcfRadius: 'f32',
    lightTan: 'f32',
    shadowExtent: 'f32',
    shadowDepthRange: 'f32',
    shadowTexelWorld: 'f32',
    mapSize: 'f32',
    depthBias: 'f32',
    normalOffset: 'f32',
    debugView: 'u32',
    noiseOffset: 'f32',
    minVariance: 'f32',
    bleedReduce: 'f32',
    contact: 'u32',
    contactLength: 'f32',
    contactThickness: 'f32',
    contactSteps: 'f32',
    camNear: 'f32',
    camFar: 'f32',
  });

  // --- Pipelines --------------------------------------------------------------
  const module = await createShader(
    device,
    'shadow-map-pcf',
    wgsl(
      commonWgsl,
      fullscreenWgsl,
      MESH_VERTEX_WGSL,
      PROP_WGSL,
      frame.wgslStruct,
      shadowWgsl,
      shadowMomentsWgsl,
      contactShadowWgsl,
      brdfWgsl,
      shaderWgsl,
    ),
  );

  const bindGroupLayout = device.createBindGroupLayout({
    label: 'shadow-map-pcf',
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
    ],
  });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
  // Light and depth-prepass passes can't bind the targets they write, so they get their own layout.
  const geometryBindGroupLayout = device.createBindGroupLayout({
    label: 'shadow-map-pcf:geometry',
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
    ],
  });
  const geometryLayout = device.createPipelineLayout({ bindGroupLayouts: [geometryBindGroupLayout] });
  const depthOnly = (label: string, entryPoint: string, format: GPUTextureFormat) =>
    device.createRenderPipelineAsync({
      label,
      layout: geometryLayout,
      vertex: { module, entryPoint, buffers: [MESH_VERTEX_LAYOUT] },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { format, depthWriteEnabled: true, depthCompare: 'less' },
    });
  // Light pass writing VSM/EVSM moments next to depth. rg for VSM, rgba for EVSM; 32-bit only if filterable.
  const momentFormats: GPUTextureFormat[] = ['rg16float', 'rgba16float', ...(float32Filterable ? (['rg32float', 'rgba32float'] as const) : [])];
  const momentPipeline = (format: GPUTextureFormat) =>
    device.createRenderPipelineAsync({
      label: `moments-pass:${format}`,
      layout: geometryLayout,
      vertex: { module, entryPoint: 'vs_shadow', buffers: [MESH_VERTEX_LAYOUT] },
      fragment: { module, entryPoint: 'fs_moments', targets: [{ format }] },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { format: SHADOW_FORMAT, depthWriteEnabled: true, depthCompare: 'less' },
    });

  // Main + overlay pipelines exist in a multisampled and a single-sampled flavour
  // (TAA needs a sampleable, single-sampled depth buffer).
  const makeScenePipelines = (count: number) =>
    Promise.all([
      device.createRenderPipelineAsync({
        label: `main-pass-x${count}`,
        layout,
        vertex: { module, entryPoint: 'vs_main', buffers: [MESH_VERTEX_LAYOUT] },
        fragment: { module, entryPoint: 'fs_main', targets: [{ format: HDR_FORMAT }] },
        primitive: { topology: 'triangle-list', cullMode: 'back' },
        depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: 'less' },
        multisample: { count },
      }),
      device.createRenderPipelineAsync({
        label: `shadow-overlay-x${count}`,
        layout,
        vertex: { module, entryPoint: 'vs_fullscreen' },
        fragment: { module, entryPoint: 'fs_overlay', targets: [{ format: HDR_FORMAT }] },
        depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: 'always' },
        multisample: { count },
      }),
    ]);
  const [shadowPipeline, prepassPipeline, momentPipelineList, msaaPipelines, singlePipelines] = await Promise.all([
    depthOnly('shadow-pass', 'vs_shadow', SHADOW_FORMAT),
    depthOnly('depth-prepass', 'vs_depth', PREPASS_FORMAT),
    Promise.all(momentFormats.map(momentPipeline)),
    makeScenePipelines(MSAA_SAMPLES),
    makeScenePipelines(1),
  ]);
  const momentPipelines = new Map(momentFormats.map((f, i) => [f, momentPipelineList[i]]));

  const shadowSampler = device.createSampler({
    label: 'shadow-compare',
    compare: 'less-equal',
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  });
  const momentSampler = device.createSampler({
    label: 'shadow-moments',
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  });
  // Stand-ins for the optional bindings when their feature is off.
  const placeholder = (format: GPUTextureFormat) =>
    device.createTexture({ label: `placeholder:${format}`, size: [1, 1], format, usage: GPUTextureUsage.TEXTURE_BINDING });
  const noMoments = placeholder('rg16float');
  const noDepth = placeholder(PREPASS_FORMAT);

  const geometryBindGroup = device.createBindGroup({
    layout: geometryBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: frame.buffer } },
      { binding: 1, resource: { buffer: props.buffer } },
    ],
  });

  // The main bind group is rebuilt whenever one of its textures is (re)created.
  let mainBindGroup: GPUBindGroup | null = null;
  let bound: GPUTexture[] = [];
  const mainBindings = (shadowMap: GPUTexture, moments: GPUTexture, sceneDepth: GPUTexture) => {
    if (mainBindGroup && bound[0] === shadowMap && bound[1] === moments && bound[2] === sceneDepth) return mainBindGroup;
    bound = [shadowMap, moments, sceneDepth];
    mainBindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: frame.buffer } },
        { binding: 1, resource: { buffer: props.buffer } },
        { binding: 2, resource: shadowMap.createView() },
        { binding: 3, resource: shadowSampler },
        { binding: 4, resource: moments.createView() },
        { binding: 5, resource: momentSampler },
        { binding: 6, resource: sceneDepth.createView() },
      ],
    });
    return mainBindGroup;
  };

  const targets = new TargetPool(device);

  const camera = new OrbitCamera(surface.canvas, { target: [0, 0.9, 0], distance: 9.5, yaw: 0.75, pitch: 0.42, maxDistance: 25 });
  let animTime = 0;
  let lightSpin = 0;

  return {
    frame({ time, dt, size }) {
      if (look.animate) animTime += dt;
      if (light.orbitLight) lightSpin += dt * 0.15;
      camera.autoRotate = look.camDrift ? 0.12 : 0;
      const aspect = size.width / size.height;
      camera.update(dt, aspect);
      const fullSize = [size.width, size.height] as const;
      const aa = look.antialiasing;
      const samples = aa === 'msaa' ? MSAA_SAMPLES : 1;
      const [mainPipeline, overlayPipeline] = samples > 1 ? msaaPipelines : singlePipelines;
      const scene = post.scene(size);
      const msaaColor = samples > 1 ? targets.get('msaa', { size: fullSize, format: HDR_FORMAT, sampleCount: samples }) : null;
      const depth = targets.get(`depth-x${samples}`, {
        size: fullSize,
        format: DEPTH_FORMAT,
        sampleCount: samples,
        usage: samples > 1 ? GPUTextureUsage.RENDER_ATTACHMENT : GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      if (aa !== 'taa') taa.reset();
      const viewProj = aa === 'taa' ? taa.jitter(camera.viewProjection, size) : camera.viewProjection;

      // Shadow-map targets. Moment modes add a colour target (and cap the resolution).
      const momentMode = shadow.mode === MODE.VSM || shadow.mode === MODE.EVSM;
      const bits = vsm.precision === 32 && float32Filterable ? 32 : 16;
      const mapSize = momentMode ? Math.min(shadow.mapSize, MAX_MOMENT_SIZE) : shadow.mapSize;
      const shadowMap = targets.get('shadow-map', { size: [mapSize, mapSize], format: SHADOW_FORMAT });
      let moments: GPUTexture = noMoments;
      if (momentMode) {
        const format: GPUTextureFormat = `${shadow.mode === MODE.VSM ? 'rg' : 'rgba'}${bits}float`;
        moments = targets.get('moments', { size: [mapSize, mapSize], format }).texture;
      } else {
        targets.release('moments');
        blur.trim();
      }
      let sceneDepth: GPUTexture = noDepth;
      if (contact.contact) sceneDepth = targets.get('depth-prepass', { size: fullSize, format: PREPASS_FORMAT }).texture;
      else targets.release('depth-prepass');

      props.update(animTime);

      // Light: direction toward the sun, then an orthographic frustum fitted around either the
      // props or the slice of the camera frustum that gets shadows.
      const az = (light.azimuth * Math.PI) / 180 + lightSpin;
      const el = (light.elevation * Math.PI) / 180;
      const toSun = vec3.fromValues(Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az));
      const bounds =
        shadow.fit === 'camera'
          ? frustumSliceSphere(mat4.inverse(camera.view), camera.fovY, aspect, camera.near, shadow.shadowDistance)
          : SCENE_BOUNDS;
      const fit = fitDirectionalShadow({ toLight: toSun, bounds, mapSize, snap: shadow.snap });

      const evsmPos = Math.min(vsm.evsmExponent, EVSM_MAX_EXP[bits]);
      const evsmExps = [evsmPos, Math.min(evsmPos, bits === 32 ? 10 : EVSM_MAX_EXP[16])];
      const lc = hexToLinear(light.color);
      const sky = hexToLinear(look.sky);
      const ground = hexToLinear(look.ground);
      const background = hexToLinear(look.background);

      frame
        .set('viewProj', viewProj)
        .set('lightViewProj', fit.viewProj)
        .set('cameraPos', camera.position)
        .set('lightDir', vec3.negate(toSun))
        .set('shadowMode', shadow.mode)
        .set('lightColor', [lc[0] * light.intensity, lc[1] * light.intensity, lc[2] * light.intensity, 0])
        .set('skyColor', [sky[0] * look.ambient, sky[1] * look.ambient, sky[2] * look.ambient, 0])
        .set('groundColor', [ground[0] * look.ambient, ground[1] * look.ambient, ground[2] * look.ambient, 0])
        .set('evsmExps', evsmExps)
        .set('screenSize', fullSize)
        .set('fadeStart', 6)
        .set('fadeEnd', 22)
        .set('pcfRadius', shadow.pcfRadius)
        .set('lightTan', Math.tan(((shadow.lightSize / 2) * Math.PI) / 180))
        .set('shadowExtent', fit.halfExtent)
        .set('shadowDepthRange', fit.depthRange)
        .set('shadowTexelWorld', fit.texelWorld)
        .set('mapSize', mapSize)
        .set('depthBias', shadow.depthBias)
        .set('normalOffset', shadow.normalOffset)
        .set('debugView', look.view)
        .set('noiseOffset', aa === 'taa' ? taa.frameNoiseOffset : 0)
        .set('minVariance', MIN_VARIANCE[bits])
        .set('bleedReduce', vsm.bleedReduce)
        .set('contact', contact.contact ? 1 : 0)
        .set('contactLength', contact.contactLength)
        .set('contactThickness', contact.contactThickness)
        .set('contactSteps', contact.contactSteps)
        .set('camNear', camera.near)
        .set('camFar', camera.far)
        .upload();

      const encoder = device.createCommandEncoder({ label: 'shadow-map-pcf' });

      // 1. Shadow map: depth only, or depth + moments (then blurred in place).
      if (momentMode) {
        // Clear to the moments of the far plane (depth 1), i.e. "nothing in the way".
        const [p, n] = evsmExps;
        const far = shadow.mode === MODE.VSM ? [1, 1, 0, 0] : [Math.exp(p), Math.exp(2 * p), -Math.exp(-n), Math.exp(-2 * n)];
        const mp = encoder.beginRenderPass({
          label: 'moments-pass',
          colorAttachments: [{ view: moments.createView(), loadOp: 'clear', storeOp: 'store', clearValue: far }],
          depthStencilAttachment: { view: shadowMap.view, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
        });
        mp.setPipeline(momentPipelines.get(moments.format)!);
        mp.setBindGroup(0, geometryBindGroup);
        props.draw(mp, { castersOnly: true });
        mp.end();
        blur.run(encoder, moments, vsm.blurRadius);
      } else {
        const sp = encoder.beginRenderPass({
          label: 'shadow-pass',
          colorAttachments: [],
          depthStencilAttachment: { view: shadowMap.view, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
        });
        sp.setPipeline(shadowPipeline);
        sp.setBindGroup(0, geometryBindGroup);
        props.draw(sp, { castersOnly: true });
        sp.end();
      }

      // 2. Optional camera depth prepass, single-sampled so the main pass can read it.
      if (contact.contact) {
        const dp = encoder.beginRenderPass({
          label: 'depth-prepass',
          colorAttachments: [],
          depthStencilAttachment: { view: sceneDepth.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
        });
        dp.setPipeline(prepassPipeline);
        dp.setBindGroup(0, geometryBindGroup);
        props.draw(dp);
        dp.end();
      }

      // 3. Main pass.
      const mp = encoder.beginRenderPass({
        label: 'main-pass',
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
      mp.setPipeline(mainPipeline);
      mp.setBindGroup(0, mainBindings(shadowMap.texture, moments, sceneDepth));
      props.draw(mp);

      if (look.showMap) {
        const s = Math.round(Math.min(size.width, size.height) * 0.3);
        const m = Math.round(16 * size.pixelRatio);
        mp.setViewport(m, size.height - s - m, s, s, 0, 1);
        mp.setPipeline(overlayPipeline);
        mp.draw(3);
      }
      mp.end();

      const source =
        aa === 'taa' ? taa.resolve(encoder, { color: scene.texture, depth: depth.texture, viewProj: camera.viewProjection }) : undefined;
      post.render(encoder, surface.currentTexture().createView(), { time, raw: look.view !== 0, background, source });
      device.queue.submit([encoder.finish()]);
    },
    dispose() {
      camera.destroy();
      targets.destroy();
      post.destroy();
      taa.destroy();
      blur.destroy();
      props.destroy();
      frame.destroy();
      noMoments.destroy();
      noDepth.destroy();
      Object.values(meshes).forEach((m) => m.destroy());
    },
  };
};

export default create;
