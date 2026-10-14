import { mat4 } from 'wgpu-matrix';
import commonWgsl from '../../shaders/common.wgsl?raw';
import fullscreenWgsl from '../../shaders/fullscreen.wgsl?raw';
import brdfWgsl from '../../shaders/brdf.wgsl?raw';
import screenSpaceWgsl from '../../shaders/screen-space.wgsl?raw';
import gtaoWgsl from '../../shaders/gtao.wgsl?raw';
import { MESH_VERTEX_LAYOUT, MESH_VERTEX_WGSL, box, plane, sphere, torus, uploadMesh } from '../../gpu/geometry';
import { createShader, wgsl } from '../../gpu/shader';
import { UniformBlock } from '../../gpu/uniforms';
import { hexToLinear } from '../../gpu/color';
import { TargetPool } from '../../gpu/targets';
import { parseHdr } from '../../ibl/hdr';
import { Ibl, iblWgsl } from '../../ibl/ibl';
import { HDR_FORMAT, PostProcess } from '../../post/post';
import { TemporalAA } from '../../post/taa';
import { GBuffer, screenCamera } from '../../post/gbuffer';
import { AO_METHOD, GtaoPass } from '../../post/gtao';
import { TemporalAccumulator } from '../../post/temporal-accumulate';
import { BilateralFilter } from '../../post/bilateral';
import { OrbitCamera } from '../../scene/orbit-camera';
import { PROP_WGSL, PropSet, type Prop } from '../../scene/props';
import { lathe } from '../../scene/shapes';
import type { ExperimentFactory } from '../types';
import shaderWgsl from './shader.wgsl?raw';

const ALBEDO_FORMAT: GPUTextureFormat = 'rgba8unorm-srgb';
/** History-length texture for the debug view while accumulation is off. */
const NO_HISTORY_FORMAT: GPUTextureFormat = 'r16float';

type EnvChoice = 'studio' | 'sky' | 'hdri';
const asset = (path: string) => new URL(path, document.baseURI).href;

const create: ExperimentFactory = async (ctx) => {
  const { device, surface, params, reducedMotion } = ctx;

  const ao = params.add(
    {
      method: {
        value: AO_METHOD.GTAO,
        options: { 'GTAO (horizon-based)': AO_METHOD.GTAO, 'Classic SSAO (hemisphere)': AO_METHOD.SSAO },
        hint: 'GTAO integrates the visible arc between two horizons per slice, in closed form. SSAO counts random hemisphere points that land behind the depth buffer.',
      },
      resolution: {
        value: 0.5,
        options: { Full: 1, Half: 0.5 },
        hint: 'Half computes AO for a quarter of the pixels, then a depth/normal-aware upsample brings it back to full resolution.',
      },
      radius: { value: 1, min: 0.1, max: 3, step: 0.05, hint: 'World-space search radius (metres in this scene).' },
      falloff: {
        value: 0.5,
        min: 0.05,
        max: 1,
        step: 0.05,
        hint: 'GTAO: fraction of the radius over which distant occluders fade out, so the radius never shows as a ring.',
      },
      slices: { value: 2, min: 1, max: 4, step: 1, hint: 'GTAO: directions per pixel per frame. Temporal accumulation rotates them every frame.' },
      steps: { value: 6, min: 2, max: 16, step: 1, hint: 'GTAO: horizon samples per side of each slice. Cost is slices × steps × 2 texel reads per AO pixel.' },
      samples: { value: 16, min: 4, max: 32, step: 1, hint: 'SSAO: hemisphere points per pixel.' },
      bias: { value: 0.03, min: 0, max: 0.2, step: 0.005, hint: 'SSAO: depth bias in metres against self-occlusion on flat surfaces.' },
      maxRadius: {
        value: 160,
        min: 16,
        max: 400,
        step: 8,
        label: 'Max radius (px)',
        hint: 'Caps the on-screen search radius (full-resolution pixels), so close-ups don\'t turn every tap into a cache miss.',
      },
      power: { value: 1.5, min: 0.5, max: 3, step: 0.05, label: 'Strength', hint: 'Exponent on visibility (artistic; 1 = as computed).' },
    },
    'Ambient occlusion',
  );
  const denoise = params.add(
    {
      temporal: {
        value: true,
        label: 'Temporal accumulation',
        hint: 'Reprojects last frame\'s AO and keeps a running average per pixel (reset where the depth no longer matches). Off shows one frame\'s raw noise.',
      },
      frames: { value: 16, min: 2, max: 64, step: 1, label: 'History frames', hint: 'Cap on the running average: more = smoother, slower to react.' },
      spatial: { value: true, label: 'Spatial denoise', hint: 'Depth- and normal-aware Gaussian over the AO buffer.' },
      blurRadius: { value: 2, min: 1, max: 4, step: 1, label: 'Denoise radius', hint: 'In AO pixels: (2r+1)² taps.' },
    },
    'Denoise',
  );
  const light = params.add(
    {
      environment: {
        value: 'studio',
        options: { 'Studio (procedural)': 'studio', 'Overcast sky': 'sky', 'Studio HDRI': 'hdri' },
      },
      envIntensity: { value: 1, min: 0, max: 4, step: 0.01, label: 'Env intensity' },
      envRotation: { value: 30, min: -180, max: 180, step: 1, label: 'Env rotation' },
      bentNormals: {
        value: true,
        label: 'Bent normals',
        hint: 'Look up diffuse irradiance along the average *unoccluded* direction instead of the normal: light comes from where the opening is.',
      },
      bentStrength: { value: 1, min: 0, max: 1, step: 0.05, label: 'Bent amount' },
      multiBounce: {
        value: true,
        label: 'Multi-bounce',
        hint: 'Jimenez\'s fit: light bounces around inside creases, so bright surfaces darken less than AO alone says.',
      },
      specOcclusion: { value: true, label: 'Specular occlusion', hint: 'Darken reflections in occluded areas too (Lagarde\'s approximation from AO).' },
    },
    'Lighting',
  );
  const look = params.add(
    {
      backdrop: { value: 'page', options: { 'Page colour': 'page', Environment: 'env' } },
      background: { value: '#d6d0c7', type: 'color', hint: 'Page colour the floor fades into (display space, matched exactly).' },
      antialiasing: { value: 'taa', options: { TAA: 'taa', Off: 'off' } },
      view: {
        value: 0,
        options: {
          Lit: 0,
          'Lit without AO': 1,
          'AO only': 2,
          'Bent normals': 3,
          'Bent vs normal (angle)': 4,
          'G-buffer normals': 5,
          'History length': 6,
        },
        hint: 'Bent vs normal: how far occlusion tilts the lookup direction. History length: red where accumulation just restarted (disocclusion).',
      },
      drift: { value: !reducedMotion, label: 'Drift camera', hint: 'Slow orbit: a test of the reprojection (watch for smearing along edges).' },
    },
    'Look & debug',
  );
  const post = await PostProcess.create(ctx, { exposure: 0.85, bloom: 0.02, vignette: 0.15, grain: 0.015 });
  const [taa, ibl, gbuffer, aoPass, history, bilateral] = await Promise.all([
    TemporalAA.create(device),
    Ibl.create(device),
    GBuffer.create(device),
    GtaoPass.create(device),
    TemporalAccumulator.create(device),
    BilateralFilter.create(device),
  ]);
  const targets = new TargetPool(device);

  // --- Scene: a sunken gallery corner full of creases and contacts --------------------------
  const meshes = {
    floor: uploadMesh(device, plane(40, 1), 'gtao:floor'),
    box: uploadMesh(device, box(1, 1, 1), 'gtao:box'),
    sphere: uploadMesh(device, sphere(1, 64, 40), 'gtao:sphere'),
    torus: uploadMesh(device, torus(0.55, 0.18, 32, 96), 'gtao:torus'),
    column: uploadMesh(
      device,
      lathe(
        [
          [0, 0],
          [0.24, 0],
          [0.24, 0.1],
          [0.2, 0.16],
          [0.17, 0.24],
          [0.15, 2.3],
          [0.19, 2.38],
          [0.23, 2.46],
          [0, 2.46],
        ],
        { segments: 40 },
      ),
      'gtao:column',
    ),
  };
  const T = (x: number, y: number, z: number) => mat4.translation([x, y, z]);
  /** A box of size (sx, sy, sz) resting on height y0, turned by `ry` radians. */
  const block = (x: number, y0: number, z: number, sx: number, sy: number, sz: number, ry = 0) =>
    mat4.scale(mat4.rotateY(T(x, y0 + sy / 2, z), ry), [sx, sy, sz]);
  const plaster = '#e6dfd4';
  const props: Prop[] = [
    { mesh: meshes.floor, color: '#d8cfc3', roughness: 0.9 },
    // Two walls meeting in a corner.
    { mesh: meshes.box, color: plaster, roughness: 0.85, transform: () => block(0.6, 0, -3.1, 10.8, 3.6, 0.2) },
    { mesh: meshes.box, color: plaster, roughness: 0.85, transform: () => block(-4.6, 0, -0.1, 0.2, 3.6, 6.2) },
    // Stairs climbing into the corner.
    ...[0, 1, 2, 3, 4].map((i) => ({
      mesh: meshes.box,
      color: '#cfc5b8',
      roughness: 0.8,
      transform: () => block(-3.85, 0, -2.8 + 0.4 * (4 - i), 1.3, 0.25 * (i + 1), 0.4),
    })),
    // A big ball against the side wall.
    { mesh: meshes.sphere, color: '#4d6a86', roughness: 0.55, transform: () => mat4.scale(T(-3.8, 0.7, 1.0), [0.7, 0.7, 0.7]) },
    // Stepped plinth with a terracotta ball on top.
    { mesh: meshes.box, color: '#bfb3a3', roughness: 0.7, transform: () => block(-0.6, 0, -0.7, 1.7, 0.28, 1.7) },
    { mesh: meshes.box, color: '#bfb3a3', roughness: 0.7, transform: () => block(-0.6, 0.28, -0.7, 1.25, 0.28, 1.25) },
    { mesh: meshes.box, color: '#bfb3a3', roughness: 0.7, transform: () => block(-0.6, 0.56, -0.7, 0.8, 0.28, 0.8) },
    { mesh: meshes.sphere, color: '#c4673f', roughness: 0.5, transform: () => mat4.scale(T(-0.6, 1.24, -0.7), [0.4, 0.4, 0.4]) },
    // A brass ring lying on the floor.
    { mesh: meshes.torus, color: '#e3b46c', roughness: 0.32, metallic: 1, transform: () => T(1.25, 0.18, 1.0) },
    // A colonnade along the back wall.
    ...[-2.4, 0.15].flatMap((x) => [
      { mesh: meshes.box, color: plaster, roughness: 0.75, transform: () => block(x, 0, -2.45, 0.62, 0.16, 0.62) },
      { mesh: meshes.column, color: '#eee8de', roughness: 0.6, transform: () => T(x, 0.16, -2.45) },
      { mesh: meshes.box, color: plaster, roughness: 0.75, transform: () => block(x, 2.62, -2.45, 0.62, 0.14, 0.62) },
    ]),
    // A fluted panel: thin slats with narrow gaps, the hardest case for screen-space AO.
    ...Array.from({ length: 13 }, (_, i) => ({
      mesh: meshes.box,
      color: '#d9cdbd',
      roughness: 0.7,
      transform: () => block(1.4 + i * 0.22, 0, -2.94, 0.12, 2.9, 0.12),
    })),
    // A pile of crates.
    { mesh: meshes.box, color: '#c8a27c', roughness: 0.75, transform: () => block(2.6, 0, -1.4, 0.8, 0.8, 0.8, 0.25) },
    { mesh: meshes.box, color: '#93a283', roughness: 0.7, transform: () => block(2.55, 0.8, -1.35, 0.5, 0.5, 0.5, 0.8) },
    { mesh: meshes.box, color: '#c8a27c', roughness: 0.75, transform: () => block(3.35, 0, -0.75, 0.42, 0.42, 0.42, -0.35) },
    { mesh: meshes.box, color: '#e0d6c6', roughness: 0.75, transform: () => block(1.85, 0, -0.85, 0.55, 0.55, 0.55, 0.7) },
  ];
  const propSet = new PropSet(device, props, 'gtao:props');
  propSet.update(0);

  const frame = new UniformBlock(device, 'Frame', {
    viewProj: 'mat4x4f',
    view: 'mat4x4f',
    invView: 'mat4x4f',
    cameraPos: 'vec3f',
    near: 'f32',
    projScale: 'vec2f',
    jitter: 'vec2f',
    far: 'f32',
    aoPower: 'f32',
    bentStrength: 'f32',
    bentNormals: 'u32',
    multiBounce: 'u32',
    specOcclusion: 'u32',
    debugView: 'u32',
    backdrop: 'u32',
    backdropBlur: 'f32',
    fadeStart: 'f32',
    fadeEnd: 'f32',
    maxFrames: 'f32',
  });

  // --- Pipelines ----------------------------------------------------------------------------
  const module = await createShader(
    device,
    'gtao-scene',
    wgsl(
      commonWgsl,
      fullscreenWgsl,
      MESH_VERTEX_WGSL,
      PROP_WGSL,
      frame.wgslStruct,
      brdfWgsl,
      ibl.wgslStruct,
      iblWgsl,
      screenSpaceWgsl,
      gtaoWgsl,
      shaderWgsl,
    ),
  );
  const geometryLayout = device.createBindGroupLayout({
    label: 'gtao:geometry',
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    ],
  });
  const tex = (binding: number, sampleType: GPUTextureSampleType = 'unfilterable-float'): GPUBindGroupLayoutEntry => ({
    binding,
    visibility: GPUShaderStage.FRAGMENT,
    texture: { sampleType },
  });
  const compositeLayout = device.createBindGroupLayout({
    label: 'gtao:composite',
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      tex(2, 'depth'),
      tex(3),
      tex(4),
      tex(5),
      tex(6),
    ],
  });
  const [geometryPipeline, compositePipeline] = await Promise.all([
    device.createRenderPipelineAsync({
      label: 'gtao:geometry',
      layout: device.createPipelineLayout({ bindGroupLayouts: [geometryLayout] }),
      vertex: { module, entryPoint: 'vs_geometry', buffers: [MESH_VERTEX_LAYOUT] },
      fragment: { module, entryPoint: 'fs_geometry', targets: [GBuffer.normalTarget, { format: ALBEDO_FORMAT }] },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: GBuffer.depthStencil,
    }),
    device.createRenderPipelineAsync({
      label: 'gtao:composite',
      layout: device.createPipelineLayout({ bindGroupLayouts: [compositeLayout, ibl.bindGroupLayout] }),
      vertex: { module, entryPoint: 'vs_fullscreen' },
      fragment: { module, entryPoint: 'fs_composite', targets: [{ format: HDR_FORMAT }] },
    }),
  ]);
  const geometryGroup = device.createBindGroup({
    layout: geometryLayout,
    entries: [
      { binding: 0, resource: { buffer: frame.buffer } },
      { binding: 1, resource: { buffer: propSet.buffer } },
    ],
  });
  const noHistory = device.createTexture({ label: 'gtao:no-history', size: [1, 1], format: NO_HISTORY_FORMAT, usage: GPUTextureUsage.TEXTURE_BINDING });

  // --- Environment ----------------------------------------------------------------------------
  let disposed = false;
  let envRequest = 0;
  let hdri: ReturnType<typeof parseHdr> | null = null;
  const loadEnvironment = async (choice: EnvChoice) => {
    const id = ++envRequest;
    if (disposed) return;
    if (choice === 'hdri') {
      hdri ??= parseHdr(await (await fetch(asset('assets/hdri/studio_small_09_1k.hdr'))).arrayBuffer());
      if (id !== envRequest || disposed) return;
      await ibl.setEquirect(hdri);
    } else if (choice === 'sky') {
      // A high, hazy sun: mostly soft skylight, where AO is the only cue to shape.
      await ibl.setProcedural({ kind: 'sky', sunAzimuth: 0.6, sunElevation: 1.1, exposure: 0.8 });
    } else {
      await ibl.setProcedural({ kind: 'studio' });
    }
  };
  await loadEnvironment(light.environment as EnvChoice);
  let currentEnv = light.environment;

  const camera = new OrbitCamera(surface.canvas, {
    target: [-0.3, 0.75, -0.8],
    distance: 9.8,
    yaw: 0.5,
    pitch: 0.36,
    minPitch: 0.02,
    minDistance: 2,
    maxDistance: 20,
  });
  let frameIndex = 0;
  let settingsKey = '';

  return {
    frame({ time, dt, size }) {
      if (light.environment !== currentEnv) {
        currentEnv = light.environment;
        void loadEnvironment(currentEnv as EnvChoice).catch((err) => console.error(err));
      }
      camera.autoRotate = look.drift ? 0.06 : 0;
      camera.update(dt, size.width / size.height);
      ibl.rotation = (light.envRotation * Math.PI) / 180;
      ibl.intensity = light.envIntensity;
      ibl.update();

      const useTaa = look.antialiasing === 'taa';
      if (!useTaa) taa.reset();
      const jitterPixels: [number, number] = useTaa ? taa.jitterPixels : [0, 0];
      const viewProj = useTaa ? taa.jitter(camera.viewProjection, size) : camera.viewProjection;
      const cam = screenCamera(camera.projection, camera.near, camera.far, size, jitterPixels);
      const scale = ao.resolution === 1 ? 1 : 0.5;

      // Restart the AO history when the signal itself changes meaning.
      const key = `${ao.method}:${scale}:${ao.radius}:${ao.falloff}:${ao.slices}:${ao.steps}:${ao.samples}:${ao.bias}:${ao.maxRadius}`;
      if (key !== settingsKey) {
        settingsKey = key;
        history.reset();
      }
      if (denoise.temporal) frameIndex++;

      frame
        .set('viewProj', viewProj)
        .set('view', camera.view)
        .set('invView', mat4.inverse(camera.view))
        .set('cameraPos', camera.position)
        .set('near', camera.near)
        .set('far', camera.far)
        .set('projScale', cam.projScale)
        .set('jitter', cam.jitter)
        .set('aoPower', ao.power)
        .set('bentStrength', light.bentStrength)
        .set('bentNormals', light.bentNormals ? 1 : 0)
        .set('multiBounce', light.multiBounce ? 1 : 0)
        .set('specOcclusion', light.specOcclusion ? 1 : 0)
        .set('debugView', look.view)
        .set('backdrop', look.backdrop === 'env' ? 1 : 0)
        .set('backdropBlur', 0.55)
        .set('fadeStart', 7)
        .set('fadeEnd', 15)
        .set('maxFrames', denoise.frames)
        .upload();

      const encoder = device.createCommandEncoder({ label: 'gtao' });

      // 1. G-buffer: view normal + roughness, albedo + metallic, depth.
      const gb = gbuffer.attachments(size);
      const albedo = targets.get('albedo', { size: [size.width, size.height], format: ALBEDO_FORMAT });
      const gp = encoder.beginRenderPass({
        label: 'gtao:geometry',
        colorAttachments: [gb.normal, { view: albedo.view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }],
        depthStencilAttachment: gb.depth,
      });
      gp.setPipeline(geometryPipeline);
      gp.setBindGroup(0, geometryGroup);
      propSet.draw(gp);
      gp.end();

      // 2. AO at full or half resolution: pack geometry, run the kernel, accumulate, denoise, upsample.
      const geo = gbuffer.resolve(encoder, scale, cam);
      let signal = aoPass.run(encoder, geo.current, cam, {
        method: ao.method,
        radius: ao.radius,
        falloff: ao.falloff,
        slices: ao.slices,
        steps: ao.method === AO_METHOD.SSAO ? ao.samples : ao.steps,
        bias: ao.bias,
        maxRadiusPx: ao.maxRadius * scale,
        noiseOffset: (frameIndex % 64) * 5.588238,
      });
      if (denoise.temporal) {
        signal = history.run(encoder, {
          signal,
          geometry: geo.current,
          previousGeometry: geo.previous,
          cam,
          view: camera.view,
          viewProj,
          maxFrames: denoise.frames,
        });
      } else {
        history.trim();
      }
      if (denoise.spatial) signal = bilateral.denoise(encoder, signal, geo.current, { radius: denoise.blurRadius });
      if (scale < 1) signal = bilateral.upsample(encoder, signal, geo.current, gbuffer, cam);

      // 3. Composite: environment lighting through the AO and bent normals.
      const compositeGroup = device.createBindGroup({
        layout: compositeLayout,
        entries: [
          { binding: 0, resource: { buffer: frame.buffer } },
          { binding: 2, resource: gbuffer.depth.createView() },
          { binding: 3, resource: gbuffer.normal.createView() },
          { binding: 4, resource: albedo.view },
          { binding: 5, resource: signal.createView() },
          { binding: 6, resource: (history.historyLength ?? noHistory).createView() },
        ],
      });
      const sceneTarget = post.scene(size);
      const cp = encoder.beginRenderPass({
        label: 'gtao:composite',
        colorAttachments: [{ view: sceneTarget.view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }],
      });
      cp.setPipeline(compositePipeline);
      cp.setBindGroup(0, compositeGroup);
      cp.setBindGroup(1, ibl.bindGroup);
      cp.draw(3);
      cp.end();

      const source = useTaa
        ? taa.resolve(encoder, { color: sceneTarget.texture, depth: gbuffer.depth, viewProj: camera.viewProjection })
        : undefined;
      post.render(encoder, surface.currentTexture().createView(), {
        time,
        raw: look.view >= 2,
        background: hexToLinear(look.background),
        source,
      });
      device.queue.submit([encoder.finish()]);
    },
    dispose() {
      disposed = true;
      camera.destroy();
      targets.destroy();
      post.destroy();
      taa.destroy();
      ibl.destroy();
      gbuffer.destroy();
      aoPass.destroy();
      history.destroy();
      bilateral.destroy();
      propSet.destroy();
      frame.destroy();
      noHistory.destroy();
      Object.values(meshes).forEach((m) => m.destroy());
    },
  };
};

export default create;
