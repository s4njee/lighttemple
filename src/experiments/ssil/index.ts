import { mat4, vec3 } from 'wgpu-matrix';
import commonWgsl from '../../shaders/common.wgsl?raw';
import fullscreenWgsl from '../../shaders/fullscreen.wgsl?raw';
import brdfWgsl from '../../shaders/brdf.wgsl?raw';
import shadowWgsl from '../../shaders/shadow.wgsl?raw';
import screenSpaceWgsl from '../../shaders/screen-space.wgsl?raw';
import gtaoWgsl from '../../shaders/gtao.wgsl?raw';
import { MESH_VERTEX_LAYOUT, MESH_VERTEX_WGSL, box, sphere, uploadMesh } from '../../gpu/geometry';
import { createShader, wgsl } from '../../gpu/shader';
import { UniformBlock } from '../../gpu/uniforms';
import { hexToLinear } from '../../gpu/color';
import { TargetPool } from '../../gpu/targets';
import { generateMipmaps, mipLevelCount } from '../../gpu/mipmaps';
import { HDR_FORMAT, PostProcess } from '../../post/post';
import { TemporalAA } from '../../post/taa';
import { GBuffer, screenCamera } from '../../post/gbuffer';
import { AO_METHOD, GtaoPass } from '../../post/gtao';
import { SsilPass } from '../../post/ssil';
import { TemporalAccumulator } from '../../post/temporal-accumulate';
import { BilateralFilter } from '../../post/bilateral';
import { OrbitCamera } from '../../scene/orbit-camera';
import { fitDirectionalShadow, type Sphere } from '../../scene/light-frustum';
import { PROP_WGSL, PropSet, type Prop } from '../../scene/props';
import type { ExperimentFactory } from '../types';
import shaderWgsl from './shader.wgsl?raw';

const ALBEDO_FORMAT: GPUTextureFormat = 'rgba8unorm-srgb';
const SHADOW_FORMAT: GPUTextureFormat = 'depth32float';
const SHADOW_SIZE = 2048;
/** Mips of the bounce-source buffer: far SSIL samples read up to 1/64 resolution. */
const RADIANCE_MIPS = 7;
const ROOM_BOUNDS: Sphere = { center: vec3.fromValues(0, 1.7, 0), radius: 5.4 };
const VIEW = { COMBINED: 0, DIRECT: 1, INDIRECT: 2, GTAO: 3 } as const;

/** Camera framings: the whole room, or a close-up of the ceiling with the sunlit floor off-screen. */
const FRAMINGS = {
  room: { target: [0, 1.45, -0.3] as [number, number, number], distance: 8.4, yaw: 0, pitch: 0.1 },
  ceiling: { target: [0, 2.7, -1.2] as [number, number, number], distance: 6.4, yaw: 0, pitch: -0.3 },
};

const create: ExperimentFactory = async (ctx) => {
  const { device, surface, params } = ctx;

  const gi = params.add(
    {
      resolution: {
        value: 0.5,
        options: { Full: 1, Half: 0.5 },
        hint: 'Half gathers for a quarter of the pixels; a depth/normal-aware upsample restores full resolution.',
      },
      radius: { value: 5.5, min: 0.5, max: 10, step: 0.1, hint: 'World-space gather radius (metres). A room needs most of its own size.' },
      thickness: {
        value: 0.6,
        min: 0.02,
        max: 4,
        step: 0.02,
        hint: 'How deep every surface on screen is assumed to be. Thin: light leaks behind objects. Thick: everything casts a long occlusion "shadow" behind it, like plain GTAO.',
      },
      slices: { value: 2, min: 1, max: 4, step: 1, hint: 'Directions per pixel per frame; accumulation rotates them every frame.' },
      steps: { value: 12, min: 4, max: 24, step: 1, hint: 'Samples per side of each slice. Cost: slices × steps × 2 texel reads + radiance taps per pixel.' },
      maxRadius: {
        value: 720,
        min: 64,
        max: 1024,
        step: 16,
        label: 'Max radius (px)',
        hint: 'Caps the on-screen gather radius (full-resolution pixels).',
      },
      giStrength: { value: 1, min: 0, max: 3, step: 0.05, label: 'GI strength' },
    },
    'Indirect light (SSIL)',
  );
  const denoise = params.add(
    {
      temporal: { value: true, label: 'Temporal accumulation', hint: 'Off shows a single frame\'s raw gather.' },
      frames: { value: 24, min: 2, max: 64, step: 1, label: 'History frames' },
      clamp: {
        value: 0,
        min: 0,
        max: 4,
        step: 0.25,
        label: 'History clamp (σ)',
        hint: 'Clip history to the current neighbourhood (mean ± kσ) so moving light updates faster, at the price of more noise. 0 = off.',
      },
      spatial: { value: true, label: 'Spatial denoise' },
      blurRadius: { value: 3, min: 1, max: 4, step: 1, label: 'Denoise radius', hint: 'In GI pixels: (2r+1)² depth/normal-aware taps.' },
    },
    'Denoise',
  );
  const sun = params.add(
    {
      azimuth: { value: 100, min: 60, max: 150, step: 1, hint: 'Degrees; 90 = square on to the window, above 90 the light slants toward the open front.' },
      elevation: { value: 24, min: 4, max: 60, step: 1 },
      intensity: { value: 9, min: 0, max: 20, step: 0.1 },
      sunColor: { value: '#ffe2bf', type: 'color', label: 'Colour' },
      animateSun: { value: false, label: 'Sweep sun', hint: 'Slowly swings the sun, so the bounce has to follow (watch the history clamp).' },
      sky: { value: '#9db4d6', type: 'color', label: 'Sky light', hint: 'Skylight through the open front, shadowed by the bitmask visibility.' },
      skyStrength: { value: 0.06, min: 0, max: 0.5, step: 0.005, label: 'Sky strength' },
    },
    'Sun & sky',
  );
  const look = params.add(
    {
      view: {
        value: VIEW.COMBINED,
        options: {
          'Combined (direct + SSIL)': VIEW.COMBINED,
          'Direct only': VIEW.DIRECT,
          'Indirect only': VIEW.INDIRECT,
          'Direct + GTAO ambient (no GI)': VIEW.GTAO,
          'Incoming indirect': 4,
          'Bitmask visibility': 5,
          'History length': 6,
        },
        hint: 'GTAO ambient is the usual fallback: a flat fill light darkened by AO. Incoming indirect is the gathered light before albedo.',
      },
      fill: { value: '#4a433d', type: 'color', label: 'Flat ambient', hint: 'The fill colour for the GTAO comparison view.' },
      framing: {
        value: 'room',
        options: { Room: 'room', 'Ceiling (sunlit floor off-screen)': 'ceiling' },
        hint: 'Screen-space failure: frame the ceiling alone and the floor that lights it leaves the screen, so its bounce vanishes.',
      },
      antialiasing: { value: 'taa', options: { TAA: 'taa', Off: 'off' } },
    },
    'Look & debug',
  );
  const post = await PostProcess.create(ctx, { exposure: 1.3, bloom: 0.03, vignette: 0.2, grain: 0.015 });
  const [taa, gbuffer, ssil, gtao, giHistory, aoHistory, bilateral] = await Promise.all([
    TemporalAA.create(device),
    GBuffer.create(device),
    SsilPass.create(device),
    GtaoPass.create(device),
    TemporalAccumulator.create(device),
    TemporalAccumulator.create(device),
    BilateralFilter.create(device),
  ]);
  const targets = new TargetPool(device);

  // --- Scene: a Cornell-ish room, open at the front, with a window in the right wall ------------
  const meshes = { box: uploadMesh(device, box(1, 1, 1), 'ssil:box'), sphere: uploadMesh(device, sphere(1, 64, 40), 'ssil:sphere') };
  /** Box from min corner to max corner. */
  const slab = (x0: number, y0: number, z0: number, x1: number, y1: number, z1: number, ry = 0) =>
    mat4.scale(mat4.rotateY(mat4.translation([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]), ry), [x1 - x0, y1 - y0, z1 - z0]);
  const W = 3; // half width
  const D = 3; // half depth
  const H = 3.4;
  const win = { y0: 0.35, y1: 3.05, z0: -2.4, z1: 1.4 };
  const red = '#b3241c';
  const green = '#2d7a48';
  const white = '#e9e4da';
  const glazing = '#f4f1ea';
  const props: Prop[] = [
    { mesh: meshes.box, color: '#e3d3b8', roughness: 0.75, transform: () => slab(-W - 0.2, -0.2, -D - 0.2, W + 0.2, 0, D + 0.4) },
    { mesh: meshes.box, color: '#f2efe9', roughness: 0.9, transform: () => slab(-W - 0.2, H, -D - 0.2, W + 0.2, H + 0.2, D) },
    { mesh: meshes.box, color: white, roughness: 0.9, transform: () => slab(-W, 0, -D - 0.2, W, H, -D) },
    { mesh: meshes.box, color: red, roughness: 0.85, transform: () => slab(-W - 0.2, 0, -D - 0.2, -W, H, D) },
    // The right wall, built around a tall window.
    { mesh: meshes.box, color: green, roughness: 0.85, transform: () => slab(W, 0, -D - 0.2, W + 0.2, win.y0, D) },
    { mesh: meshes.box, color: green, roughness: 0.85, transform: () => slab(W, win.y1, -D - 0.2, W + 0.2, H, D) },
    { mesh: meshes.box, color: green, roughness: 0.85, transform: () => slab(W, win.y0, -D - 0.2, W + 0.2, win.y1, win.z0) },
    { mesh: meshes.box, color: green, roughness: 0.85, transform: () => slab(W, win.y0, win.z1, W + 0.2, win.y1, D) },
    { mesh: meshes.box, color: glazing, roughness: 0.6, transform: () => slab(W + 0.04, win.y0, -0.54, W + 0.12, win.y1, -0.46) },
    { mesh: meshes.box, color: glazing, roughness: 0.6, transform: () => slab(W + 0.04, 1.66, win.z0, W + 0.12, 1.74, win.z1) },
    // An orange rug where the sun lands: the brightest, most saturated bounce in the room.
    { mesh: meshes.box, color: '#d8742a', roughness: 0.95, transform: () => slab(-2.5, 0, -1.5, 0.7, 0.006, 1.5) },
    // Furniture in white plaster, to catch the coloured bounce.
    { mesh: meshes.sphere, color: white, roughness: 0.7, transform: () => mat4.scale(mat4.translation([-2.25, 0.55, -2.1]), [0.55, 0.55, 0.55]) },
    { mesh: meshes.box, color: white, roughness: 0.8, transform: () => slab(-0.75, 0, -0.75, 0.05, 1.7, 0.05, 0.35) },
    { mesh: meshes.box, color: white, roughness: 0.8, transform: () => slab(-1.2, 0, -2.95, 1.6, 0.45, -2.4) },
    // A thin board: the thickness test (it should block light only a short way behind itself).
    { mesh: meshes.box, color: '#e8dcc6', roughness: 0.7, transform: () => slab(1.5, 0, 1.2, 1.56, 2.3, 2.1) },
  ];
  const propSet = new PropSet(device, props, 'ssil:props');
  propSet.update(0);

  const frame = new UniformBlock(device, 'Frame', {
    viewProj: 'mat4x4f',
    view: 'mat4x4f',
    invView: 'mat4x4f',
    lightViewProj: 'mat4x4f',
    cameraPos: 'vec3f',
    giStrength: 'f32',
    lightDir: 'vec3f',
    debugView: 'u32',
    lightColor: 'vec4f',
    skyColor: 'vec4f',
    fillColor: 'vec4f',
    outsideColor: 'vec4f',
    shadowTexelWorld: 'f32',
    normalOffset: 'f32',
    depthBias: 'f32',
    shadowSoftness: 'f32',
    noiseOffset: 'f32',
    maxFrames: 'f32',
  });

  // --- Pipelines ----------------------------------------------------------------------------
  const module = await createShader(
    device,
    'ssil-scene',
    wgsl(commonWgsl, fullscreenWgsl, MESH_VERTEX_WGSL, PROP_WGSL, frame.wgslStruct, shadowWgsl, brdfWgsl, screenSpaceWgsl, gtaoWgsl, shaderWgsl),
  );
  const shadowLayout = device.createBindGroupLayout({
    label: 'ssil:shadow',
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
    ],
  });
  const geometryLayout = device.createBindGroupLayout({
    label: 'ssil:geometry',
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
    ],
  });
  const tex = (binding: number, sampleType: GPUTextureSampleType = 'unfilterable-float'): GPUBindGroupLayoutEntry => ({
    binding,
    visibility: GPUShaderStage.FRAGMENT,
    texture: { sampleType },
  });
  const compositeLayout = device.createBindGroupLayout({
    label: 'ssil:composite',
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      tex(4, 'depth'),
      tex(5),
      tex(6),
      tex(7),
      tex(8),
      tex(9),
      tex(10),
    ],
  });
  const layout = (l: GPUBindGroupLayout) => device.createPipelineLayout({ bindGroupLayouts: [l] });
  const [shadowPipeline, geometryPipeline, compositePipeline] = await Promise.all([
    device.createRenderPipelineAsync({
      label: 'ssil:shadow',
      layout: layout(shadowLayout),
      vertex: { module, entryPoint: 'vs_shadow', buffers: [MESH_VERTEX_LAYOUT] },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { format: SHADOW_FORMAT, depthWriteEnabled: true, depthCompare: 'less' },
    }),
    device.createRenderPipelineAsync({
      label: 'ssil:geometry',
      layout: layout(geometryLayout),
      vertex: { module, entryPoint: 'vs_geometry', buffers: [MESH_VERTEX_LAYOUT] },
      fragment: { module, entryPoint: 'fs_geometry', targets: [GBuffer.normalTarget, { format: ALBEDO_FORMAT }, { format: HDR_FORMAT }] },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: GBuffer.depthStencil,
    }),
    device.createRenderPipelineAsync({
      label: 'ssil:composite',
      layout: layout(compositeLayout),
      vertex: { module, entryPoint: 'vs_fullscreen' },
      fragment: { module, entryPoint: 'fs_composite', targets: [{ format: HDR_FORMAT }] },
    }),
  ]);

  const shadowMap = device.createTexture({
    label: 'ssil:shadow-map',
    size: [SHADOW_SIZE, SHADOW_SIZE],
    format: SHADOW_FORMAT,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  const shadowView = shadowMap.createView();
  const shadowSampler = device.createSampler({ label: 'ssil:shadow', compare: 'less-equal', magFilter: 'linear', minFilter: 'linear' });
  const shadowGroup = device.createBindGroup({
    layout: shadowLayout,
    entries: [
      { binding: 0, resource: { buffer: frame.buffer } },
      { binding: 1, resource: { buffer: propSet.buffer } },
    ],
  });
  const geometryGroup = device.createBindGroup({
    layout: geometryLayout,
    entries: [
      { binding: 0, resource: { buffer: frame.buffer } },
      { binding: 1, resource: { buffer: propSet.buffer } },
      { binding: 2, resource: shadowView },
      { binding: 3, resource: shadowSampler },
    ],
  });
  const placeholder = (format: GPUTextureFormat) =>
    device.createTexture({ label: `ssil:placeholder:${format}`, size: [1, 1], format, usage: GPUTextureUsage.TEXTURE_BINDING });
  const noAo = placeholder('rgba16float');
  const noHistory = placeholder('r16float');

  // The bounce source: direct light with a mip chain (rendered into mip 0 by the geometry pass).
  let radiance: { texture: GPUTexture; view0: GPUTextureView } | null = null;
  const radianceTarget = (w: number, h: number) => {
    if (radiance?.texture.width !== w || radiance.texture.height !== h) {
      radiance?.texture.destroy();
      const texture = device.createTexture({
        label: 'ssil:direct',
        size: [w, h],
        format: HDR_FORMAT,
        mipLevelCount: Math.min(RADIANCE_MIPS, mipLevelCount(w, h)),
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      radiance = { texture, view0: texture.createView({ baseMipLevel: 0, mipLevelCount: 1 }) };
    }
    return radiance;
  };

  const camera = new OrbitCamera(surface.canvas, { ...FRAMINGS.room, minPitch: -0.9, maxPitch: 1.2, minDistance: 1.5, maxDistance: 16 });
  let framing = look.framing;
  if (framing !== 'room') camera.setView(FRAMINGS[framing as keyof typeof FRAMINGS]);
  let frameIndex = 0;
  let sunTime = 0;
  let settingsKey = '';
  let wasComparing = false;

  return {
    frame({ time, dt, size }) {
      if (look.framing !== framing) {
        framing = look.framing;
        camera.setView(FRAMINGS[framing as keyof typeof FRAMINGS]);
      }
      camera.update(dt, size.width / size.height);
      if (sun.animateSun) sunTime += dt;

      const useTaa = look.antialiasing === 'taa';
      if (!useTaa) taa.reset();
      const jitterPixels: [number, number] = useTaa ? taa.jitterPixels : [0, 0];
      const viewProj = useTaa ? taa.jitter(camera.viewProjection, size) : camera.viewProjection;
      const cam = screenCamera(camera.projection, camera.near, camera.far, size, jitterPixels);
      const scale = gi.resolution === 1 ? 1 : 0.5;
      const comparing = look.view === VIEW.GTAO;

      // Restart the history when the signal changes meaning; an idle accumulator's history is stale too.
      const key = `${scale}:${gi.radius}:${gi.thickness}:${gi.slices}:${gi.steps}:${gi.maxRadius}`;
      if (key !== settingsKey || comparing !== wasComparing) {
        settingsKey = key;
        wasComparing = comparing;
        giHistory.reset();
        aoHistory.reset();
      }
      if (denoise.temporal) frameIndex++;
      const noiseOffset = (frameIndex % 64) * 5.588238;

      // Sun: optionally sweeping ±12° around the chosen azimuth.
      const az = ((sun.azimuth + 12 * Math.sin(sunTime * 0.25)) * Math.PI) / 180;
      const el = (sun.elevation * Math.PI) / 180;
      const toSun = vec3.fromValues(Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az));
      const fit = fitDirectionalShadow({ toLight: toSun, bounds: ROOM_BOUNDS, mapSize: SHADOW_SIZE });
      const lc = hexToLinear(sun.sunColor);
      const sky = hexToLinear(sun.sky);
      const fill = hexToLinear(look.fill);

      frame
        .set('viewProj', viewProj)
        .set('view', camera.view)
        .set('invView', mat4.inverse(camera.view))
        .set('lightViewProj', fit.viewProj)
        .set('cameraPos', camera.position)
        .set('giStrength', gi.giStrength)
        .set('lightDir', vec3.negate(toSun))
        .set('debugView', look.view)
        .set('lightColor', [lc[0] * sun.intensity, lc[1] * sun.intensity, lc[2] * sun.intensity, 0])
        .set('skyColor', [sky[0] * sun.skyStrength, sky[1] * sun.skyStrength, sky[2] * sun.skyStrength, 0])
        .set('fillColor', [...fill, 0])
        .set('outsideColor', [sky[0] * 2.5, sky[1] * 2.5, sky[2] * 2.5, 1])
        .set('shadowTexelWorld', fit.texelWorld)
        .set('normalOffset', 1.5)
        .set('depthBias', 0.0005)
        .set('shadowSoftness', 2.5 / SHADOW_SIZE)
        .set('noiseOffset', useTaa ? taa.frameNoiseOffset : 0)
        .set('maxFrames', denoise.frames)
        .upload();

      const encoder = device.createCommandEncoder({ label: 'ssil' });

      // 1. Sun shadow map.
      const sp = encoder.beginRenderPass({
        label: 'ssil:shadow',
        colorAttachments: [],
        depthStencilAttachment: { view: shadowView, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
      });
      sp.setPipeline(shadowPipeline);
      sp.setBindGroup(0, shadowGroup);
      propSet.draw(sp);
      sp.end();

      // 2. G-buffer + direct light (the bounce source), then its mip chain.
      const gb = gbuffer.attachments(size);
      const albedo = targets.get('albedo', { size: [size.width, size.height], format: ALBEDO_FORMAT });
      const direct = radianceTarget(size.width, size.height);
      const clear = (view: GPUTextureView): GPURenderPassColorAttachment => ({ view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] });
      const gp = encoder.beginRenderPass({
        label: 'ssil:geometry',
        colorAttachments: [gb.normal, clear(albedo.view), clear(direct.view0)],
        depthStencilAttachment: gb.depth,
      });
      gp.setPipeline(geometryPipeline);
      gp.setBindGroup(0, geometryGroup);
      propSet.draw(gp);
      gp.end();
      generateMipmaps(device, direct.texture, encoder);

      // 3. Indirect light (or, for the comparison view, GTAO) at full or half resolution.
      const geo = gbuffer.resolve(encoder, scale, cam);
      const history = comparing ? aoHistory : giHistory;
      let signal = comparing
        ? gtao.run(encoder, geo.current, cam, {
            method: AO_METHOD.GTAO,
            radius: 0.9,
            falloff: 0.5,
            slices: 2,
            steps: 6,
            maxRadiusPx: 160 * scale,
            noiseOffset,
          })
        : ssil.run(encoder, geo.current, direct.texture, cam, {
            radius: gi.radius,
            maxRadiusPx: gi.maxRadius * scale,
            thickness: gi.thickness,
            slices: gi.slices,
            steps: gi.steps,
            noiseOffset,
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
          clampSigma: comparing ? 0 : denoise.clamp,
        });
      } else {
        giHistory.trim();
        aoHistory.trim();
      }
      if (denoise.spatial) signal = bilateral.denoise(encoder, signal, geo.current, { radius: denoise.blurRadius });
      if (scale < 1) signal = bilateral.upsample(encoder, signal, geo.current, gbuffer, cam, { background: [0, 0, 0, 1] });

      // 4. Composite.
      const compositeGroup = device.createBindGroup({
        layout: compositeLayout,
        entries: [
          { binding: 0, resource: { buffer: frame.buffer } },
          { binding: 4, resource: gbuffer.depth.createView() },
          { binding: 5, resource: gbuffer.normal.createView() },
          { binding: 6, resource: albedo.view },
          { binding: 7, resource: direct.texture.createView() },
          { binding: 8, resource: (comparing ? noAo : signal).createView() },
          { binding: 9, resource: (comparing ? signal : noAo).createView() },
          { binding: 10, resource: (denoise.temporal ? history.historyLength : null)?.createView() ?? noHistory.createView() },
        ],
      });
      const sceneTarget = post.scene(size);
      const cp = encoder.beginRenderPass({
        label: 'ssil:composite',
        colorAttachments: [clear(sceneTarget.view)],
      });
      cp.setPipeline(compositePipeline);
      cp.setBindGroup(0, compositeGroup);
      cp.draw(3);
      cp.end();

      const source = useTaa
        ? taa.resolve(encoder, { color: sceneTarget.texture, depth: gbuffer.depth, viewProj: camera.viewProjection })
        : undefined;
      post.render(encoder, surface.currentTexture().createView(), {
        time,
        raw: look.view >= 4,
        background: hexToLinear('#1b1917'),
        source,
      });
      device.queue.submit([encoder.finish()]);
    },
    dispose() {
      camera.destroy();
      targets.destroy();
      post.destroy();
      taa.destroy();
      gbuffer.destroy();
      ssil.destroy();
      gtao.destroy();
      giHistory.destroy();
      aoHistory.destroy();
      bilateral.destroy();
      propSet.destroy();
      frame.destroy();
      shadowMap.destroy();
      noAo.destroy();
      noHistory.destroy();
      radiance?.texture.destroy();
      Object.values(meshes).forEach((m) => m.destroy());
    },
  };
};

export default create;
