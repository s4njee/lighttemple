import { mat4 } from 'wgpu-matrix';
import commonWgsl from '../../shaders/common.wgsl?raw';
import fullscreenWgsl from '../../shaders/fullscreen.wgsl?raw';
import noiseWgsl from '../../shaders/noise.wgsl?raw';
import brdfWgsl from '../../shaders/brdf.wgsl?raw';
import shadowWgsl from '../../shaders/shadow.wgsl?raw';
import sunRoomWgsl from '../../shaders/sun-room.wgsl?raw';
import { MESH_VERTEX_LAYOUT, MESH_VERTEX_WGSL } from '../../gpu/geometry';
import { createShader, wgsl } from '../../gpu/shader';
import { TargetPool } from '../../gpu/targets';
import { UniformBlock } from '../../gpu/uniforms';
import { hexToLinear } from '../../gpu/color';
import { HDR_FORMAT, PostProcess } from '../../post/post';
import { TemporalAA } from '../../post/taa';
import { OrbitCamera } from '../../scene/orbit-camera';
import { fitDirectionalShadow } from '../../scene/light-frustum';
import { PROP_WGSL } from '../../scene/props';
import { DEFAULT_ROOM_COLORS, DEFAULT_SUN, SUN_ROOM_BOUNDS, SUN_ROOM_CAMERA, SunRoom, outdoorColors, sunDirection } from '../../scene/sun-room';
import type { ExperimentFactory } from '../types';
import shaderWgsl from './shader.wgsl?raw';

const MSAA_SAMPLES = 4;
const DEPTH_FORMAT: GPUTextureFormat = 'depth24plus';
const SHADOW_FORMAT: GPUTextureFormat = 'depth32float';
const FLUX_FORMAT: GPUTextureFormat = 'rgba16float';
/** Low-resolution G-buffer: full-precision positions (bilateral tests compare them), half-float normals. */
const GPOS_FORMAT: GPUTextureFormat = 'rgba32float';
const VIEW = { lit: 0, indirect: 1, irradiance: 2, direct: 3, lowRes: 4, fallback: 5 } as const;

const create: ExperimentFactory = async (ctx) => {
  const { device, surface, params, reducedMotion } = ctx;
  const coarse = window.matchMedia('(pointer: coarse)').matches;

  const sun = params.add(
    {
      azimuth: { value: DEFAULT_SUN.azimuth, min: -150, max: -30, step: 1, hint: 'The window faces −x; the sun has to be on that side to get in.' },
      elevation: { value: DEFAULT_SUN.elevation, min: 12, max: 70, step: 1 },
      sunIntensity: { value: 9, min: 0, max: 20, step: 0.05, label: 'Intensity' },
      sunColor: { value: '#fff0dc', type: 'color', label: 'Colour' },
      sky: { value: '#9cc0ee', type: 'color', hint: 'The view through the window, and the colour of the fill light.' },
      fill: {
        value: 0.05,
        min: 0,
        max: 0.5,
        step: 0.005,
        label: 'Sky fill',
        hint: 'A flat ambient term standing in for the skylight through the window, which RSM does not model (it only bounces the sun).',
      },
      sway: { value: !reducedMotion, label: 'Sway sun', hint: 'Swings the sun slowly. RSM is recomputed from scratch every frame, so it follows instantly.' },
    },
    'Light',
  );
  const rsm = params.add(
    {
      rsmSize: {
        value: coarse ? 256 : 512,
        options: { '256': 256, '512': 512, '1024': 1024 },
        label: 'RSM size',
        hint: 'Texels per side of the flux/normal/depth maps. Small is fine: the gather blurs them anyway, and fewer texels means less aliasing in a sparse gather.',
      },
      samples: {
        value: coarse ? 64 : 128,
        min: 8,
        max: 256,
        step: 8,
        hint: 'RSM texels gathered per receiver. Noise falls with the square root of the count; TAA averages the per-frame rotation on top.',
      },
      radius: {
        value: 3.5,
        min: 0.25,
        max: 5,
        step: 0.05,
        label: 'Sampling radius',
        hint: 'Metres around the receiver\'s own position in the light\'s view. Light from farther texels is missed; a larger disc is noisier for the same samples.',
      },
      denoise: {
        value: 3,
        min: 0,
        max: 8,
        step: 1,
        label: 'Denoise radius',
        hint: 'In gather texels. A separable blur of the gathered light that stops at G-buffer edges (normal and plane tests), before upsampling. 0 shows the raw gather.',
      },
      intensity: { value: 1, min: 0, max: 4, step: 0.01, label: 'Indirect intensity' },
      resolution: {
        value: 0.25,
        options: { Full: 1, '½': 0.5, '¼': 0.25 },
        label: 'Gather resolution',
        hint: 'Gather at a fraction of the screen resolution, then upsample. ¼ is 16× fewer gathers, but more pixels fail the bilateral test and gather again at full resolution.',
      },
      upsample: {
        value: 1,
        options: { 'Bilateral + fallback': 1, Bilinear: 0 },
        hint: 'Bilateral keeps low-resolution gathers whose position and normal match the pixel and re-gathers at full resolution where fewer than three of four match. Bilinear blurs light across edges.',
      },
    },
    'Reflective shadow map',
  );
  const look = params.add(
    {
      rug: { value: DEFAULT_ROOM_COLORS.rug, type: 'color' },
      walls: { value: DEFAULT_ROOM_COLORS.walls, type: 'color' },
      view: {
        value: VIEW.lit,
        options: {
          Lit: VIEW.lit,
          'Indirect only': VIEW.indirect,
          'Bounce irradiance': VIEW.irradiance,
          'Direct only': VIEW.direct,
          'Low-res gather': VIEW.lowRes,
          'Upsample fallback': VIEW.fallback,
        },
        hint: 'Low-res gather: the indirect light before upsampling. Upsample fallback: magenta where the bilateral test failed and the pixel gathered at full resolution.',
      },
      showRsm: { value: false, label: 'Show RSM', hint: 'The three RSM buffers from the sun: reflected flux, normal, world position.' },
      shadowSize: { value: coarse ? 1024 : 2048, options: { '1024': 1024, '2048': 2048, '4096': 4096 }, label: 'Sun map size' },
      antialiasing: {
        value: 'taa',
        options: { TAA: 'taa', 'MSAA 4×': 'msaa', Off: 'off' },
        hint: 'TAA also averages the gather\'s per-frame rotation, turning its noise into extra samples.',
      },
    },
    'Look & debug',
  );
  const post = await PostProcess.create(ctx, { exposure: 2.2, bloom: 0.04, vignette: 0.25, grain: 0.015 });
  const taa = await TemporalAA.create(device);

  const room = new SunRoom(device, 'reflective-shadow-maps');

  const frame = new UniformBlock(device, 'Frame', {
    viewProj: 'mat4x4f',
    lightViewProj: 'mat4x4f',
    invLightViewProj: 'mat4x4f',
    cameraPos: 'vec3f',
    shadowTexelWorld: 'f32',
    toSun: 'vec3f',
    intensity: 'f32',
    sunColor: 'vec4f',
    fillColor: 'vec4f',
    skyZenith: 'vec4f',
    skyHorizon: 'vec4f',
    outdoorGround: 'vec4f',
    fullSize: 'vec2f',
    lowSize: 'vec2f',
    rsmExtent: 'f32',
    radius: 'f32',
    samples: 'u32',
    debugView: 'u32',
    upsample: 'u32',
    denoise: 'f32',
    noiseOffset: 'f32',
    normalOffset: 'f32',
  });

  const module = await createShader(
    device,
    'reflective-shadow-maps',
    wgsl(commonWgsl, fullscreenWgsl, noiseWgsl, brdfWgsl, shadowWgsl, sunRoomWgsl, MESH_VERTEX_WGSL, PROP_WGSL, frame.wgslStruct, shaderWgsl),
  );

  // --- Layouts: scene (0), RSM + low-res G-buffer (1), main-pass extras (2) ----------------
  const sceneLayout = device.createBindGroupLayout({
    label: 'rsm:scene',
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    ],
  });
  const buffersLayout = device.createBindGroupLayout({
    label: 'rsm:buffers',
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
    ],
  });
  const mainLayout = device.createBindGroupLayout({
    label: 'rsm:main',
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  });
  const denoiseLayout = device.createBindGroupLayout({
    label: 'rsm:denoise',
    entries: [{ binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } }],
  });
  const pipelineLayout = (...layouts: GPUBindGroupLayout[]) => device.createPipelineLayout({ bindGroupLayouts: layouts });
  const sceneOnly = pipelineLayout(sceneLayout);
  const withBuffers = pipelineLayout(sceneLayout, buffersLayout);
  const full = pipelineLayout(sceneLayout, buffersLayout, mainLayout);

  const primitive: GPUPrimitiveState = { topology: 'triangle-list', cullMode: 'back' };
  const cameraPipelines = (count: number) =>
    Promise.all([
      device.createRenderPipelineAsync({
        label: `rsm:main-x${count}`,
        layout: full,
        vertex: { module, entryPoint: 'vs_main', buffers: [MESH_VERTEX_LAYOUT] },
        fragment: { module, entryPoint: 'fs_main', targets: [{ format: HDR_FORMAT }] },
        primitive,
        depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: 'less' },
        multisample: { count },
      }),
      device.createRenderPipelineAsync({
        label: `rsm:overlay-x${count}`,
        layout: withBuffers,
        vertex: { module, entryPoint: 'vs_overlay' },
        fragment: { module, entryPoint: 'fs_overlay', targets: [{ format: HDR_FORMAT }] },
        depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: 'always' },
        multisample: { count },
      }),
    ]);
  const [shadowPipeline, rsmPipeline, gbufferPipeline, gatherPipeline, denoisePipeline, msaaPipelines, singlePipelines] = await Promise.all([
    device.createRenderPipelineAsync({
      label: 'rsm:shadow',
      layout: sceneOnly,
      vertex: { module, entryPoint: 'vs_shadow', buffers: [MESH_VERTEX_LAYOUT] },
      primitive,
      depthStencil: { format: SHADOW_FORMAT, depthWriteEnabled: true, depthCompare: 'less', depthBias: 2, depthBiasSlopeScale: 1.5 },
    }),
    device.createRenderPipelineAsync({
      label: 'rsm:rsm',
      layout: sceneOnly,
      vertex: { module, entryPoint: 'vs_light', buffers: [MESH_VERTEX_LAYOUT] },
      fragment: { module, entryPoint: 'fs_rsm', targets: [{ format: FLUX_FORMAT }, { format: FLUX_FORMAT }] },
      primitive,
      depthStencil: { format: SHADOW_FORMAT, depthWriteEnabled: true, depthCompare: 'less' },
    }),
    device.createRenderPipelineAsync({
      label: 'rsm:gbuffer',
      layout: sceneOnly,
      vertex: { module, entryPoint: 'vs_main', buffers: [MESH_VERTEX_LAYOUT] },
      fragment: { module, entryPoint: 'fs_gbuffer', targets: [{ format: GPOS_FORMAT }, { format: FLUX_FORMAT }] },
      primitive,
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: 'less' },
    }),
    device.createRenderPipelineAsync({
      label: 'rsm:gather',
      layout: withBuffers,
      vertex: { module, entryPoint: 'vs_fullscreen' },
      fragment: { module, entryPoint: 'fs_gather', targets: [{ format: HDR_FORMAT }] },
    }),
    device.createRenderPipelineAsync({
      label: 'rsm:denoise',
      layout: pipelineLayout(sceneLayout, buffersLayout, denoiseLayout),
      vertex: { module, entryPoint: 'vs_denoise' },
      fragment: { module, entryPoint: 'fs_denoise', targets: [{ format: HDR_FORMAT }] },
    }),
    cameraPipelines(MSAA_SAMPLES),
    cameraPipelines(1),
  ]);

  const shadowSampler = device.createSampler({
    label: 'rsm:shadow-compare',
    compare: 'less-equal',
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  });
  const linearSampler = device.createSampler({ label: 'rsm:linear', magFilter: 'linear', minFilter: 'linear' });
  const sceneGroup = device.createBindGroup({
    layout: sceneLayout,
    entries: [
      { binding: 0, resource: { buffer: frame.buffer } },
      { binding: 1, resource: { buffer: room.props.buffer } },
    ],
  });

  // Bind groups are rebuilt whenever one of their textures is (re)created.
  const cached = <T extends GPUTexture[]>(build: (...t: T) => GPUBindGroup) => {
    let key: GPUTexture[] = [];
    let group: GPUBindGroup | null = null;
    return (...t: T) => {
      if (!group || t.some((x, i) => x !== key[i])) {
        key = t;
        group = build(...t);
      }
      return group;
    };
  };
  const buffersGroup = cached((depth: GPUTexture, flux: GPUTexture, normal: GPUTexture, gPos: GPUTexture, gNormal: GPUTexture) =>
    device.createBindGroup({
      layout: buffersLayout,
      entries: [depth, flux, normal, gPos, gNormal].map((t, binding) => ({ binding, resource: t.createView() })),
    }),
  );
  const denoiseGroups = new WeakMap<GPUTexture, GPUBindGroup>();
  const denoiseGroup = (src: GPUTexture) => {
    let g = denoiseGroups.get(src);
    if (!g) {
      g = device.createBindGroup({ layout: denoiseLayout, entries: [{ binding: 4, resource: src.createView() }] });
      denoiseGroups.set(src, g);
    }
    return g;
  };
  const mainGroup = cached((shadowMap: GPUTexture, indirect: GPUTexture) =>
    device.createBindGroup({
      layout: mainLayout,
      entries: [
        { binding: 0, resource: shadowMap.createView() },
        { binding: 1, resource: shadowSampler },
        { binding: 2, resource: indirect.createView() },
        { binding: 3, resource: linearSampler },
      ],
    }),
  );

  const targets = new TargetPool(device);
  const camera = new OrbitCamera(surface.canvas, SUN_ROOM_CAMERA);
  let swayTime = 0;
  const background = hexToLinear('#1b1917');

  return {
    frame({ time, dt, size }) {
      if (sun.sway) swayTime += dt;
      camera.update(dt, size.width / size.height);
      room.updateCutaway(camera.position);
      room.setColors({ rug: look.rug, walls: look.walls });

      const toSun = sunDirection(sun.azimuth + 12 * Math.sin(swayTime * 0.11), sun.elevation + 3 * Math.sin(swayTime * 0.07 + 1));
      const sc = hexToLinear(sun.sunColor).map((c) => c * sun.sunIntensity);
      const sky = hexToLinear(sun.sky);
      const outdoor = outdoorColors(sc, sky.map((c) => c * 1.5), toSun, 1.5);
      // One light frustum around the room serves both the shadow map and the RSM (no texel
      // snapping: the frustum never moves with the camera).
      const fit = fitDirectionalShadow({ toLight: toSun, bounds: SUN_ROOM_BOUNDS, mapSize: look.shadowSize, snap: false });

      const aa = look.antialiasing;
      const samples = aa === 'msaa' ? MSAA_SAMPLES : 1;
      const [mainPipeline, overlayPipeline] = samples > 1 ? msaaPipelines : singlePipelines;
      if (aa !== 'taa') taa.reset();
      const viewProj = aa === 'taa' ? taa.jitter(camera.viewProjection, size) : camera.viewProjection;

      const fullSize = [size.width, size.height] as const;
      const scale = rsm.resolution;
      const rsmSize = [rsm.rsmSize, rsm.rsmSize] as const;
      const shadowMap = targets.get('shadow', { size: [look.shadowSize, look.shadowSize], format: SHADOW_FORMAT });
      const rsmDepth = targets.get('rsm-depth', { size: rsmSize, format: SHADOW_FORMAT });
      const rsmFlux = targets.get('rsm-flux', { size: rsmSize, format: FLUX_FORMAT });
      const rsmNormal = targets.get('rsm-normal', { size: rsmSize, format: FLUX_FORMAT });
      const gPos = targets.get('g-pos', { size: fullSize, scale, format: GPOS_FORMAT });
      const gNormal = targets.get('g-normal', { size: fullSize, scale, format: FLUX_FORMAT });
      const gDepth = targets.get('g-depth', { size: fullSize, scale, format: DEPTH_FORMAT, usage: GPUTextureUsage.RENDER_ATTACHMENT });
      const indirect = targets.get('indirect', { size: fullSize, scale, format: HDR_FORMAT });
      const denoise = rsm.denoise > 0 ? targets.get('indirect-blur', { size: fullSize, scale, format: HDR_FORMAT }) : null;
      if (!denoise) targets.release('indirect-blur');
      const scene = post.scene(size);
      const msaaColor = samples > 1 ? targets.get('msaa', { size: fullSize, format: HDR_FORMAT, sampleCount: samples }) : null;
      const depth = targets.get(`depth-x${samples}`, {
        size: fullSize,
        format: DEPTH_FORMAT,
        sampleCount: samples,
        usage: samples > 1 ? GPUTextureUsage.RENDER_ATTACHMENT : GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });

      // The fill is half sky colour, half neutral: a soft skylight, not a blue cast.
      const fill = sky.map((c) => (0.5 * c + 0.45) * sun.fill);
      frame
        .set('viewProj', viewProj)
        .set('lightViewProj', fit.viewProj)
        .set('invLightViewProj', mat4.inverse(fit.viewProj))
        .set('cameraPos', camera.position)
        .set('shadowTexelWorld', fit.texelWorld)
        .set('toSun', toSun)
        .set('intensity', rsm.intensity)
        .set('sunColor', [...sc, 0])
        .set('fillColor', [...fill, 0])
        .set('skyZenith', [...outdoor.zenith, 0])
        .set('skyHorizon', [...outdoor.horizon, 0])
        .set('outdoorGround', [...outdoor.ground, 0])
        .set('fullSize', fullSize)
        .set('lowSize', [gPos.width, gPos.height])
        .set('rsmExtent', fit.halfExtent)
        .set('radius', rsm.radius)
        .set('samples', rsm.samples)
        .set('debugView', look.view)
        .set('upsample', rsm.upsample)
        .set('denoise', rsm.denoise)
        .set('noiseOffset', aa === 'taa' ? taa.frameNoiseOffset : 0)
        .set('normalOffset', 1.5)
        .upload();

      const encoder = device.createCommandEncoder({ label: 'reflective-shadow-maps' });

      // 1. Sun shadow map (depth only), then the RSM: depth + flux + normal.
      const sp = encoder.beginRenderPass({
        label: 'rsm:shadow',
        colorAttachments: [],
        depthStencilAttachment: { view: shadowMap.view, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
      });
      sp.setPipeline(shadowPipeline);
      sp.setBindGroup(0, sceneGroup);
      room.draw(sp);
      sp.end();
      const rp = encoder.beginRenderPass({
        label: 'rsm:rsm',
        colorAttachments: [
          { view: rsmFlux.view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] },
          { view: rsmNormal.view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] },
        ],
        depthStencilAttachment: { view: rsmDepth.view, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
      });
      rp.setPipeline(rsmPipeline);
      rp.setBindGroup(0, sceneGroup);
      room.draw(rp);
      rp.end();

      // 2. Low-resolution G-buffer, then one gather per texel.
      const gp = encoder.beginRenderPass({
        label: 'rsm:gbuffer',
        colorAttachments: [
          { view: gPos.view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] },
          { view: gNormal.view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] },
        ],
        depthStencilAttachment: { view: gDepth.view, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard' },
      });
      gp.setPipeline(gbufferPipeline);
      gp.setBindGroup(0, sceneGroup);
      room.draw(gp, { camera: true });
      gp.end();
      const buffers = buffersGroup(rsmDepth.texture, rsmFlux.texture, rsmNormal.texture, gPos.texture, gNormal.texture);
      const ap = encoder.beginRenderPass({
        label: 'rsm:gather',
        colorAttachments: [{ view: indirect.view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }],
      });
      ap.setPipeline(gatherPipeline);
      ap.setBindGroup(0, sceneGroup);
      ap.setBindGroup(1, buffers);
      ap.draw(3);
      ap.end();
      if (denoise) {
        // Horizontal into the scratch target, vertical back.
        for (const [axis, src, dst] of [[0, indirect, denoise], [1, denoise, indirect]] as const) {
          const dp = encoder.beginRenderPass({
            label: `rsm:denoise-${axis ? 'v' : 'h'}`,
            colorAttachments: [{ view: dst.view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }],
          });
          dp.setPipeline(denoisePipeline);
          dp.setBindGroup(0, sceneGroup);
          dp.setBindGroup(1, buffers);
          dp.setBindGroup(2, denoiseGroup(src.texture));
          dp.draw(3, 1, 0, axis);
          dp.end();
        }
      }

      // 3. Main pass: direct sun + upsampled bounce.
      const mp = encoder.beginRenderPass({
        label: 'rsm:main',
        colorAttachments: [
          msaaColor
            ? { view: msaaColor.view, resolveTarget: scene.view, loadOp: 'clear', storeOp: 'discard', clearValue: [0, 0, 0, 0] }
            : { view: scene.view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] },
        ],
        depthStencilAttachment: { view: depth.view, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: aa === 'taa' ? 'store' : 'discard' },
      });
      mp.setPipeline(mainPipeline);
      mp.setBindGroup(0, sceneGroup);
      mp.setBindGroup(1, buffers);
      mp.setBindGroup(2, mainGroup(shadowMap.texture, indirect.texture));
      room.draw(mp, { camera: true });
      if (look.showRsm) {
        const s = Math.round(Math.min(size.height * 0.26, size.width * 0.2));
        const m = Math.round(12 * size.pixelRatio);
        mp.setPipeline(overlayPipeline);
        for (let i = 0; i < 3; i++) {
          mp.setViewport(m + i * (s + m / 2), size.height - s - m, s, s, 0, 1);
          mp.draw(3, 1, 0, i);
        }
      }
      mp.end();

      const source =
        aa === 'taa' ? taa.resolve(encoder, { color: scene.texture, depth: depth.texture, viewProj: camera.viewProjection }) : undefined;
      post.render(encoder, surface.currentTexture().createView(), { time, background, source });
      device.queue.submit([encoder.finish()]);
    },
    dispose() {
      camera.destroy();
      targets.destroy();
      post.destroy();
      taa.destroy();
      room.destroy();
      frame.destroy();
    },
  };
};

export default create;
