import { mat4, vec3 } from 'wgpu-matrix';
import brdfWgsl from '../../shaders/brdf.wgsl?raw';
import commonWgsl from '../../shaders/common.wgsl?raw';
import fullscreenWgsl from '../../shaders/fullscreen.wgsl?raw';
import sdfWgsl from '../../shaders/sdf.wgsl?raw';
import shadowWgsl from '../../shaders/shadow.wgsl?raw';
import { hexToLinear } from '../../gpu/color';
import { createShader, wgsl } from '../../gpu/shader';
import { TargetPool } from '../../gpu/targets';
import { UniformBlock } from '../../gpu/uniforms';
import { HDR_FORMAT, PostProcess } from '../../post/post';
import { TemporalAA } from '../../post/taa';
import { fitDirectionalShadow } from '../../scene/light-frustum';
import { OrbitCamera } from '../../scene/orbit-camera';
import type { ExperimentFactory } from '../types';
import bakeWgsl from './bake.wgsl?raw';
import sceneWgsl from './scene.wgsl?raw';
import shaderWgsl from './shader.wgsl?raw';

const DEPTH_FORMAT: GPUTextureFormat = 'depth24plus';
const SHADOW_FORMAT: GPUTextureFormat = 'depth32float';
// rgba16float: filterable *and* storage-writable in core WebGPU (r16float is not a storage
// format, r32float is not filterable without an optional feature). Only .r is used.
const BAKE_FORMAT: GPUTextureFormat = 'rgba16float';
/** The baked volume: the static sculptures plus an empty margin of at least 0.1 m. */
const BOUNDS_MIN = [-2.75, -0.15, -2.1] as const;
const BOUNDS_SIZE = [4.35, 2.65, 3.95] as const;
/** The region the PCSS shadow map covers. */
const SHADOW_BOUNDS = { center: vec3.fromValues(-0.3, 0.9, -0.1), radius: 4 };
const MODES = {
  'Hard (1 ray)': 0,
  'Soft: classic (Quilez)': 1,
  'Soft: improved': 2,
  'Baked SDF (3D texture)': 3,
  'Shadow map + PCSS': 4,
  'Split: improved | PCSS': 5,
} as const;

const create: ExperimentFactory = async (ctx) => {
  const { device, surface, params, reducedMotion } = ctx;
  const coarse = window.matchMedia('(pointer: coarse)').matches;

  const sun = params.add(
    {
      azimuth: { value: -70, min: -180, max: 180, step: 1 },
      elevation: { value: 30, min: 4, max: 85, step: 1 },
      sunSize: {
        value: 4,
        min: 0.25,
        max: 20,
        step: 0.05,
        label: 'Sun size (°)',
        hint: 'Angular diameter of the light. Every technique here uses the same value, so their penumbrae are directly comparable.',
      },
      intensity: { value: 3.2, min: 0, max: 10, step: 0.05 },
      color: { value: '#fff0dc', type: 'color' },
      orbit: { value: !reducedMotion, label: 'Orbit sun' },
    },
    'Sun',
  );
  const shadow = params.add(
    {
      mode: { value: 2, options: MODES, hint: 'Split puts the improved SDF estimate on the left and PCSS on the right; move Split at to slide the divider.' },
      shadowSteps: { value: coarse ? 48 : 96, min: 8, max: 192, step: 1, label: 'Shadow steps', hint: 'Max sphere-tracing steps per shadow ray.' },
      bakeRes: {
        value: 96,
        options: { '32': 32, '64': 64, '96': 96, '128': 128, '192': 192 },
        label: 'Bake resolution',
        hint: 'Voxels along the longest side of the baked volume (rebaked by a compute pass on change).',
      },
      mapSize: { value: 1024, options: { '512': 512, '1024': 1024, '2048': 2048 }, label: 'Map size', hint: 'PCSS shadow map, ray-marched from the sun.' },
      depthBias: { value: 1.5, min: 0, max: 8, step: 0.1, label: 'Depth bias', hint: 'PCSS only, in shadow-map texels.' },
      normalOffset: { value: 1.5, min: 0, max: 6, step: 0.1, label: 'Normal offset', hint: 'PCSS only, in shadow-map texels.' },
      splitX: { value: 0.5, min: 0.05, max: 0.95, step: 0.01, label: 'Split at' },
    },
    'Shadows',
  );
  const look = params.add(
    {
      background: { value: '#cfc8bf', type: 'color', hint: 'Page colour the floor fades into.' },
      sky: { value: '#a7bad4', type: 'color' },
      ground: { value: '#7a6a5c', type: 'color' },
      ambient: { value: 0.3, min: 0, max: 1.5, step: 0.01 },
      maxSteps: { value: coarse ? 64 : 112, min: 24, max: 256, step: 1, label: 'Primary steps', hint: 'Max sphere-tracing steps for camera rays.' },
      antialiasing: { value: 'taa', options: { TAA: 'taa', Off: 'off' }, hint: 'Jittered rays + temporal accumulation: anti-aliased edges, and smooth PCSS grain.' },
      view: {
        value: 0,
        options: { Lit: 0, 'Shadow mask': 1, 'Shadow ray steps': 2, 'Primary ray steps': 3, 'Distance field slice (y = 30 cm)': 4 },
        hint: 'Step heatmaps: blue = few, red = the step limit. The slice shows the field the shadow rays see (baked in Baked mode).',
      },
      animate: { value: !reducedMotion },
    },
    'Look & debug',
  );
  const post = await PostProcess.create(ctx, { bloom: 0.02, vignette: 0.12, grain: 0.02 });
  const taa = await TemporalAA.create(device);

  const frame = new UniformBlock(device, 'Frame', {
    invViewProj: 'mat4x4f',
    viewProj: 'mat4x4f',
    lightViewProj: 'mat4x4f',
    invLightViewProj: 'mat4x4f',
    cameraPos: 'vec3f',
    bobY: 'f32',
    sunDir: 'vec3f',
    sunTan: 'f32',
    sunColor: 'vec4f',
    skyColor: 'vec4f',
    groundColor: 'vec4f',
    boundsMin: 'vec3f',
    shadowTexelWorld: 'f32',
    boundsSize: 'vec3f',
    shadowDepthRange: 'f32',
    resolution: 'vec2f',
    shadowMode: 'u32',
    debugView: 'u32',
    maxSteps: 'u32',
    shadowSteps: 'u32',
    shadowExtent: 'f32',
    depthBias: 'f32',
    normalOffset: 'f32',
    noiseOffset: 'f32',
    splitX: 'f32',
    fadeStart: 'f32',
    fadeEnd: 'f32',
  });

  const [module, bakeModule] = await Promise.all([
    createShader(device, 'sdf-soft-shadows', wgsl(commonWgsl, fullscreenWgsl, sdfWgsl, brdfWgsl, frame.wgslStruct, sceneWgsl, shadowWgsl, shaderWgsl)),
    createShader(device, 'sdf-soft-shadows:bake', wgsl(sdfWgsl, frame.wgslStruct, sceneWgsl, bakeWgsl)),
  ]);

  const bindGroupLayout = device.createBindGroupLayout({
    label: 'sdf-soft-shadows',
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '3d' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
    ],
  });
  const [mainPipeline, lightPipeline, bakePipeline] = await Promise.all([
    device.createRenderPipelineAsync({
      label: 'sdf-main',
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      vertex: { module, entryPoint: 'vs_fullscreen' },
      fragment: { module, entryPoint: 'fs_main', targets: [{ format: HDR_FORMAT }] },
      // Depth written from the hit point, for TAA's reprojection.
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: 'always' },
    }),
    device.createRenderPipelineAsync({
      label: 'sdf-light-depth',
      layout: 'auto',
      vertex: { module, entryPoint: 'vs_fullscreen' },
      fragment: { module, entryPoint: 'fs_light_depth', targets: [] },
      depthStencil: { format: SHADOW_FORMAT, depthWriteEnabled: true, depthCompare: 'always' },
    }),
    device.createComputePipelineAsync({ label: 'sdf-bake', layout: 'auto', compute: { module: bakeModule, entryPoint: 'cs_bake' } }),
  ]);

  const linearSampler = device.createSampler({ label: 'sdf-baked', magFilter: 'linear', minFilter: 'linear' });
  const compareSampler = device.createSampler({ label: 'sdf-shadow-compare', compare: 'less-equal', magFilter: 'linear', minFilter: 'linear' });
  const lightBindGroup = device.createBindGroup({
    layout: lightPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: frame.buffer } }],
  });

  // --- Baked volume and shadow map (recreated when their resolution changes) ------
  let baked: GPUTexture | null = null;
  let shadowMap: GPUTexture | null = null;
  let mainBindGroup: GPUBindGroup | null = null;
  let lightKey = '';
  const longest = Math.max(...BOUNDS_SIZE);
  const ensureResources = (encoder: GPUCommandEncoder, bakeRes: number, mapSize: number) => {
    const dims = BOUNDS_SIZE.map((s) => Math.max(4, Math.round((s / longest) * bakeRes)));
    let changed = false;
    if (!baked || baked.width !== dims[0] || baked.height !== dims[1] || baked.depthOrArrayLayers !== dims[2]) {
      baked?.destroy();
      baked = device.createTexture({
        label: 'sdf-baked',
        size: dims,
        dimension: '3d',
        format: BAKE_FORMAT,
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      });
      // The bake: one compute thread per voxel, once (the baked objects never move).
      const pass = encoder.beginComputePass({ label: 'sdf-bake' });
      pass.setPipeline(bakePipeline);
      pass.setBindGroup(
        0,
        device.createBindGroup({
          layout: bakePipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: frame.buffer } },
            { binding: 1, resource: baked.createView() },
          ],
        }),
      );
      pass.dispatchWorkgroups(Math.ceil(dims[0] / 4), Math.ceil(dims[1] / 4), Math.ceil(dims[2] / 4));
      pass.end();
      changed = true;
    }
    if (shadowMap?.width !== mapSize) {
      shadowMap?.destroy();
      shadowMap = device.createTexture({
        label: 'sdf-shadow-map',
        size: [mapSize, mapSize],
        format: SHADOW_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      lightKey = '';
      changed = true;
    }
    if (changed) {
      mainBindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: frame.buffer } },
          { binding: 1, resource: baked.createView() },
          { binding: 2, resource: linearSampler },
          { binding: 3, resource: shadowMap!.createView() },
          { binding: 4, resource: compareSampler },
        ],
      });
    }
  };

  const targets = new TargetPool(device);
  const camera = new OrbitCamera(surface.canvas, { target: [-0.2, 0.7, -0.1], distance: 8, yaw: 0.55, pitch: 0.36, maxDistance: 25 });
  let animTime = 0;
  let sunSpin = 0;

  return {
    frame({ time, dt, size }) {
      if (look.animate) animTime += dt;
      if (sun.orbit) sunSpin += dt * 0.12;
      camera.update(dt, size.width / size.height);
      const aa = look.antialiasing;
      if (aa !== 'taa') taa.reset();
      const viewProj = aa === 'taa' ? taa.jitter(camera.viewProjection, size) : camera.viewProjection;

      const az = (sun.azimuth * Math.PI) / 180 + sunSpin;
      const el = (sun.elevation * Math.PI) / 180;
      const toSun = vec3.fromValues(Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az));
      const fit = fitDirectionalShadow({ toLight: toSun, bounds: SHADOW_BOUNDS, mapSize: shadow.mapSize, casterPadding: 1 });
      const bobY = 0.62 + 0.18 * Math.sin(animTime * 0.9);

      const lc = hexToLinear(sun.color);
      const sky = hexToLinear(look.sky);
      const ground = hexToLinear(look.ground);
      frame
        .set('invViewProj', mat4.inverse(viewProj))
        .set('viewProj', viewProj)
        .set('lightViewProj', fit.viewProj)
        .set('invLightViewProj', mat4.inverse(fit.viewProj))
        .set('cameraPos', camera.position)
        .set('bobY', bobY)
        .set('sunDir', toSun)
        .set('sunTan', Math.tan(((sun.sunSize / 2) * Math.PI) / 180))
        .set('sunColor', [lc[0] * sun.intensity, lc[1] * sun.intensity, lc[2] * sun.intensity, 0])
        .set('skyColor', [sky[0] * look.ambient, sky[1] * look.ambient, sky[2] * look.ambient, 0])
        .set('groundColor', [ground[0] * look.ambient, ground[1] * look.ambient, ground[2] * look.ambient, 0])
        .set('boundsMin', BOUNDS_MIN)
        .set('boundsSize', BOUNDS_SIZE)
        .set('shadowTexelWorld', fit.texelWorld)
        .set('shadowDepthRange', fit.depthRange)
        .set('shadowExtent', fit.halfExtent)
        .set('resolution', [size.width, size.height])
        .set('shadowMode', shadow.mode)
        .set('debugView', look.view)
        .set('maxSteps', look.maxSteps)
        .set('shadowSteps', shadow.shadowSteps)
        .set('depthBias', shadow.depthBias)
        .set('normalOffset', shadow.normalOffset)
        .set('noiseOffset', aa === 'taa' ? taa.frameNoiseOffset : 0)
        .set('splitX', shadow.splitX)
        .set('fadeStart', 7)
        .set('fadeEnd', 18)
        .upload();

      const encoder = device.createCommandEncoder({ label: 'sdf-soft-shadows' });
      ensureResources(encoder, shadow.bakeRes, shadow.mapSize);

      // PCSS comparison: re-march the shadow map only when the sun or the pearl moved.
      if (shadow.mode >= 4) {
        const key = `${toSun.join()}:${bobY}:${shadow.mapSize}`;
        if (key !== lightKey) {
          lightKey = key;
          const lp = encoder.beginRenderPass({
            label: 'sdf-light-depth',
            colorAttachments: [],
            depthStencilAttachment: { view: shadowMap!.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
          });
          lp.setPipeline(lightPipeline);
          lp.setBindGroup(0, lightBindGroup);
          lp.draw(3);
          lp.end();
        }
      }

      const scene = post.scene(size);
      const depth = targets.get('depth', {
        size: [size.width, size.height],
        format: DEPTH_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      const pass = encoder.beginRenderPass({
        label: 'sdf-main',
        colorAttachments: [{ view: scene.view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }],
        depthStencilAttachment: { view: depth.view, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
      });
      pass.setPipeline(mainPipeline);
      pass.setBindGroup(0, mainBindGroup!);
      pass.draw(3);
      pass.end();

      const source =
        aa === 'taa' ? taa.resolve(encoder, { color: scene.texture, depth: depth.texture, viewProj: camera.viewProjection }) : undefined;
      post.render(encoder, surface.currentTexture().createView(), {
        time,
        raw: look.view !== 0,
        background: hexToLinear(look.background),
        source,
      });
      device.queue.submit([encoder.finish()]);
    },
    dispose() {
      camera.destroy();
      baked?.destroy();
      shadowMap?.destroy();
      targets.destroy();
      post.destroy();
      taa.destroy();
      frame.destroy();
    },
  };
};

export default create;
