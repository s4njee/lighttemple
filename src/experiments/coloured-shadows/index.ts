import { mat4, vec3 } from 'wgpu-matrix';
import brdfWgsl from '../../shaders/brdf.wgsl?raw';
import commonWgsl from '../../shaders/common.wgsl?raw';
import fullscreenWgsl from '../../shaders/fullscreen.wgsl?raw';
import noiseWgsl from '../../shaders/noise.wgsl?raw';
import { hexToLinear } from '../../gpu/color';
import { MESH_VERTEX_LAYOUT, MESH_VERTEX_WGSL, box, plane, sphere, uploadMesh, type Mesh } from '../../gpu/geometry';
import { createShader, wgsl } from '../../gpu/shader';
import { TargetPool } from '../../gpu/targets';
import { UniformBlock, uniformStruct } from '../../gpu/uniforms';
import { HDR_FORMAT, PostProcess } from '../../post/post';
import { TemporalAA } from '../../post/taa';
import { fitDirectionalShadow } from '../../scene/light-frustum';
import { OrbitCamera } from '../../scene/orbit-camera';
import { PROP_WGSL, PropSet, type Prop } from '../../scene/props';
import type { ExperimentFactory } from '../types';
import shaderWgsl from './shader.wgsl?raw';

const MSAA = 4;
const DEPTH_FORMAT: GPUTextureFormat = 'depth24plus';
const SHADOW_FORMAT: GPUTextureFormat = 'depth32float';
const TINT_FORMAT: GPUTextureFormat = 'rgba16float';
const PASS_STRIDE = 256; // dynamic-offset alignment
const MODE = { Transmittance: 0, StochasticGrey: 1, StochasticRgb: 2 } as const;
/** Stained-glass prop kinds (the prop's `extra`); see shader.wgsl. */
const STAINED_A = 1;
const STAINED_B = 2;
const ACRYLIC = 3;

const create: ExperimentFactory = async (ctx) => {
  const { device, surface, params, reducedMotion } = ctx;

  const shadow = params.add(
    {
      mode: {
        value: MODE.Transmittance,
        options: { 'Transmittance map': MODE.Transmittance, 'Stochastic (grey)': MODE.StochasticGrey, 'Stochastic (RGB, 3 maps)': MODE.StochasticRgb },
        hint: 'Transmittance map: one depth pass + one tint pass (rgba16f), exact colours, but only the nearest translucent layer\'s depth. Stochastic: translucent casters write depth with a hashed per-texel discard; grey uses one map, RGB one map per channel (3 passes). Needs filtering or TAA to hide the noise.',
      },
      mapSize: { value: 2048, options: { '1024': 1024, '2048': 2048 }, label: 'Map size', hint: 'Depth (4 B/texel) + tint map (8 B/texel): 48 MB at 2048², 12 MB at 1024².' },
      filterRadius: { value: 1.5, min: 0, max: 8, step: 0.1, label: 'Filter radius', hint: 'Poisson filter radius in shadow-map texels (16 taps). Stochastic shadows need a wider filter, or TAA.' },
      depthBias: { value: 0.0004, min: 0, max: 0.005, step: 0.0001, label: 'Depth bias' },
      normalOffset: { value: 1.5, min: 0, max: 6, step: 0.1, label: 'Normal offset', hint: 'In shadow-map texels.' },
    },
    'Shadows',
  );
  const sun = params.add(
    {
      azimuth: { value: -164, min: -180, max: 180, step: 1, hint: 'The sun is behind the wall; patterns fall toward you.' },
      elevation: { value: 34, min: 8, max: 80, step: 1 },
      intensity: { value: 6, min: 0, max: 12, step: 0.05 },
      color: { value: '#ffe3bd', type: 'color' },
      ambient: { value: 0.4, min: 0, max: 1.5, step: 0.01 },
      drift: { value: !reducedMotion, label: 'Drift sun', hint: 'Slowly sway the sun so the coloured patches move.' },
    },
    'Sun',
  );
  const look = params.add(
    {
      glassGlow: { value: 0.12, min: 0, max: 0.6, step: 0.01, label: 'Glass glow', hint: 'Light scattered forward through the glass when it is backlit.' },
      antialiasing: {
        value: 'msaa',
        options: { 'MSAA 4×': 'msaa', TAA: 'taa' },
        hint: 'TAA re-rolls the stochastic hash and the filter rotation every frame and accumulates them: the stochastic modes converge to smooth, correct partial shadow.',
      },
      view: { value: 0, options: { Lit: 0, 'Sun reaching surfaces': 1 }, hint: 'Shows only the filtered sun term (visibility × tint) on every surface.' },
      map: {
        value: 0,
        options: { Off: 0, 'Transmittance map': 1, 'Translucent depth': 2, 'Opaque depth': 3 },
        label: 'Show map',
        hint: 'Light-space maps in a corner. The transmittance and translucent-depth maps are only rendered in transmittance mode.',
      },
    },
    'Look & debug',
  );
  const post = await PostProcess.create(ctx, { bloom: 0.05, vignette: 0.18, grain: 0.02 });
  const taa = await TemporalAA.create(device);

  // --- Scene -----------------------------------------------------------------------------
  const meshes: Record<string, Mesh> = {
    floor: uploadMesh(device, plane(80, 1), 'floor'),
    unit: uploadMesh(device, box(1, 1, 1), 'unit-box'),
    sphere: uploadMesh(device, sphere(1, 48, 32), 'sphere'),
  };
  /** A box spanning [x0, x1] × [y0, y1] at depth z (thickness d). */
  const slab = (x0: number, x1: number, y0: number, y1: number, z: number, d: number) =>
    mat4.scale(mat4.translation([(x0 + x1) / 2, (y0 + y1) / 2, z]), [x1 - x0, y1 - y0, d]);
  const wall = { color: '#cfc3b2', roughness: 0.9, mesh: meshes.unit };
  const WALL_Z = -1;
  const WALL_D = 0.3;
  const opaque: Prop[] = [
    { mesh: meshes.floor, color: '#e0d6c6', roughness: 0.85, castsShadow: false },
    // A wall with two lancet openings: x in [-3.5, -2.3] and [-1.3, -0.1], y in [0.9, 3.3].
    { ...wall, transform: () => slab(-4.6, 1.2, 0, 0.9, WALL_Z, WALL_D) },
    { ...wall, transform: () => slab(-4.6, 1.2, 3.3, 4.2, WALL_Z, WALL_D) },
    { ...wall, transform: () => slab(-4.6, -3.5, 0.89, 3.31, WALL_Z, WALL_D) },
    { ...wall, transform: () => slab(-2.3, -1.3, 0.89, 3.31, WALL_Z, WALL_D) },
    { ...wall, transform: () => slab(-0.1, 1.2, 0.89, 3.31, WALL_Z, WALL_D) },
    // Receivers inside the coloured light.
    { mesh: meshes.sphere, color: '#f1ece4', roughness: 0.45, transform: () => mat4.scale(mat4.translation([0.1, 0.42, 2.0]), [0.42, 0.42, 0.42]) },
    { mesh: meshes.unit, color: '#ece6dc', roughness: 0.7, transform: () => slab(-2.5, -1.9, 0, 0.45, 3.2, 0.6) },
  ];
  const acrylic = (x: number, z: number, yaw: number, color: string): Prop => ({
    mesh: meshes.unit,
    color,
    roughness: 0.05,
    extra: ACRYLIC,
    transform: () => mat4.scale(mat4.rotateY(mat4.translation([x, 0.72, z]), yaw), [1, 1.4, 0.04]),
  });
  const glass: Prop[] = [
    // Panes overlap the opening by a few cm (hidden in the wall), so no light leaks through the seam.
    { mesh: meshes.unit, color: '#ffffff', roughness: 0.1, extra: STAINED_A, transform: () => slab(-3.53, -2.27, 0.87, 3.33, WALL_Z, 0.03) },
    { mesh: meshes.unit, color: '#ffffff', roughness: 0.1, extra: STAINED_B, transform: () => slab(-1.33, -0.07, 0.87, 3.33, WALL_Z, 0.03) },
    // Tinted acrylic sheets in open sun: overlapping shadows multiply (cyan × magenta = blue...).
    acrylic(2.2, -0.45, 0.3, '#45dcef'),
    acrylic(2.95, 0.15, -0.15, '#f04cc8'),
    acrylic(3.7, -0.3, 0.35, '#ffd23a'),
  ];
  const opaqueSet = new PropSet(device, opaque, 'coloured-opaque');
  const glassSet = new PropSet(device, glass, 'coloured-glass');
  opaqueSet.update(0);
  glassSet.update(0);

  const frame = new UniformBlock(device, 'Frame', {
    viewProj: 'mat4x4f',
    invViewProj: 'mat4x4f',
    lightViewProj: 'mat4x4f',
    cameraPos: 'vec3f',
    mode: 'u32',
    toLight: 'vec3f',
    debugView: 'u32',
    sunColor: 'vec3f',
    ambient: 'f32',
    horizonColor: 'vec3f',
    fogStart: 'f32',
    zenithColor: 'vec3f',
    fogEnd: 'f32',
    shadowTexelWorld: 'f32',
    normalOffset: 'f32',
    depthBias: 'f32',
    tintBias: 'f32',
    filterRadius: 'f32',
    noiseOffset: 'f32',
    glassGlow: 'f32',
    overlay: 'u32',
  });
  const PASS_FIELDS = { channel: 'u32', seed: 'f32' } as const;

  const module = await createShader(
    device,
    'coloured-shadows',
    wgsl(commonWgsl, fullscreenWgsl, noiseWgsl, MESH_VERTEX_WGSL, PROP_WGSL, brdfWgsl, frame.wgslStruct, uniformStruct('ShadowPass', PASS_FIELDS), shaderWgsl),
  );

  // --- Layouts & pipelines -------------------------------------------------------------
  const VF = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
  const mainLayout = device.createBindGroupLayout({
    label: 'coloured-main',
    entries: [
      { binding: 0, visibility: VF, buffer: { type: 'uniform' } },
      { binding: 1, visibility: VF, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth', viewDimension: '2d-array' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
      { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
    ],
  });
  const lightLayout = device.createBindGroupLayout({
    label: 'coloured-light',
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      { binding: 1, visibility: VF, buffer: { type: 'read-only-storage' } },
    ],
  });
  const passLayout = device.createBindGroupLayout({
    label: 'coloured-shadow-pass',
    entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 16 } }],
  });
  const layout = (...g: GPUBindGroupLayout[]) => device.createPipelineLayout({ bindGroupLayouts: g });

  const multiply: GPUBlendState = {
    color: { srcFactor: 'zero', dstFactor: 'src' },
    alpha: { srcFactor: 'zero', dstFactor: 'one' },
  };
  const add: GPUBlendState = {
    color: { srcFactor: 'one', dstFactor: 'one' },
    alpha: { srcFactor: 'zero', dstFactor: 'one' },
  };
  const shadowDepth = (write: boolean): GPUDepthStencilState => ({ format: SHADOW_FORMAT, depthWriteEnabled: write, depthCompare: 'less' });

  const makeScenePipelines = (count: number) => {
    const scene = (label: string, fragment: string, blend: GPUBlendState | undefined, write: boolean, cullMode: GPUCullMode) =>
      device.createRenderPipelineAsync({
        label: `coloured:${label}-x${count}`,
        layout: layout(mainLayout),
        vertex: { module, entryPoint: 'vs_scene', buffers: [MESH_VERTEX_LAYOUT] },
        fragment: { module, entryPoint: fragment, targets: [{ format: HDR_FORMAT, blend }] },
        primitive: { cullMode },
        depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: write, depthCompare: 'less' },
        multisample: { count },
      });
    const fullscreen = (fragment: string) =>
      device.createRenderPipelineAsync({
        label: `coloured:${fragment}-x${count}`,
        layout: layout(mainLayout),
        vertex: { module, entryPoint: 'vs_fullscreen' },
        fragment: { module, entryPoint: fragment, targets: [{ format: HDR_FORMAT }] },
        depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: 'always' },
        multisample: { count },
      });
    return Promise.all([
      fullscreen('fs_sky'),
      scene('opaque', 'fs_opaque', undefined, true, 'back'),
      // Only the camera-facing side of each pane, so the filter applies once.
      scene('glass-filter', 'fs_glass_filter', multiply, false, 'back'),
      scene('glass-light', 'fs_glass_light', add, false, 'back'),
      fullscreen('fs_overlay'),
    ]).then(([sky, opaque, glassFilter, glassLight, overlay]) => ({ sky, opaque, glassFilter, glassLight, overlay }));
  };

  const [opaqueShadowPipeline, transmitPipeline, stochasticPipeline, msaaPipelines, singlePipelines] = await Promise.all([
    device.createRenderPipelineAsync({
      label: 'coloured:opaque-shadow',
      layout: layout(lightLayout),
      vertex: { module, entryPoint: 'vs_shadow', buffers: [MESH_VERTEX_LAYOUT] },
      primitive: { cullMode: 'back' },
      depthStencil: shadowDepth(true),
    }),
    device.createRenderPipelineAsync({
      label: 'coloured:transmittance',
      layout: layout(lightLayout),
      vertex: { module, entryPoint: 'vs_transmit', buffers: [MESH_VERTEX_LAYOUT] },
      fragment: {
        module,
        entryPoint: 'fs_transmit',
        targets: [
          {
            format: TINT_FORMAT,
            // rgb: dst *= T (order-independent); a: keep the nearest translucent depth.
            blend: {
              color: { operation: 'add', srcFactor: 'zero', dstFactor: 'src' },
              alpha: { operation: 'min', srcFactor: 'one', dstFactor: 'one' },
            },
          },
        ],
      },
      // The side facing the light only, so each pane filters once.
      primitive: { cullMode: 'back' },
      depthStencil: shadowDepth(false),
    }),
    device.createRenderPipelineAsync({
      label: 'coloured:stochastic',
      layout: layout(lightLayout, passLayout),
      vertex: { module, entryPoint: 'vs_transmit', buffers: [MESH_VERTEX_LAYOUT] },
      fragment: { module, entryPoint: 'fs_stochastic', targets: [] },
      primitive: { cullMode: 'back' },
      depthStencil: shadowDepth(true),
    }),
    makeScenePipelines(MSAA),
    makeScenePipelines(1),
  ]);

  // --- Resources -------------------------------------------------------------------------
  const cmp = device.createSampler({ compare: 'less-equal', magFilter: 'linear', minFilter: 'linear' });
  const cmpPoint = device.createSampler({ compare: 'less-equal' });
  const passBuffer = device.createBuffer({ label: 'shadow-passes', size: 4 * PASS_STRIDE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const passGroup = device.createBindGroup({ layout: passLayout, entries: [{ binding: 0, resource: { buffer: passBuffer, size: 16 } }] });
  const passData = new ArrayBuffer(4 * PASS_STRIDE);
  const lightGroups = [opaqueSet, glassSet].map((set) =>
    device.createBindGroup({
      layout: lightLayout,
      entries: [
        { binding: 0, resource: { buffer: frame.buffer } },
        { binding: 1, resource: { buffer: set.buffer } },
      ],
    }),
  );
  const targets = new TargetPool(device);

  // Shadow maps: a depth array (1 layer, or 3 for per-channel stochastic) and the tint map.
  let maps: { key: string; depth: GPUTexture; tint: GPUTexture; layers: GPUTextureView[]; groups: GPUBindGroup[] } | null = null;
  const ensureMaps = (size: number, layerCount: number) => {
    const key = `${size}:${layerCount}`;
    if (maps?.key === key) return maps;
    maps?.depth.destroy();
    maps?.tint.destroy();
    const depth = device.createTexture({
      label: 'coloured-shadow-depth',
      size: [size, size, layerCount],
      format: SHADOW_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    const tint = device.createTexture({
      label: 'coloured-tint',
      size: [size, size],
      format: TINT_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    const arrayView = depth.createView({ dimension: '2d-array' });
    const groups = [opaqueSet, glassSet].map((set) =>
      device.createBindGroup({
        layout: mainLayout,
        entries: [
          { binding: 0, resource: { buffer: frame.buffer } },
          { binding: 1, resource: { buffer: set.buffer } },
          { binding: 2, resource: arrayView },
          { binding: 3, resource: cmp },
          { binding: 4, resource: tint.createView() },
          { binding: 5, resource: cmpPoint },
        ],
      }),
    );
    const layers = Array.from({ length: layerCount }, (_, i) => depth.createView({ dimension: '2d', baseArrayLayer: i, arrayLayerCount: 1 }));
    maps = { key, depth, tint, layers, groups };
    return maps;
  };

  const camera = new OrbitCamera(surface.canvas, { target: [0.2, 1.2, 0.9], distance: 11, yaw: 0.3, pitch: 0.3, maxDistance: 22 });
  let sunTime = 0;
  let frameIndex = 0;

  return {
    frame({ time, dt, size }) {
      if (sun.drift) sunTime += dt;
      camera.update(dt, size.width / size.height);
      const mode = shadow.mode;
      const m = ensureMaps(shadow.mapSize, mode === MODE.StochasticRgb ? 3 : 1);
      const useTaa = look.antialiasing === 'taa';
      const samples = useTaa ? 1 : MSAA;
      const pipes = useTaa ? singlePipelines : msaaPipelines;
      if (!useTaa) taa.reset();
      frameIndex++;

      const az = ((sun.azimuth + 7 * Math.sin(sunTime * 0.12)) * Math.PI) / 180;
      const el = (sun.elevation * Math.PI) / 180;
      const toSun = vec3.fromValues(Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az));
      const fit = fitDirectionalShadow({ toLight: toSun, bounds: { center: vec3.fromValues(-0.5, 1.4, 0.6), radius: 6.2 }, mapSize: shadow.mapSize });

      const lc = hexToLinear(sun.color);
      const viewProj = useTaa ? taa.jitter(camera.viewProjection, size) : camera.viewProjection;
      frame
        .set('viewProj', viewProj)
        .set('invViewProj', mat4.inverse(viewProj))
        .set('lightViewProj', fit.viewProj)
        .set('cameraPos', camera.position)
        .set('mode', mode)
        .set('toLight', toSun)
        .set('debugView', look.view)
        .set('sunColor', lc.map((c) => c * sun.intensity))
        .set('ambient', sun.ambient)
        .set('horizonColor', hexToLinear('#f3d8b6'))
        .set('zenithColor', hexToLinear('#5b84c2'))
        .set('fogStart', 14)
        .set('fogEnd', 45)
        .set('shadowTexelWorld', fit.texelWorld)
        .set('normalOffset', shadow.normalOffset)
        .set('depthBias', shadow.depthBias)
        // ~2 cm in light depth: the translucent depth is stored at half precision.
        .set('tintBias', 0.02 / fit.depthRange)
        .set('filterRadius', shadow.filterRadius)
        .set('noiseOffset', useTaa ? taa.frameNoiseOffset : 0)
        .set('glassGlow', look.glassGlow)
        .set('overlay', look.map)
        .upload();

      // Per-pass channel (0..2, 3 = grey) and the stochastic hash seed (re-rolled under TAA).
      const seed = useTaa ? (frameIndex % 64) * 17.371 : 0;
      for (let c = 0; c < 4; c++) {
        new Uint32Array(passData, c * PASS_STRIDE, 1)[0] = c;
        new Float32Array(passData, c * PASS_STRIDE + 4, 1)[0] = seed + c * 3.1;
      }
      device.queue.writeBuffer(passBuffer, 0, passData);

      const encoder = device.createCommandEncoder({ label: 'coloured-shadows' });
      const depthPass = (view: GPUTextureView) =>
        encoder.beginRenderPass({
          label: 'coloured-depth',
          colorAttachments: [],
          depthStencilAttachment: { view, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
        });
      if (mode === MODE.Transmittance) {
        // 1. Opaque depth.
        const dp = depthPass(m.layers[0]);
        dp.setPipeline(opaqueShadowPipeline);
        dp.setBindGroup(0, lightGroups[0]);
        opaqueSet.draw(dp, { castersOnly: true });
        dp.end();
        // 2. Tint: translucent casters multiplied together, depth-tested (read-only) against 1.
        const tp = encoder.beginRenderPass({
          label: 'coloured-tint',
          colorAttachments: [{ view: m.tint.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [1, 1, 1, 1] }],
          depthStencilAttachment: { view: m.layers[0], depthReadOnly: true },
        });
        tp.setPipeline(transmitPipeline);
        tp.setBindGroup(0, lightGroups[1]);
        glassSet.draw(tp);
        tp.end();
      } else {
        // Stochastic: opaque + hashed translucent depth, one map (grey) or one per channel.
        m.layers.forEach((view, layer) => {
          const sp = depthPass(view);
          sp.setPipeline(opaqueShadowPipeline);
          sp.setBindGroup(0, lightGroups[0]);
          opaqueSet.draw(sp, { castersOnly: true });
          sp.setPipeline(stochasticPipeline);
          sp.setBindGroup(0, lightGroups[1]);
          sp.setBindGroup(1, passGroup, [(mode === MODE.StochasticGrey ? 3 : layer) * PASS_STRIDE]);
          glassSet.draw(sp);
          sp.end();
        });
      }

      // 3. Camera pass.
      const full = [size.width, size.height] as const;
      const scene = post.scene(size);
      const color = samples > 1 ? targets.get('msaa', { size: full, format: HDR_FORMAT, sampleCount: samples }) : null;
      const depth = targets.get(`depth-x${samples}`, {
        size: full,
        format: DEPTH_FORMAT,
        sampleCount: samples,
        usage: samples > 1 ? GPUTextureUsage.RENDER_ATTACHMENT : GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      const pass = encoder.beginRenderPass({
        label: 'coloured-main',
        colorAttachments: [
          color
            ? { view: color.view, resolveTarget: scene.view, loadOp: 'clear', storeOp: 'discard', clearValue: [0, 0, 0, 1] }
            : { view: scene.view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] },
        ],
        depthStencilAttachment: { view: depth.view, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: useTaa ? 'store' : 'discard' },
      });
      pass.setBindGroup(0, m.groups[0]);
      pass.setPipeline(pipes.sky);
      pass.draw(3);
      pass.setPipeline(pipes.opaque);
      opaqueSet.draw(pass);
      pass.setBindGroup(0, m.groups[1]);
      pass.setPipeline(pipes.glassFilter);
      glassSet.draw(pass);
      pass.setPipeline(pipes.glassLight);
      glassSet.draw(pass);
      if (look.map !== 0) {
        const s = Math.round(Math.min(size.width, size.height) * 0.32);
        const margin = Math.round(16 * size.pixelRatio);
        pass.setViewport(margin, size.height - s - margin, s, s, 0, 1);
        pass.setPipeline(pipes.overlay);
        pass.draw(3);
      }
      pass.end();

      const source = useTaa ? taa.resolve(encoder, { color: scene.texture, depth: depth.texture, viewProj: camera.viewProjection }) : undefined;
      post.render(encoder, surface.currentTexture().createView(), { time, raw: look.view !== 0, source });
      device.queue.submit([encoder.finish()]);
    },
    dispose() {
      camera.destroy();
      maps?.depth.destroy();
      maps?.tint.destroy();
      passBuffer.destroy();
      opaqueSet.destroy();
      glassSet.destroy();
      Object.values(meshes).forEach((mesh) => mesh.destroy());
      targets.destroy();
      frame.destroy();
      taa.destroy();
      post.destroy();
    },
  };
};

export default create;
