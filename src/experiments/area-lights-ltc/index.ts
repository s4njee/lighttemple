import { vec3, type Vec3 } from 'wgpu-matrix';
import commonWgsl from '../../shaders/common.wgsl?raw';
import fullscreenWgsl from '../../shaders/fullscreen.wgsl?raw';
import brdfWgsl from '../../shaders/brdf.wgsl?raw';
import { MESH_VERTEX_LAYOUT, MESH_VERTEX_WGSL } from '../../gpu/geometry';
import { createLtcTables, ltcWgsl } from '../../gpu/ltc';
import { createShader, wgsl } from '../../gpu/shader';
import { TargetPool } from '../../gpu/targets';
import { UniformBlock } from '../../gpu/uniforms';
import { hexToLinear } from '../../gpu/color';
import { HDR_FORMAT, PostProcess } from '../../post/post';
import { OrbitCamera } from '../../scene/orbit-camera';
import { PROP_WGSL, PropSet } from '../../scene/props';
import type { ParamSchema, ParamSink } from '../../ui/params';
import type { ExperimentFactory } from '../types';
import { createMeshes, productScene, sphereChart, type StudioScene } from './scene';
import shaderWgsl from './shader.wgsl?raw';

const MSAA = 4;
const DEPTH_FORMAT: GPUTextureFormat = 'depth24plus';
const MAX_PROXIES = 16;
const LIGHT_COUNT = 3;
/** Reference view: stop accumulating after this many frames (the average has converged). */
const MAX_ACCUMULATED = 2048;
const SHAPE = { off: 0, rect: 1, disc: 2 } as const;
const MODEL = { ltc: 0, point: 1, reference: 2 } as const;
const VIEW = { lit: 0, specular: 1, diffuse: 2, footprint: 3, shadows: 4 } as const;

type Shape = keyof typeof SHAPE;

interface LightDefaults {
  prefix: string;
  label: string;
  shape: Shape;
  intensity: number;
  color: string;
  width: number;
  height: number;
  azimuth: number;
  elevation: number;
  distance: number;
  roll: number;
}

/** A classic three-light product setup: big key, a narrow rim strip behind, a round top light. */
const LIGHTS: LightDefaults[] = [
  { prefix: 'key', label: 'Key light', shape: 'rect', intensity: 10, color: '#fff1dc', width: 1.8, height: 2.4, azimuth: -48, elevation: 30, distance: 2.8, roll: 0 },
  { prefix: 'strip', label: 'Strip light (rim)', shape: 'rect', intensity: 16, color: '#d6e4ff', width: 0.4, height: 2.6, azimuth: 128, elevation: 12, distance: 2.9, roll: 0 },
  { prefix: 'top', label: 'Top light', shape: 'disc', intensity: 6, color: '#ffffff', width: 1.6, height: 1.6, azimuth: 10, elevation: 70, distance: 2.6, roll: 0 },
];

const area = (shape: Shape, w: number, h: number) => (shape === 'disc' ? (Math.PI / 4) * w * h : w * h);

/** One light's params; keys are prefixed (keyWidth, stripWidth, …) because they double as URL keys. */
function addLight(params: ParamSink, d: LightDefaults) {
  const key = (s: string) => `${d.prefix}${s}`;
  const schema: ParamSchema = {
    [key('Shape')]: { value: d.shape, options: { Rectangle: 'rect', 'Disc / ellipse': 'disc', Off: 'off' }, label: 'Shape' },
    [key('Intensity')]: {
      value: d.intensity,
      min: 0,
      max: 40,
      step: 0.1,
      label: 'Intensity',
      hint: 'Radiance at the default size. With "Constant power" on, resizing keeps the light\'s power: bigger is softer and dimmer per m².',
    },
    [key('Color')]: { value: d.color, type: 'color', label: 'Colour' },
    [key('Width')]: { value: d.width, min: 0.05, max: 4, step: 0.01, label: 'Width' },
    [key('Height')]: { value: d.height, min: 0.05, max: 4, step: 0.01, label: 'Height' },
    [key('Azimuth')]: { value: d.azimuth, min: -180, max: 180, step: 1, label: 'Azimuth', hint: 'Around the subject; 0° is the camera side.' },
    [key('Elevation')]: { value: d.elevation, min: -10, max: 90, step: 1, label: 'Elevation' },
    [key('Distance')]: { value: d.distance, min: 1, max: 8, step: 0.05, label: 'Distance', hint: 'From the subject; the light always faces it.' },
    [key('Roll')]: { value: d.roll, min: -90, max: 90, step: 1, label: 'Roll', hint: 'Turns the light about its own axis.' },
  };
  const v = params.add(schema, d.label) as Record<string, unknown>;
  const num = (s: string) => v[key(s)] as number;
  return {
    defaults: d,
    get shape() {
      return v[key('Shape')] as Shape;
    },
    get color() {
      return v[key('Color')] as string;
    },
    get intensity() {
      return num('Intensity');
    },
    get width() {
      return num('Width');
    },
    get height() {
      return num('Height');
    },
    get azimuth() {
      return num('Azimuth');
    },
    get elevation() {
      return num('Elevation');
    },
    get distance() {
      return num('Distance');
    },
    get roll() {
      return num('Roll');
    },
  };
}

type LightParams = ReturnType<typeof addLight>;

const create: ExperimentFactory = async (ctx) => {
  const { device, surface, params, reducedMotion } = ctx;
  const studio = params.add(
    {
      scene: { value: 'product', options: { 'Product still life': 'product', 'Roughness chart': 'chart' } },
      sweepColor: { value: '#8a857f', type: 'color', label: 'Sweep colour' },
      sweepRoughness: { value: 0.42, min: 0.05, max: 1, step: 0.01, label: 'Sweep roughness', hint: 'Lower it to see the soft boxes reflected in the floor.' },
      ambient: { value: 0.18, min: 0, max: 1, step: 0.005, hint: 'A dim hemisphere standing in for light bouncing off the sweep (LTC lights are direct only).' },
      background: { value: '#262427', type: 'color', hint: 'Page colour the sweep fades into at its edges (display space, matched exactly).' },
      turntable: { value: !reducedMotion, hint: 'Slowly orbits the camera, so the reflections slide over the surfaces. Paused in the reference view.' },
    },
    'Studio',
  );
  const lighting = params.add(
    {
      model: {
        value: 'ltc',
        options: { 'Area lights (LTC)': 'ltc', 'Point lights (same power)': 'point', 'Reference (Monte Carlo)': 'reference' },
        hint: 'Point: each soft box collapsed to a point of equal power, the way lights were done before LTC. Reference: unbiased per-pixel sampling of the same integral, accumulated while nothing moves.',
      },
      horizon: {
        value: 2,
        options: { 'Sphere approximation': 2, 'Clip polygon (exact)': 1, 'None (wrong)': 0 },
        label: 'Horizon',
        hint: 'How the part of a light below a surface\'s horizon is removed. Clipping is exact but branchy (up to 5 edges per rectangle); the sphere approximation swaps the shape for a cap with the same vector form factor and reads its clipped value from a table. Ellipses always use the sphere.',
      },
      discMethod: {
        value: 0,
        options: { 'Ellipse (cubic, 2017)': 0, 'Polygon, 8 sides': 8, 'Polygon, 16 sides': 16, 'Polygon, 32 sides': 32 },
        label: 'Disc as',
        hint: 'Heitz & Hill 2017 integrate the ellipse via a cubic solve; an n-gon (area-matched) costs n edges per integral instead.',
      },
      twoSided: { value: false, label: 'Two-sided', hint: 'Lights emit from both faces (a bare diffuser panel) instead of only the front of the soft box.' },
      constantPower: { value: true, label: 'Constant power', hint: 'Resizing a light keeps its total power, like a real strobe in a bigger soft box.' },
      shadows: {
        value: true,
        label: 'Proxy shadows',
        hint: 'LTC lights are unshadowed. This dims each light by how much of its cone a few capsule proxies cover, and adds proxy contact occlusion to the ambient.',
      },
      refSamples: {
        value: 4,
        min: 1,
        max: 32,
        step: 1,
        label: 'Reference samples',
        hint: 'Light + BRDF sample pairs per light per pixel per frame in the reference view.',
      },
    },
    'Lighting',
  );
  const lightParams: LightParams[] = LIGHTS.map((d) => addLight(params, d));
  const look = params.add(
    {
      overrideRoughness: { value: false, label: 'Override roughness', hint: 'Applies one roughness to every object (not the sweep).' },
      roughness: { value: 0.25, min: 0, max: 1, step: 0.01 },
      view: {
        value: VIEW.lit,
        options: { Lit: VIEW.lit, 'Specular only': VIEW.specular, 'Diffuse only': VIEW.diffuse, 'Light footprint': VIEW.footprint, 'Proxy shadows': VIEW.shadows },
        hint: 'Footprint: each light\'s diffuse form factor (key amber, strip cyan, top magenta) with iso-lines every factor of 2. Proxy shadows: each light\'s shadow in its hue, times the proxy contact occlusion.',
      },
      lut: {
        value: 0,
        options: { Off: 0, 'Inverse matrix (a b / c d)': 1, 'Magnitude, Fresnel / albedo, sphere': 2 },
        label: 'LUT overlay',
        hint: 'The 64×64 LTC tables, x = roughness, y = √(1 − cos θ) (the sphere tile: x = cos of its elevation, y = form factor). Warm = positive, blue = negative.',
      },
    },
    'Materials & debug',
  );
  const post = await PostProcess.create(ctx, { bloom: 0.035, vignette: 0.22, grain: 0.015 });
  const ltc = createLtcTables(device);

  // --- Scene ---------------------------------------------------------------------
  const meshes = createMeshes(device);
  const withProps = (name: string, scene: StudioScene) => ({ scene, props: new PropSet(device, scene.props, `area-lights:${name}`) });
  const scenes: Record<string, ReturnType<typeof withProps>> = {
    product: withProps('product', productScene(meshes)),
    chart: withProps('chart', sphereChart(meshes)),
  };

  const frame = new UniformBlock(device, 'Frame', {
    viewProj: 'mat4x4f',
    cameraPos: 'vec3f',
    lightModel: 'u32',
    skyColor: 'vec4f',
    groundColor: 'vec4f',
    fadeX: 'vec2f',
    fadeZ: 'vec2f',
    horizon: 'u32',
    discSides: 'u32',
    twoSided: 'u32',
    view: 'u32',
    shadows: 'u32',
    proxyCount: 'u32',
    refSamples: 'u32',
    frameIndex: 'u32',
    roughOverride: 'f32',
    accumWeight: 'f32',
    emitterScale: 'f32',
    lutView: 'u32',
  });
  const lightData = new Float32Array(LIGHT_COUNT * 16);
  const lightBuffer = device.createBuffer({
    label: 'area-lights:lights',
    size: lightData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const proxyData = new Float32Array(MAX_PROXIES * 8);
  const proxyBuffer = device.createBuffer({
    label: 'area-lights:proxies',
    size: proxyData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // --- Pipelines -------------------------------------------------------------------
  const module = await createShader(
    device,
    'area-lights-ltc',
    wgsl(commonWgsl, fullscreenWgsl, MESH_VERTEX_WGSL, PROP_WGSL, frame.wgslStruct, brdfWgsl, ltcWgsl, shaderWgsl),
  );
  const VF = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
  const bindGroupLayout = device.createBindGroupLayout({
    label: 'area-lights-ltc',
    entries: [
      { binding: 0, visibility: VF, buffer: { type: 'uniform' } },
      { binding: 1, visibility: VF, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: VF, buffer: { type: 'uniform' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 6, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
  const [mainPipeline, emitterPipeline, lutPipeline, accumulatePipeline] = await Promise.all([
    device.createRenderPipelineAsync({
      label: 'area-lights:main',
      layout,
      vertex: { module, entryPoint: 'vs_main', buffers: [MESH_VERTEX_LAYOUT] },
      fragment: { module, entryPoint: 'fs_main', targets: [{ format: HDR_FORMAT }] },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: 'less' },
      multisample: { count: MSAA },
    }),
    device.createRenderPipelineAsync({
      label: 'area-lights:emitters',
      layout,
      vertex: { module, entryPoint: 'vs_emitter', buffers: [MESH_VERTEX_LAYOUT] },
      fragment: { module, entryPoint: 'fs_emitter', targets: [{ format: HDR_FORMAT }] },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: 'less' },
      multisample: { count: MSAA },
    }),
    // The LUT overlay goes onto the display target after post, untouched by tonemapping.
    device.createRenderPipelineAsync({
      label: 'area-lights:lut',
      layout,
      vertex: { module, entryPoint: 'vs_fullscreen' },
      fragment: { module, entryPoint: 'fs_lut', targets: [{ format: ctx.format }] },
    }),
    device.createRenderPipelineAsync({
      label: 'area-lights:accumulate',
      layout: 'auto',
      vertex: { module, entryPoint: 'vs_fullscreen' },
      fragment: { module, entryPoint: 'fs_accumulate', targets: [{ format: HDR_FORMAT }] },
    }),
  ]);
  const bindGroups = new Map(
    Object.entries(scenes).map(([name, s]) => [
      name,
      device.createBindGroup({
        label: `area-lights:${name}`,
        layout: bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: frame.buffer } },
          { binding: 1, resource: { buffer: s.props.buffer } },
          { binding: 2, resource: { buffer: lightBuffer } },
          { binding: 3, resource: { buffer: proxyBuffer } },
          { binding: 4, resource: ltc.matrix.createView() },
          { binding: 5, resource: ltc.magnitude.createView() },
          { binding: 6, resource: ltc.sampler },
        ],
      }),
    ]),
  );

  const targets = new TargetPool(device);
  // Accumulation bind groups, per (current frame, history) texture pair; both only change on resize.
  const accumulateGroups = new WeakMap<GPUTexture, { current: GPUTexture; group: GPUBindGroup }>();
  const accumulateBindGroup = (current: GPUTexture, history: GPUTexture) => {
    const hit = accumulateGroups.get(history);
    if (hit?.current === current) return hit.group;
    const group = device.createBindGroup({
      label: 'area-lights:accumulate',
      layout: accumulatePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: frame.buffer } },
        { binding: 8, resource: current.createView() },
        { binding: 9, resource: history.createView() },
      ],
    });
    accumulateGroups.set(history, { current, group });
    return group;
  };
  const camera = new OrbitCamera(surface.canvas, { target: [0, 0.7, 0], distance: 6.2, yaw: 0.32, pitch: 0.2, minDistance: 2, maxDistance: 14 });

  // Reference accumulation restarts whenever anything that changes the image does.
  let accumulated = 0;
  let historyIndex = 0;
  let paramsChanged = true;
  const unsubscribe = params.onChange(() => (paramsChanged = true));
  const lastViewProj = new Float32Array(16);
  let lastSize = '';
  let frameIndex = 0;
  let currentScene = '';

  /** World-space centre, half-axes and area of a light aimed at `aim`. */
  const placeLight = (l: LightParams, aim: Vec3) => {
    const az = (l.azimuth * Math.PI) / 180;
    const el = (l.elevation * Math.PI) / 180;
    const dir = vec3.fromValues(Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az));
    const center = vec3.addScaled(aim, dir, l.distance);
    // Emit toward the subject: ex × ey = forward.
    const forward = vec3.negate(dir);
    const up = Math.abs(forward[1]) > 0.99 ? vec3.fromValues(0, 0, 1) : vec3.fromValues(0, 1, 0);
    const right = vec3.normalize(vec3.cross(up, forward));
    const upward = vec3.cross(forward, right);
    const roll = (l.roll * Math.PI) / 180;
    const ex = vec3.scale(vec3.add(vec3.scale(right, Math.cos(roll)), vec3.scale(upward, Math.sin(roll))), l.width / 2);
    const ey = vec3.scale(vec3.subtract(vec3.scale(upward, Math.cos(roll)), vec3.scale(right, Math.sin(roll))), l.height / 2);
    return { center, ex, ey, area: area(l.shape, l.width, l.height) };
  };

  return {
    frame({ time, dt, size }) {
      const { scene, props } = scenes[studio.scene] ?? scenes.product;
      if (studio.scene !== currentScene) {
        currentScene = studio.scene;
        camera.setView({ target: scene.aim, distance: scene.cameraDistance });
        proxyData.fill(0);
        scene.proxies.slice(0, MAX_PROXIES).forEach((p, i) => proxyData.set([...p.a, p.radius, ...p.b, p.owner], i * 8));
        device.queue.writeBuffer(proxyBuffer, 0, proxyData);
      }
      const model = MODEL[lighting.model as keyof typeof MODEL] ?? MODEL.ltc;
      const reference = model === MODEL.reference;
      camera.autoRotate = studio.turntable && !reference ? 0.1 : 0;
      camera.update(dt, size.width / size.height);

      scene.props[0].color = studio.sweepColor;
      scene.props[0].roughness = studio.sweepRoughness;
      props.update(0);

      const aim = vec3.fromValues(...scene.aim);
      lightParams.forEach((l, i) => {
        const o = i * 16;
        const { center, ex, ey, area: a } = placeLight(l, aim);
        const d = l.defaults;
        // Constant power: radiance scales with 1 / area, normalised to the default size and shape.
        const scale = lighting.constantPower ? area(d.shape, d.width, d.height) / a : 1;
        const c = hexToLinear(l.color);
        // Housing depth (none for a two-sided panel).
        const depth = lighting.twoSided ? 0 : 0.18 + 0.25 * Math.min(l.width, l.height);
        lightData.set([...center, SHAPE[l.shape] ?? SHAPE.off, ...ex, a, ...ey, depth], o);
        lightData.set([c[0] * l.intensity * scale, c[1] * l.intensity * scale, c[2] * l.intensity * scale, 0], o + 12);
      });
      device.queue.writeBuffer(lightBuffer, 0, lightData);

      // Accumulate the reference only while the image is static.
      const sizeKey = `${size.width}x${size.height}`;
      const moved = camera.viewProjection.some((x, i) => Math.abs(x - lastViewProj[i]) > 1e-6);
      lastViewProj.set(camera.viewProjection);
      if (!reference || moved || paramsChanged || sizeKey !== lastSize) accumulated = 0;
      paramsChanged = false;
      lastSize = sizeKey;

      const sweep = hexToLinear(studio.sweepColor);
      const amb = studio.ambient;
      frame
        .set('viewProj', camera.viewProjection)
        .set('cameraPos', camera.position)
        .set('lightModel', model)
        .set('skyColor', [0.32 * amb, 0.33 * amb, 0.35 * amb, 0])
        .set('groundColor', [sweep[0] * 2 * amb, sweep[1] * 2 * amb, sweep[2] * 2 * amb, 0])
        .set('fadeX', [6, 13])
        .set('fadeZ', [4, 9.5])
        .set('horizon', lighting.horizon)
        .set('discSides', lighting.discMethod)
        .set('twoSided', lighting.twoSided ? 1 : 0)
        .set('view', look.view)
        .set('shadows', lighting.shadows ? 1 : 0)
        .set('proxyCount', Math.min(scene.proxies.length, MAX_PROXIES))
        .set('refSamples', lighting.refSamples)
        .set('frameIndex', frameIndex++)
        .set('roughOverride', look.overrideRoughness ? look.roughness : -1)
        .set('accumWeight', 1 / (accumulated + 1))
        .set('emitterScale', model === MODEL.point ? 0.04 / Math.max(...lightParams.map((l) => Math.max(l.width, l.height))) : 1)
        .set('lutView', look.lut)
        .upload();

      const full = [size.width, size.height] as const;
      const color = targets.get('msaa', { size: full, format: HDR_FORMAT, sampleCount: MSAA });
      const depth = targets.get('depth', { size: full, format: DEPTH_FORMAT, sampleCount: MSAA });
      const sceneTarget = post.scene(size);
      const encoder = device.createCommandEncoder({ label: 'area-lights-ltc' });
      const pass = encoder.beginRenderPass({
        label: 'area-lights:main',
        colorAttachments: [{ view: color.view, resolveTarget: sceneTarget.view, loadOp: 'clear', storeOp: 'discard', clearValue: [0, 0, 0, 0] }],
        depthStencilAttachment: { view: depth.view, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard' },
      });
      pass.setBindGroup(0, bindGroups.get(currentScene) ?? bindGroups.get('product')!);
      pass.setPipeline(mainPipeline);
      props.draw(pass);
      pass.setPipeline(emitterPipeline);
      lightParams.forEach((l, i) => {
        const housing = meshes.housings[l.shape === 'disc' ? 'disc' : 'rect'];
        pass.setVertexBuffer(0, housing.vertexBuffer);
        pass.setIndexBuffer(housing.indexBuffer, 'uint32');
        pass.drawIndexed(housing.indexCount, 1, 0, 0, i);
      });
      pass.end();

      let source: GPUTexture | undefined;
      if (reference) {
        const read = targets.get(`history${historyIndex}`, { size: full, format: HDR_FORMAT });
        const write = targets.get(`history${1 - historyIndex}`, { size: full, format: HDR_FORMAT });
        const ap = encoder.beginRenderPass({
          label: 'area-lights:accumulate',
          colorAttachments: [{ view: write.view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }],
        });
        ap.setPipeline(accumulatePipeline);
        ap.setBindGroup(0, accumulateBindGroup(sceneTarget.texture, read.texture));
        ap.draw(3);
        ap.end();
        source = write.texture;
        historyIndex = 1 - historyIndex;
        accumulated = Math.min(accumulated + 1, MAX_ACCUMULATED);
      } else {
        targets.release('history0');
        targets.release('history1');
      }

      const raw = look.view === VIEW.footprint || look.view === VIEW.shadows;
      const output = surface.currentTexture().createView();
      post.render(encoder, output, { time, raw, background: hexToLinear(studio.background), source });
      if (look.lut !== 0) {
        const lp = encoder.beginRenderPass({ label: 'area-lights:lut', colorAttachments: [{ view: output, loadOp: 'load', storeOp: 'store' }] });
        const s = Math.round(Math.min(size.width, size.height) * 0.42);
        const margin = Math.round(16 * size.pixelRatio);
        lp.setViewport(margin, size.height - s - margin, s, s, 0, 1);
        lp.setPipeline(lutPipeline);
        lp.setBindGroup(0, bindGroups.get(currentScene) ?? bindGroups.get('product')!);
        lp.draw(3);
        lp.end();
      }
      device.queue.submit([encoder.finish()]);
    },
    dispose() {
      unsubscribe();
      camera.destroy();
      targets.destroy();
      post.destroy();
      ltc.destroy();
      frame.destroy();
      lightBuffer.destroy();
      proxyBuffer.destroy();
      Object.values(scenes).forEach((s) => s.props.destroy());
      meshes.destroy();
    },
  };
};

export default create;
