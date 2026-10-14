import { mat4, vec3, vec4, type Mat4, type Vec3 } from 'wgpu-matrix';
import brdfWgsl from '../../shaders/brdf.wgsl?raw';
import commonWgsl from '../../shaders/common.wgsl?raw';
import fullscreenWgsl from '../../shaders/fullscreen.wgsl?raw';
import noiseWgsl from '../../shaders/noise.wgsl?raw';
import sdfWgsl from '../../shaders/sdf.wgsl?raw';
import { hexToLinear } from '../../gpu/color';
import { MESH_VERTEX_LAYOUT, MESH_VERTEX_WGSL, plane, sphere, torus, uploadMesh } from '../../gpu/geometry';
import { generateMipmaps, mipLevelCount } from '../../gpu/mipmaps';
import { createShader, wgsl } from '../../gpu/shader';
import { TargetPool } from '../../gpu/targets';
import { UniformBlock } from '../../gpu/uniforms';
import { HDR_FORMAT, PostProcess } from '../../post/post';
import { TemporalAA } from '../../post/taa';
import { OrbitCamera } from '../../scene/orbit-camera';
import { PROP_WGSL, PropSet, type Prop } from '../../scene/props';
import { lathe, subdividedBox } from '../../scene/shapes';
import type { ExperimentFactory } from '../types';
import cookieWgsl from './cookie.wgsl?raw';
import shaderWgsl from './shader.wgsl?raw';

const MSAA_SAMPLES = 4;
const DEPTH_FORMAT: GPUTextureFormat = 'depth24plus';
const SHADOW_FORMAT: GPUTextureFormat = 'depth32float';
const COOKIE_FORMAT: GPUTextureFormat = 'rgba8unorm';
const COOKIE_SIZE = 512;
const SHADOW_NEAR = 0.5;
const SHADOW_FAR = 20;
/** The back wall's plane (z) and the default spot on it. */
const WALL_Z = -2;
const DEFAULT_AIM = [0.4, 1.6, WALL_Z] as const;
const MAT = { plain: 0, wall: 1, floor: 2 } as const;
const CIRCLE_SEGMENTS = 48;

const create: ExperimentFactory = async (ctx) => {
  const { device, surface, params, pointer, reducedMotion } = ctx;
  const coarse = window.matchMedia('(pointer: coarse)').matches;

  const spot = params.add(
    {
      intensity: { value: 3, min: 0, max: 12, step: 0.05, hint: 'Illuminance at the aim point (falloff is inverse-square).' },
      color: { value: '#ffd9a8', type: 'color' },
      cone: { value: 54, min: 10, max: 110, step: 1, label: 'Cone angle (°)', hint: 'Full angle of the outer cone; the field of view of the shadow map follows it.' },
      softness: { value: 0.35, min: 0, max: 1, step: 0.01, label: 'Cone softness', hint: 'Width of the falloff from the inner to the outer cone.' },
      azimuth: { value: 50, min: -70, max: 70, step: 1 },
      elevation: { value: 26, min: 5, max: 75, step: 1 },
      distance: { value: 7, min: 3, max: 12, step: 0.1, hint: 'Light distance from the aim point. Closer = wider fan of shadows, worse texel density at the far end.' },
      followPointer: { value: true, label: 'Aim at pointer', hint: 'Hover the canvas to aim the spot at the wall.' },
    },
    'Spot light',
  );
  const gobo = params.add(
    {
      cookie: { value: 'leaves', options: { 'Leaves (dapple)': 'leaves', 'Venetian blinds': 'blinds', 'Window & curtain': 'window', None: 'none' } },
      cookieBlur: {
        value: 1.4,
        min: 0,
        max: 6,
        step: 0.05,
        label: 'Cookie blur',
        hint: 'Focus of the projector: which mip of the cookie is sampled (0 = sharp; each level doubles the blur).',
      },
      cookieZoom: { value: 1, min: 0.5, max: 2.5, step: 0.01, label: 'Cookie zoom', hint: 'Projected size of the gobo relative to the cone.' },
      sway: { value: reducedMotion ? 0 : 1, min: 0, max: 2, step: 0.01, hint: 'Wind in the leaves, breeze in the blinds and curtain.' },
    },
    'Gobo',
  );
  const shadow = params.add(
    {
      filterMode: { value: 3, options: { Off: 0, 'Hard (2×2 HW)': 1, 'PCF (Poisson)': 2, PCSS: 3 }, label: 'Filter' },
      mapSize: { value: coarse ? 1024 : 2048, options: { '512': 512, '1024': 1024, '2048': 2048, '4096': 4096 }, label: 'Map size' },
      lightSize: { value: 0.06, min: 0.005, max: 0.4, step: 0.005, label: 'Bulb radius', hint: 'Physical size of the light (m), used by PCSS.' },
      pcfRadius: { value: 2, min: 0, max: 8, step: 0.1, label: 'PCF radius', hint: 'Poisson kernel radius in texels.' },
      depthBias: { value: 1.2, min: 0, max: 8, step: 0.1, label: 'Depth bias', hint: 'In texels at the receiver, applied to linear depth.' },
      normalOffset: { value: 1, min: 0, max: 6, step: 0.1, label: 'Normal offset', hint: 'In texels at the receiver.' },
    },
    'Shadows',
  );
  const look = params.add(
    {
      wall: { value: '#8f8478', type: 'color' },
      floor: { value: '#4d4741', type: 'color' },
      ambient: { value: 0.05, min: 0, max: 0.5, step: 0.005 },
      antialiasing: { value: 'msaa', options: { 'MSAA 4×': 'msaa', TAA: 'taa', Off: 'off' } },
      view: {
        value: 0,
        options: { Lit: 0, Visibility: 1, 'Light pattern (cone × cookie)': 2, 'Shadow texel density': 3, 'Light depth': 4 },
        hint: 'Texel density shows 32×32-texel blocks of the shadow map: where they look big, shadows will look blocky (perspective aliasing).',
      },
      overlay: { value: 'none', options: { None: 'none', 'Shadow map': 'map', Cookie: 'cookie' } },
      showFrustum: { value: false, label: 'Show light frustum' },
      animate: { value: !reducedMotion },
    },
    'Look & debug',
  );
  const post = await PostProcess.create(ctx, { exposure: 1.05, bloom: 0.03, vignette: 0.4, grain: 0.03 });
  const taa = await TemporalAA.create(device);

  // --- Scene ------------------------------------------------------------------
  const meshes = {
    plane: uploadMesh(device, plane(1, 8), 'plane'),
    box: uploadMesh(device, subdividedBox(1, 1, 1, 1), 'box'),
    sphere: uploadMesh(device, sphere(1, 48, 32), 'sphere'),
    ring: uploadMesh(device, torus(0.3, 0.035, 24, 96), 'ring'),
    vase: uploadMesh(
      device,
      lathe([[0, 0], [0.12, 0], [0.17, 0.12], [0.18, 0.32], [0.12, 0.6], [0.07, 0.78], [0.075, 0.86], [0.09, 0.9], [0, 0.9]], { creaseAngle: 1.2 }),
      'vase',
    ),
    bowl: uploadMesh(device, lathe([[0, 0], [0.12, 0], [0.24, 0.06], [0.3, 0.14], [0, 0.14]], { creaseAngle: 0.9 }), 'bowl'),
  };
  const T = (x: number, y: number, z: number) => mat4.translation([x, y, z]);
  const box = (x: number, y: number, z: number, sx: number, sy: number, sz: number, rotY = 0): Mat4 =>
    mat4.scale(mat4.rotateY(T(x, y, z), rotY), [sx, sy, sz]);
  const props: Prop[] = [
    { mesh: meshes.plane, color: '#ffffff', roughness: 0.9, extra: MAT.wall, transform: () => mat4.scale(mat4.rotateX(T(0, 4, WALL_Z), Math.PI / 2), [24, 1, 8]) },
    { mesh: meshes.plane, color: '#ffffff', roughness: 0.4, extra: MAT.floor, transform: () => mat4.scale(T(0, 0, 3), [24, 1, 10]) },
    // Plinth with a brass sphere and ring.
    { mesh: meshes.box, color: '#b9b0a4', roughness: 0.85, transform: () => box(1.0, 0.5, -1.1, 0.7, 1.0, 0.7) },
    { mesh: meshes.sphere, color: '#d9a35a', roughness: 0.28, metallic: 1, transform: () => mat4.uniformScale(T(1.0, 1.27, -1.1), 0.27) },
    { mesh: meshes.ring, color: '#d9a35a', roughness: 0.3, metallic: 1, transform: () => mat4.rotateZ(mat4.rotateX(T(1.0, 1.27, -1.1), 1.2), 0.5) },
    // Tall vase with branches: long, thin shadows across the wall.
    { mesh: meshes.vase, color: '#e7e0d4', roughness: 0.35, transform: () => T(2.1, 0, -1.25) },
    // A low bench with a bowl.
    { mesh: meshes.box, color: '#5b4331', roughness: 0.6, transform: () => box(-1.5, 0.42, -1.45, 1.8, 0.06, 0.42) },
    { mesh: meshes.box, color: '#5b4331', roughness: 0.6, transform: () => box(-2.25, 0.2, -1.45, 0.06, 0.4, 0.38) },
    { mesh: meshes.box, color: '#5b4331', roughness: 0.6, transform: () => box(-0.75, 0.2, -1.45, 0.06, 0.4, 0.38) },
    { mesh: meshes.bowl, color: '#2f3b40', roughness: 0.4, transform: () => T(-1.1, 0.45, -1.45) },
    { mesh: meshes.box, color: '#a8402f', roughness: 0.7, transform: () => box(-1.9, 0.48, -1.45, 0.32, 0.06, 0.24, 0.15) },
    { mesh: meshes.box, color: '#c8b48a', roughness: 0.7, transform: () => box(-1.88, 0.53, -1.45, 0.3, 0.05, 0.22, -0.1) },
  ];
  // Branches: a few stems with twigs, angled out of the vase.
  const stems: [number, number, number, number][] = [
    [0.22, 0.2, 1.3, 0.012],
    [-0.28, -0.5, 1.15, 0.01],
    [0.05, 1.9, 1.45, 0.011],
  ];
  for (const [lean, yaw, len, r] of stems) {
    const base = mat4.rotateZ(mat4.rotateY(T(2.1, 0.86, -1.25), yaw), lean);
    props.push({ mesh: meshes.box, color: '#3b2f25', roughness: 0.8, transform: () => mat4.scale(mat4.translate(base, [0, len / 2, 0]), [r, len, r]) });
    for (let k = 0; k < 3; k++) {
      const at = len * (0.45 + 0.18 * k);
      const side = k % 2 ? 1 : -1;
      const twig = mat4.rotateZ(mat4.translate(base, [0, at, 0]), side * 0.7);
      props.push({ mesh: meshes.box, color: '#3b2f25', roughness: 0.8, transform: () => mat4.scale(mat4.translate(twig, [0, 0.14, 0]), [r * 0.6, 0.28, r * 0.6]) });
    }
  }
  const propSet = new PropSet(device, props, 'spot-gobo');
  propSet.update(0);

  // --- Uniforms & pipelines -----------------------------------------------------
  const frame = new UniformBlock(device, 'Frame', {
    viewProj: 'mat4x4f',
    lightViewProj: 'mat4x4f',
    cameraPos: 'vec3f',
    tanHalfFov: 'f32',
    lightPos: 'vec3f',
    cosOuter: 'f32',
    lightDir: 'vec3f',
    cosInner: 'f32',
    lightColor: 'vec4f',
    skyColor: 'vec4f',
    groundColor: 'vec4f',
    wallColor: 'vec4f',
    floorColor: 'vec4f',
    shadowNear: 'f32',
    shadowFar: 'f32',
    mapSize: 'f32',
    filterMode: 'u32',
    lightSize: 'f32',
    pcfRadius: 'f32',
    depthBias: 'f32',
    normalOffset: 'f32',
    cookieBlur: 'f32',
    cookieScale: 'f32',
    cookieOn: 'u32',
    debugView: 'u32',
    noiseOffset: 'f32',
  });
  const cookieUniforms = new UniformBlock(device, 'Cookie', { time: 'f32', kind: 'u32', sway: 'f32' });

  const module = await createShader(
    device,
    'spot-light-gobo',
    wgsl(commonWgsl, fullscreenWgsl, noiseWgsl, brdfWgsl, MESH_VERTEX_WGSL, PROP_WGSL, frame.wgslStruct, shaderWgsl),
  );
  const cookieModule = await createShader(
    device,
    'spot-light-gobo:cookie',
    wgsl(commonWgsl, fullscreenWgsl, noiseWgsl, sdfWgsl, cookieUniforms.wgslStruct, cookieWgsl),
  );

  const bindGroupLayout = device.createBindGroupLayout({
    label: 'spot-gobo',
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
  const shadowBindGroupLayout = device.createBindGroupLayout({
    label: 'spot-shadow',
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
    ],
  });
  const lineLayout: GPUVertexBufferLayout = { arrayStride: 16, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }] };

  const makeScenePipelines = (count: number) =>
    Promise.all([
      device.createRenderPipelineAsync({
        label: `spot-main-x${count}`,
        layout,
        vertex: { module, entryPoint: 'vs_main', buffers: [MESH_VERTEX_LAYOUT] },
        fragment: { module, entryPoint: 'fs_main', targets: [{ format: HDR_FORMAT }] },
        primitive: { topology: 'triangle-list', cullMode: 'back' },
        depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: 'less' },
        multisample: { count },
      }),
      device.createRenderPipelineAsync({
        label: `spot-frustum-x${count}`,
        layout,
        vertex: { module, entryPoint: 'vs_line', buffers: [lineLayout] },
        fragment: { module, entryPoint: 'fs_line', targets: [{ format: HDR_FORMAT }] },
        primitive: { topology: 'line-list' },
        depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: 'less-equal' },
        multisample: { count },
      }),
      ...(['fs_overlay_map', 'fs_overlay_cookie'] as const).map((entryPoint) =>
        device.createRenderPipelineAsync({
          label: `spot-${entryPoint}-x${count}`,
          layout,
          vertex: { module, entryPoint: 'vs_fullscreen' },
          fragment: { module, entryPoint, targets: [{ format: HDR_FORMAT }] },
          depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: 'always' },
          multisample: { count },
        }),
      ),
    ]);
  const [shadowPipeline, cookiePipeline, msaaPipelines, singlePipelines] = await Promise.all([
    device.createRenderPipelineAsync({
      label: 'spot-shadow',
      layout: device.createPipelineLayout({ bindGroupLayouts: [shadowBindGroupLayout] }),
      vertex: { module, entryPoint: 'vs_shadow', buffers: [MESH_VERTEX_LAYOUT] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      // Slope-scaled bias in hardware, on top of the shader's linear bias.
      depthStencil: { format: SHADOW_FORMAT, depthWriteEnabled: true, depthCompare: 'less', depthBiasSlopeScale: 1.5 },
    }),
    device.createRenderPipelineAsync({
      label: 'spot-cookie',
      layout: 'auto',
      vertex: { module: cookieModule, entryPoint: 'vs_fullscreen' },
      fragment: { module: cookieModule, entryPoint: 'fs_cookie', targets: [{ format: COOKIE_FORMAT }] },
    }),
    makeScenePipelines(MSAA_SAMPLES),
    makeScenePipelines(1),
  ]);

  const compareSampler = device.createSampler({ label: 'spot-shadow-compare', compare: 'less-equal', magFilter: 'linear', minFilter: 'linear' });
  // Clamp-to-edge: the cookie's border texels are black, so the pattern stops at its frame.
  const cookieSampler = device.createSampler({ label: 'spot-cookie', magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear' });
  const cookieTexture = device.createTexture({
    label: 'spot-cookie',
    size: [COOKIE_SIZE, COOKIE_SIZE],
    format: COOKIE_FORMAT,
    mipLevelCount: mipLevelCount(COOKIE_SIZE, COOKIE_SIZE),
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  const cookieBindGroup = device.createBindGroup({
    layout: cookiePipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: cookieUniforms.buffer } }],
  });
  const shadowBindGroup = device.createBindGroup({
    layout: shadowBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: frame.buffer } },
      { binding: 1, resource: { buffer: propSet.buffer } },
    ],
  });
  const lineVertices = new Float32Array((4 + 4 + CIRCLE_SEGMENTS + 1) * 2 * 4);
  const lineBuffer = device.createBuffer({ label: 'spot-frustum', size: lineVertices.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });

  let shadowMap: GPUTexture | null = null;
  let mainBindGroup: GPUBindGroup | null = null;
  const ensureShadowMap = (size: number) => {
    if (shadowMap?.width === size) return;
    shadowMap?.destroy();
    shadowMap = device.createTexture({
      label: 'spot-shadow-map',
      size: [size, size],
      format: SHADOW_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    mainBindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: frame.buffer } },
        { binding: 1, resource: { buffer: propSet.buffer } },
        { binding: 2, resource: shadowMap.createView() },
        { binding: 3, resource: compareSampler },
        { binding: 4, resource: cookieTexture.createView() },
        { binding: 5, resource: cookieSampler },
      ],
    });
  };

  const targets = new TargetPool(device);
  const camera = new OrbitCamera(surface.canvas, {
    target: [0.2, 1.45, -1.2],
    distance: 7.6,
    yaw: 0.16,
    pitch: 0.12,
    minPitch: -0.05,
    maxPitch: 0.9,
    minDistance: 3,
    maxDistance: 14,
  });
  let animTime = 0;
  let cookieKey = '';
  const aim = vec3.fromValues(...DEFAULT_AIM);

  /** The point on the back wall under the pointer, or null if the ray misses it. */
  const pointerOnWall = (): Vec3 | null => {
    const inv = mat4.inverse(camera.viewProjection);
    const [u, v] = pointer.smoothUv;
    const ndc = [u * 2 - 1, 1 - v * 2];
    const near = vec4.transformMat4([ndc[0], ndc[1], 0, 1], inv);
    const far = vec4.transformMat4([ndc[0], ndc[1], 1, 1], inv);
    const a = vec3.scale([near[0], near[1], near[2]], 1 / near[3]);
    const b = vec3.scale([far[0], far[1], far[2]], 1 / far[3]);
    const dz = b[2] - a[2];
    if (Math.abs(dz) < 1e-6) return null;
    const t = (WALL_Z - a[2]) / dz;
    if (t < 0) return null;
    const p = vec3.lerp(a, b, t);
    return vec3.fromValues(Math.min(5, Math.max(-5, p[0])), Math.min(4.5, Math.max(0.2, p[1])), WALL_Z);
  };

  return {
    frame({ time, dt, size }) {
      if (look.animate) animTime += dt;
      camera.update(dt, size.width / size.height);
      ensureShadowMap(shadow.mapSize);

      // Aim: the pointer on the wall, else a slow drift around the default spot.
      const hover = spot.followPointer && pointer.inside && pointer.idle < 4 ? pointerOnWall() : null;
      const goal =
        hover ??
        vec3.fromValues(DEFAULT_AIM[0] + 0.5 * Math.sin(animTime * 0.13), DEFAULT_AIM[1] + 0.25 * Math.sin(animTime * 0.19), WALL_Z);
      const k = 1 - Math.pow(2, -dt / (hover ? 0.12 : 0.6));
      vec3.lerp(aim, goal, k, aim);

      const az = (spot.azimuth * Math.PI) / 180;
      const el = (spot.elevation * Math.PI) / 180;
      const toLight = vec3.fromValues(Math.sin(az) * Math.cos(el), Math.sin(el), Math.cos(az) * Math.cos(el));
      const lightPos = vec3.addScaled(aim, toLight, spot.distance);
      const lightDir = vec3.negate(toLight);
      const outer = ((spot.cone / 2) * Math.PI) / 180;
      const inner = outer * (1 - spot.softness);
      // The map's frustum covers the outer cone with a little margin.
      const halfFov = Math.min(outer * 1.04 + 0.01, 1.45);
      const up = Math.abs(toLight[1]) > 0.99 ? [0, 0, 1] : [0, 1, 0];
      const lightView = mat4.lookAt(lightPos, aim, up);
      const lightProj = mat4.perspective(2 * halfFov, 1, SHADOW_NEAR, SHADOW_FAR);
      const lightViewProj = mat4.multiply(lightProj, lightView);

      const aa = look.antialiasing;
      const samples = aa === 'msaa' ? MSAA_SAMPLES : 1;
      const [mainPipeline, linePipeline, mapOverlay, cookieOverlay] = samples > 1 ? msaaPipelines : singlePipelines;
      const fullSize = [size.width, size.height] as const;
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

      const lc = hexToLinear(spot.color);
      // Intensity is the illuminance at the aim point, so moving the light keeps the exposure.
      const power = spot.intensity * spot.distance * spot.distance;
      const amb = look.ambient;
      frame
        .set('viewProj', viewProj)
        .set('lightViewProj', lightViewProj)
        .set('cameraPos', camera.position)
        .set('tanHalfFov', Math.tan(halfFov))
        .set('lightPos', lightPos)
        .set('cosOuter', Math.cos(outer))
        .set('lightDir', lightDir)
        .set('cosInner', Math.cos(inner))
        .set('lightColor', [lc[0] * power, lc[1] * power, lc[2] * power, 0])
        .set('skyColor', [0.55 * amb, 0.6 * amb, 0.75 * amb, 0])
        .set('groundColor', [0.3 * amb, 0.25 * amb, 0.2 * amb, 0])
        .set('wallColor', [...hexToLinear(look.wall), 0])
        .set('floorColor', [...hexToLinear(look.floor), 0])
        .set('shadowNear', SHADOW_NEAR)
        .set('shadowFar', SHADOW_FAR)
        .set('mapSize', shadow.mapSize)
        .set('filterMode', shadow.filterMode)
        .set('lightSize', shadow.lightSize)
        .set('pcfRadius', shadow.pcfRadius)
        .set('depthBias', shadow.depthBias)
        .set('normalOffset', shadow.normalOffset)
        .set('cookieBlur', gobo.cookieBlur)
        // Map the cookie square onto the outer cone (the map's frustum is a little wider).
        .set('cookieScale', Math.tan(halfFov) / Math.tan(outer) / gobo.cookieZoom)
        .set('cookieOn', gobo.cookie === 'none' ? 0 : 1)
        .set('debugView', look.view)
        .set('noiseOffset', aa === 'taa' ? taa.frameNoiseOffset : 0)
        .upload();

      const encoder = device.createCommandEncoder({ label: 'spot-light-gobo' });

      // 1. Cookie: regenerated (with its mips) only when it changes.
      const kind = { leaves: 0, blinds: 1, window: 2, none: 0 }[gobo.cookie] ?? 0;
      const animated = gobo.sway > 0 && look.animate;
      const key = `${gobo.cookie}:${gobo.sway}:${animated ? animTime : 0}`;
      if (key !== cookieKey && gobo.cookie !== 'none') {
        cookieKey = key;
        cookieUniforms.set('time', animTime).set('kind', kind).set('sway', gobo.sway).upload();
        const cp = encoder.beginRenderPass({
          label: 'spot-cookie',
          colorAttachments: [
            { view: cookieTexture.createView({ baseMipLevel: 0, mipLevelCount: 1 }), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] },
          ],
        });
        cp.setPipeline(cookiePipeline);
        cp.setBindGroup(0, cookieBindGroup);
        cp.draw(3);
        cp.end();
        generateMipmaps(device, cookieTexture, encoder);
      }

      // 2. Shadow map from the spot.
      if (shadow.filterMode !== 0 || look.overlay === 'map' || look.view === 4) {
        const sp = encoder.beginRenderPass({
          label: 'spot-shadow',
          colorAttachments: [],
          depthStencilAttachment: { view: shadowMap!.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
        });
        sp.setPipeline(shadowPipeline);
        sp.setBindGroup(0, shadowBindGroup);
        propSet.draw(sp, { castersOnly: true });
        sp.end();
      }

      // 3. Main pass (+ frustum lines and overlays).
      const mp = encoder.beginRenderPass({
        label: 'spot-main',
        colorAttachments: [
          msaaColor
            ? { view: msaaColor.view, resolveTarget: scene.view, loadOp: 'clear', storeOp: 'discard', clearValue: [0, 0, 0, 0] }
            : { view: scene.view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] },
        ],
        depthStencilAttachment: { view: depth.view, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: aa === 'taa' ? 'store' : 'discard' },
      });
      mp.setPipeline(mainPipeline);
      mp.setBindGroup(0, mainBindGroup!);
      propSet.draw(mp);

      if (look.showFrustum) {
        writeFrustum(lineVertices, lightPos, mat4.inverse(lightViewProj), aim, lightDir, Math.tan(outer));
        device.queue.writeBuffer(lineBuffer, 0, lineVertices);
        mp.setPipeline(linePipeline);
        mp.setVertexBuffer(0, lineBuffer);
        mp.draw(lineVertices.length / 4);
      }
      if (look.overlay !== 'none') {
        const s = Math.round(Math.min(size.width, size.height) * 0.3);
        const m = Math.round(16 * size.pixelRatio);
        mp.setViewport(m, size.height - s - m, s, s, 0, 1);
        mp.setPipeline(look.overlay === 'map' ? mapOverlay : cookieOverlay);
        mp.draw(3);
      }
      mp.end();

      const source =
        aa === 'taa' ? taa.resolve(encoder, { color: scene.texture, depth: depth.texture, viewProj: camera.viewProjection }) : undefined;
      post.render(encoder, surface.currentTexture().createView(), { time, raw: look.view !== 0, source });
      device.queue.submit([encoder.finish()]);
    },
    dispose() {
      camera.destroy();
      shadowMap?.destroy();
      cookieTexture.destroy();
      lineBuffer.destroy();
      targets.destroy();
      post.destroy();
      taa.destroy();
      propSet.destroy();
      frame.destroy();
      cookieUniforms.destroy();
      Object.values(meshes).forEach((m) => m.destroy());
    },
  };
};

/**
 * Line-list vertices (xyz + brightness) for the spot's frustum: the four edges from the apex to
 * the corners of the map's far rectangle (cut at the aim distance), that rectangle, the outer
 * cone's circle at the aim distance, and the axis.
 */
function writeFrustum(out: Float32Array, apex: Vec3, invViewProj: Mat4, aim: Vec3, dir: Vec3, tanOuter: number): void {
  let o = 0;
  const push = (p: ArrayLike<number>, w: number) => {
    out.set([p[0], p[1], p[2], w], o);
    o += 4;
  };
  const reach = vec3.distance(apex, aim) * 1.15;
  const corners = [
    [-1, -1],
    [1, -1],
    [1, 1],
    [-1, 1],
  ].map(([x, y]) => {
    const p = vec4.transformMat4([x, y, 1, 1], invViewProj);
    const d = vec3.normalize(vec3.sub([p[0] / p[3], p[1] / p[3], p[2] / p[3]], apex));
    // Scale so the point sits `reach` along the axis.
    return vec3.addScaled(apex, d, reach / vec3.dot(d, dir));
  });
  for (let i = 0; i < 4; i++) {
    push(apex, 1.2);
    push(corners[i], 0.5);
  }
  for (let i = 0; i < 4; i++) {
    push(corners[i], 0.5);
    push(corners[(i + 1) % 4], 0.5);
  }
  // Cone circle at the aim distance.
  const dist = vec3.distance(apex, aim);
  const helper = Math.abs(dir[1]) > 0.99 ? [1, 0, 0] : [0, 1, 0];
  const t = vec3.normalize(vec3.cross(dir, helper));
  const b = vec3.cross(t, dir);
  const r = dist * tanOuter;
  const centre = vec3.addScaled(apex, dir, dist);
  const at = (i: number) => {
    const a = (i / CIRCLE_SEGMENTS) * Math.PI * 2;
    return vec3.add(centre, vec3.add(vec3.scale(t, Math.cos(a) * r), vec3.scale(b, Math.sin(a) * r)));
  };
  for (let i = 0; i < CIRCLE_SEGMENTS; i++) {
    push(at(i), 1.5);
    push(at(i + 1), 1.5);
  }
  push(apex, 0.8);
  push(centre, 0.8);
}

export default create;
