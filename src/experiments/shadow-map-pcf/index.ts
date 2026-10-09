import { mat4, vec3, type Mat4 } from 'wgpu-matrix';
import commonWgsl from '../../shaders/common.wgsl?raw';
import fullscreenWgsl from '../../shaders/fullscreen.wgsl?raw';
import shadowWgsl from '../../shaders/shadow.wgsl?raw';
import { MESH_VERTEX_LAYOUT, MESH_VERTEX_WGSL, box, plane, sphere, torus, uploadMesh, type Mesh } from '../../gpu/geometry';
import { createShader, wgsl } from '../../gpu/shader';
import { UniformBlock } from '../../gpu/uniforms';
import { OrbitCamera } from '../../scene/orbit-camera';
import { hexToLinear, linearToSrgb } from '../../ui/params';
import type { ExperimentFactory } from '../types';
import shaderWgsl from './shader.wgsl?raw';

const SAMPLE_COUNT = 4;
const DEPTH_FORMAT: GPUTextureFormat = 'depth24plus';
const SHADOW_FORMAT: GPUTextureFormat = 'depth32float';
/** Half-width of the light's orthographic frustum (world units); covers the props. */
const SHADOW_EXTENT = 5.5;
const LIGHT_DISTANCE = 20;
const OBJECT_STRIDE = 80; // mat4x4f + vec4f

interface Prop {
  mesh: Mesh;
  color: string;
  gloss: number;
  /** Model matrix as a function of animation time. */
  transform: (t: number) => Mat4;
}

const create: ExperimentFactory = async ({ device, format, surface, params, reducedMotion }) => {
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
      mode: { value: 4, options: { Off: 0, 'Hard (2×2 HW)': 1, 'PCF grid': 2, 'Poisson disc': 3, PCSS: 4 } },
      mapSize: { value: 2048, options: { '512': 512, '1024': 1024, '2048': 2048, '4096': 4096 }, label: 'Map size' },
      pcfRadius: { value: 2, min: 0, max: 6, step: 1, label: 'PCF radius', hint: 'Kernel radius in texels (PCF grid & Poisson).' },
      lightSize: { value: 2.5, min: 0.1, max: 12, step: 0.1, label: 'Sun size (°)', hint: 'Angular diameter used by PCSS. The real sun is ~0.53°.' },
      depthBias: { value: 0.0006, min: 0, max: 0.01, step: 0.0001, label: 'Depth bias' },
      normalOffset: { value: 1.5, min: 0, max: 6, step: 0.1, label: 'Normal offset', hint: 'In shadow-map texels.' },
    },
    'Shadows',
  );
  const look = params.add(
    {
      background: { value: '#c9c5bf', type: 'color' },
      sky: { value: '#a9bcd6', type: 'color' },
      ground: { value: '#6b5d52', type: 'color' },
      ambient: { value: 0.28, min: 0, max: 1.5, step: 0.01 },
      exposure: { value: 1, min: 0.1, max: 4, step: 0.01 },
      tonemap: { value: 2, options: { None: 0, ACES: 1, AgX: 2 } },
      view: { value: 0, options: { Lit: 0, Visibility: 1, 'Light depth': 2 } },
      showMap: { value: false, label: 'Show shadow map' },
      animate: { value: !reducedMotion },
    },
    'Look & debug',
  );

  // --- Scene ------------------------------------------------------------------
  const meshes = {
    ground: uploadMesh(device, plane(60, 1), 'ground'),
    box: uploadMesh(device, box(1, 1, 1), 'box'),
    sphere: uploadMesh(device, sphere(1, 64, 40), 'sphere'),
    torus: uploadMesh(device, torus(0.8, 0.26, 40, 128), 'torus'),
  };
  const T = (x: number, y: number, z: number) => mat4.translation([x, y, z]);
  const props: Prop[] = [
    { mesh: meshes.ground, color: '#d8cfc4', gloss: 0.1, transform: () => mat4.identity() },
    {
      mesh: meshes.torus,
      color: '#e9b78f',
      gloss: 0.6,
      transform: (t) => mat4.rotateX(mat4.rotateY(T(0, 1.15, 0), t * 0.35), Math.PI / 2),
    },
    { mesh: meshes.sphere, color: '#c95f4b', gloss: 0.5, transform: () => mat4.scale(T(-1.9, 0.6, 0.9), [0.6, 0.6, 0.6]) },
    { mesh: meshes.sphere, color: '#e8e2d6', gloss: 0.3, transform: () => mat4.scale(T(-0.7, 0.35, 2.1), [0.35, 0.35, 0.35]) },
    {
      mesh: meshes.sphere,
      color: '#7e9cc4',
      gloss: 0.7,
      transform: (t) => mat4.scale(T(0.4, 2.4 + 0.25 * Math.sin(t * 0.8), -1.9), [0.38, 0.38, 0.38]),
    },
    { mesh: meshes.box, color: '#a9b49f', gloss: 0.2, transform: () => mat4.scale(T(1.9, 1.1, -0.8), [0.6, 2.2, 0.6]) },
    {
      mesh: meshes.box,
      color: '#3f4a5c',
      gloss: 0.4,
      transform: () => mat4.scale(mat4.rotateY(T(1.7, 0.2, 1.5), 0.6), [1.2, 0.4, 1.2]),
    },
    // A thin pole: the textbook demo of contact-hardening (sharp at the base, soft at the tip).
    { mesh: meshes.box, color: '#2b2b30', gloss: 0.3, transform: () => mat4.scale(T(-1.7, 1.5, -1.6), [0.07, 3, 0.07]) },
  ];

  const objectData = new Float32Array((props.length * OBJECT_STRIDE) / 4);
  const objectBuffer = device.createBuffer({
    label: 'objects',
    size: objectData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const frame = new UniformBlock(device, 'Frame', {
    viewProj: 'mat4x4f',
    lightViewProj: 'mat4x4f',
    cameraPos: 'vec3f',
    exposure: 'f32',
    lightDir: 'vec3f',
    shadowMode: 'u32',
    lightColor: 'vec4f',
    skyColor: 'vec4f',
    groundColor: 'vec4f',
    fogColor: 'vec4f',
    fadeStart: 'f32',
    fadeEnd: 'f32',
    pcfRadius: 'f32',
    lightTan: 'f32',
    shadowExtent: 'f32',
    shadowDepthRange: 'f32',
    shadowTexelWorld: 'f32',
    depthBias: 'f32',
    normalOffset: 'f32',
    tonemap: 'u32',
    debugView: 'u32',
  });

  // --- Pipelines --------------------------------------------------------------
  const module = await createShader(
    device,
    'shadow-map-pcf',
    wgsl(commonWgsl, fullscreenWgsl, MESH_VERTEX_WGSL, frame.wgslStruct, shadowWgsl, shaderWgsl),
  );

  const bindGroupLayout = device.createBindGroupLayout({
    label: 'shadow-map-pcf',
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
    ],
  });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
  // The shadow pass can't bind the shadow map it is writing to, so it gets its own layout.
  const shadowBindGroupLayout = device.createBindGroupLayout({
    label: 'shadow-pass',
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
    ],
  });

  const [shadowPipeline, mainPipeline, overlayPipeline] = await Promise.all([
    device.createRenderPipelineAsync({
      label: 'shadow-pass',
      layout: device.createPipelineLayout({ bindGroupLayouts: [shadowBindGroupLayout] }),
      vertex: { module, entryPoint: 'vs_shadow', buffers: [MESH_VERTEX_LAYOUT] },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { format: SHADOW_FORMAT, depthWriteEnabled: true, depthCompare: 'less' },
    }),
    device.createRenderPipelineAsync({
      label: 'main-pass',
      layout,
      vertex: { module, entryPoint: 'vs_main', buffers: [MESH_VERTEX_LAYOUT] },
      fragment: { module, entryPoint: 'fs_main', targets: [{ format }] },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: 'less' },
      multisample: { count: SAMPLE_COUNT },
    }),
    device.createRenderPipelineAsync({
      label: 'shadow-overlay',
      layout,
      vertex: { module, entryPoint: 'vs_fullscreen' },
      fragment: { module, entryPoint: 'fs_overlay', targets: [{ format }] },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: 'always' },
      multisample: { count: SAMPLE_COUNT },
    }),
  ]);

  const shadowSampler = device.createSampler({
    label: 'shadow-compare',
    compare: 'less-equal',
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  });

  const shadowBindGroup = device.createBindGroup({
    layout: shadowBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: frame.buffer } },
      { binding: 1, resource: { buffer: objectBuffer } },
    ],
  });

  // --- Size-dependent resources -----------------------------------------------
  let shadowMap: GPUTexture | null = null;
  let mainBindGroup: GPUBindGroup | null = null;
  const ensureShadowMap = (size: number) => {
    if (shadowMap?.width === size) return;
    shadowMap?.destroy();
    shadowMap = device.createTexture({
      label: 'shadow-map',
      size: [size, size],
      format: SHADOW_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    mainBindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: frame.buffer } },
        { binding: 1, resource: { buffer: objectBuffer } },
        { binding: 2, resource: shadowMap.createView() },
        { binding: 3, resource: shadowSampler },
      ],
    });
  };

  let msaaColor: GPUTexture | null = null;
  let depth: GPUTexture | null = null;
  const ensureTargets = (w: number, h: number) => {
    if (msaaColor?.width === w && msaaColor.height === h) return;
    msaaColor?.destroy();
    depth?.destroy();
    msaaColor = device.createTexture({
      label: 'msaa-color',
      size: [w, h],
      format,
      sampleCount: SAMPLE_COUNT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    depth = device.createTexture({
      label: 'depth',
      size: [w, h],
      format: DEPTH_FORMAT,
      sampleCount: SAMPLE_COUNT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  };

  const camera = new OrbitCamera(surface.canvas, { target: [0, 0.9, 0], distance: 9.5, yaw: 0.75, pitch: 0.42, maxDistance: 25 });
  let animTime = 0;
  let lightSpin = 0;

  return {
    frame({ dt, size }) {
      if (look.animate) animTime += dt;
      if (light.orbitLight) lightSpin += dt * 0.15;
      camera.update(dt, size.width / size.height);
      ensureShadowMap(shadow.mapSize);
      ensureTargets(size.width, size.height);

      // Objects.
      props.forEach((p, i) => {
        const o = (i * OBJECT_STRIDE) / 4;
        objectData.set(p.transform(animTime), o);
        objectData.set([...hexToLinear(p.color), p.gloss], o + 16);
      });
      device.queue.writeBuffer(objectBuffer, 0, objectData);

      // Light: direction the light travels (from sun toward the scene).
      const az = (light.azimuth * Math.PI) / 180 + lightSpin;
      const el = (light.elevation * Math.PI) / 180;
      const toSun = vec3.fromValues(Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az));
      const center = vec3.fromValues(0, 1, 0);
      const eye = vec3.addScaled(center, toSun, LIGHT_DISTANCE);
      const up = Math.abs(toSun[1]) > 0.99 ? [0, 0, 1] : [0, 1, 0];
      const lightView = mat4.lookAt(eye, center, up);
      const near = 1;
      const far = LIGHT_DISTANCE * 2;
      const lightProj = mat4.ortho(-SHADOW_EXTENT, SHADOW_EXTENT, -SHADOW_EXTENT, SHADOW_EXTENT, near, far);
      const lightViewProj = mat4.multiply(lightProj, lightView);

      const lc = hexToLinear(light.color);
      const sky = hexToLinear(look.sky);
      const ground = hexToLinear(look.ground);
      const fog = hexToLinear(look.background);

      frame
        .set('viewProj', camera.viewProjection)
        .set('lightViewProj', lightViewProj)
        .set('cameraPos', camera.position)
        .set('exposure', look.exposure)
        .set('lightDir', vec3.negate(toSun))
        .set('shadowMode', shadow.mode)
        .set('lightColor', [lc[0] * light.intensity, lc[1] * light.intensity, lc[2] * light.intensity, 0])
        .set('skyColor', [sky[0] * look.ambient, sky[1] * look.ambient, sky[2] * look.ambient, 0])
        .set('groundColor', [ground[0] * look.ambient, ground[1] * look.ambient, ground[2] * look.ambient, 0])
        .set('fogColor', [...fog, 0])
        .set('fadeStart', 6)
        .set('fadeEnd', 22)
        .set('pcfRadius', shadow.pcfRadius)
        .set('lightTan', Math.tan(((shadow.lightSize / 2) * Math.PI) / 180))
        .set('shadowExtent', SHADOW_EXTENT)
        .set('shadowDepthRange', far - near)
        .set('shadowTexelWorld', (2 * SHADOW_EXTENT) / shadow.mapSize)
        .set('depthBias', shadow.depthBias)
        .set('normalOffset', shadow.normalOffset)
        .set('tonemap', look.tonemap)
        .set('debugView', look.view)
        .upload();

      const encoder = device.createCommandEncoder({ label: 'shadow-map-pcf' });

      const drawProps = (pass: GPURenderPassEncoder) => {
        props.forEach((p, i) => {
          pass.setVertexBuffer(0, p.mesh.vertexBuffer);
          pass.setIndexBuffer(p.mesh.indexBuffer, 'uint32');
          pass.drawIndexed(p.mesh.indexCount, 1, 0, 0, i);
        });
      };

      // 1. Shadow map.
      const sp = encoder.beginRenderPass({
        label: 'shadow-pass',
        colorAttachments: [],
        depthStencilAttachment: {
          view: shadowMap!.createView(),
          depthClearValue: 1,
          depthLoadOp: 'clear',
          depthStoreOp: 'store',
        },
      });
      sp.setPipeline(shadowPipeline);
      sp.setBindGroup(0, shadowBindGroup);
      drawProps(sp);
      sp.end();

      // 2. Main pass.
      const mp = encoder.beginRenderPass({
        label: 'main-pass',
        colorAttachments: [
          {
            view: msaaColor!.createView(),
            resolveTarget: surface.currentTexture().createView(),
            loadOp: 'clear',
            storeOp: 'discard',
            clearValue: [...fog.map(linearToSrgb), 1],
          },
        ],
        depthStencilAttachment: {
          view: depth!.createView(),
          depthClearValue: 1,
          depthLoadOp: 'clear',
          depthStoreOp: 'discard',
        },
      });
      mp.setPipeline(mainPipeline);
      mp.setBindGroup(0, mainBindGroup!);
      drawProps(mp);

      if (look.showMap) {
        const s = Math.round(Math.min(size.width, size.height) * 0.3);
        const m = Math.round(16 * size.pixelRatio);
        mp.setViewport(m, size.height - s - m, s, s, 0, 1);
        mp.setPipeline(overlayPipeline);
        mp.draw(3);
      }
      mp.end();

      device.queue.submit([encoder.finish()]);
    },
    dispose() {
      camera.destroy();
      shadowMap?.destroy();
      msaaColor?.destroy();
      depth?.destroy();
      objectBuffer.destroy();
      frame.destroy();
      Object.values(meshes).forEach((m) => m.destroy());
    },
  };
};

export default create;
