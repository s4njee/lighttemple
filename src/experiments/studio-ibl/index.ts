import { mat4, vec3 } from 'wgpu-matrix';
import brdfWgsl from '../../shaders/brdf.wgsl?raw';
import commonWgsl from '../../shaders/common.wgsl?raw';
import fullscreenWgsl from '../../shaders/fullscreen.wgsl?raw';
import { MESH_VERTEX_LAYOUT, MESH_VERTEX_WGSL, plane, uploadMesh } from '../../gpu/geometry';
import { createShader, wgsl } from '../../gpu/shader';
import { TargetPool } from '../../gpu/targets';
import { UniformBlock } from '../../gpu/uniforms';
import { GLTF_VERTEX_LAYOUT, getMaterialBindGroupLayout, gltfWgsl, loadGltf, type GltfModel } from '../../gltf/gltf';
import { parseHdr } from '../../ibl/hdr';
import { Ibl, iblWgsl } from '../../ibl/ibl';
import { HDR_FORMAT, PostProcess } from '../../post/post';
import { OrbitCamera } from '../../scene/orbit-camera';
import { hexToLinear } from '../../ui/params';
import type { ExperimentFactory } from '../types';
import { materialGridGltf } from './material-grid';
import shaderWgsl from './shader.wgsl?raw';

const MSAA = 4;
const DEPTH_FORMAT: GPUTextureFormat = 'depth24plus';
const KEY_MAP_SIZE = 2048;
const CONTACT_MAP_SIZE = 512;

const asset = (path: string) => new URL(path, document.baseURI).href;

type EnvChoice = 'studio' | 'sunset' | 'day' | 'hdri';
type ModelChoice = 'lantern' | 'grid';

/** Per-model framing: fitted height, camera distance. */
const FRAMING: Record<ModelChoice | 'dropped', { height: number; distance: number }> = {
  lantern: { height: 2.4, distance: 6.2 },
  grid: { height: 1.1, distance: 4.6 },
  dropped: { height: 1.8, distance: 5.2 },
};

const create: ExperimentFactory = async (ctx) => {
  const { device, surface, params, reducedMotion } = ctx;

  const scene = params.add(
    {
      model: { value: 'lantern', options: { 'Lantern (glTF)': 'lantern', 'Material grid': 'grid' } },
      environment: {
        value: 'studio',
        options: { 'Studio (procedural)': 'studio', 'Sunset sky': 'sunset', 'Day sky': 'day', 'Studio HDRI': 'hdri' },
        hint: 'Drop your own .hdr (equirectangular) or .glb onto the canvas to try them.',
      },
      rotation: { value: 0, min: -180, max: 180, step: 1, label: 'Env rotation' },
      intensity: { value: 1, min: 0, max: 4, step: 0.01, label: 'Env intensity' },
      emissive: { value: 3, min: 0, max: 20, step: 0.1, label: 'Emissive' },
      autoRotate: { value: !reducedMotion, label: 'Auto-rotate' },
    },
    'Scene',
  );
  const backdrop = params.add(
    {
      background: { value: 'env', options: { Environment: 'env', 'Page colour': 'page' } },
      blur: { value: 0.45, min: 0, max: 1, step: 0.01, label: 'Background blur' },
      pageColor: { value: '#d9d3ca', type: 'color', label: 'Page colour' },
      shadowOpacity: { value: 0.55, min: 0, max: 1, step: 0.01, label: 'Shadow' },
      shadowSoftness: { value: 0.35, min: 0.01, max: 1.5, step: 0.01, label: 'Softness', hint: 'Filter radius in world units.' },
      contact: { value: 0.75, min: 0, max: 1, step: 0.01, label: 'Contact AO' },
    },
    'Backdrop & shadow',
  );
  const post = await PostProcess.create(ctx, { bloom: 0.05, vignette: 0.2, grain: 0.02 });

  const ibl = await Ibl.create(device);
  const targets = new TargetPool(device);

  const frame = new UniformBlock(device, 'Frame', {
    viewProj: 'mat4x4f',
    invViewProj: 'mat4x4f',
    modelRoot: 'mat4x4f',
    keyViewProj: 'mat4x4f',
    contactViewProj: 'mat4x4f',
    cameraPos: 'vec3f',
    backgroundBlur: 'f32',
    emissiveStrength: 'f32',
    groundSize: 'f32',
    shadowOpacity: 'f32',
    shadowSoftness: 'f32',
    keyExtent: 'f32',
    contactExtent: 'f32',
    contactRange: 'f32',
    contactRadius: 'f32',
    contactFalloff: 'f32',
    contactStrength: 'f32',
  });

  const module = await createShader(
    device,
    'studio-ibl',
    wgsl(commonWgsl, fullscreenWgsl, MESH_VERTEX_WGSL, brdfWgsl, frame.wgslStruct, ibl.wgslStruct, iblWgsl, gltfWgsl, shaderWgsl),
  );

  // --- Layouts & pipelines ---------------------------------------------------------
  const g0 = device.createBindGroupLayout({
    label: 'studio-frame',
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
    ],
  });
  const shadowG0 = device.createBindGroupLayout({
    label: 'studio-shadow',
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
    ],
  });
  const materialLayout = getMaterialBindGroupLayout(device);
  const layout = (...groups: GPUBindGroupLayout[]) => device.createPipelineLayout({ bindGroupLayouts: groups });

  const depthState = (write: boolean): GPUDepthStencilState => ({ format: DEPTH_FORMAT, depthWriteEnabled: write, depthCompare: 'less' });
  const shadowPipeline = (entryPoint: string) =>
    device.createRenderPipelineAsync({
      label: `studio:${entryPoint}`,
      layout: layout(shadowG0),
      vertex: { module, entryPoint, buffers: [GLTF_VERTEX_LAYOUT] },
      primitive: { cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
    });
  const modelPipeline = (cullMode: GPUCullMode) =>
    device.createRenderPipelineAsync({
      label: `studio:model-${cullMode}`,
      layout: layout(g0, ibl.bindGroupLayout, materialLayout),
      vertex: { module, entryPoint: 'vs_model', buffers: [GLTF_VERTEX_LAYOUT] },
      fragment: { module, entryPoint: 'fs_model', targets: [{ format: HDR_FORMAT }] },
      primitive: { cullMode },
      depthStencil: depthState(true),
      multisample: { count: MSAA },
    });
  const premultipliedOver: GPUBlendState = {
    color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
    alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
  };
  const [keyPipeline, contactPipeline, modelBack, modelNone, skyPipeline, groundPipeline] = await Promise.all([
    shadowPipeline('vs_key'),
    shadowPipeline('vs_contact'),
    modelPipeline('back'),
    modelPipeline('none'),
    device.createRenderPipelineAsync({
      label: 'studio:sky',
      layout: layout(g0, ibl.bindGroupLayout),
      vertex: { module, entryPoint: 'vs_fullscreen' },
      fragment: { module, entryPoint: 'fs_sky', targets: [{ format: HDR_FORMAT }] },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: 'always' },
      multisample: { count: MSAA },
    }),
    device.createRenderPipelineAsync({
      label: 'studio:ground',
      layout: layout(g0),
      vertex: { module, entryPoint: 'vs_ground', buffers: [MESH_VERTEX_LAYOUT] },
      fragment: { module, entryPoint: 'fs_ground', targets: [{ format: HDR_FORMAT, blend: premultipliedOver }] },
      depthStencil: depthState(false),
      multisample: { count: MSAA },
    }),
  ]);

  // --- Resources ---------------------------------------------------------------------
  const ground = uploadMesh(device, plane(1, 1), 'ground');
  const shadowTex = (size: number, label: string) =>
    device.createTexture({ label, size: [size, size], format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
  const keyMap = shadowTex(KEY_MAP_SIZE, 'key-shadow');
  const contactMap = shadowTex(CONTACT_MAP_SIZE, 'contact-depth');
  const cmp = device.createSampler({ compare: 'less-equal', magFilter: 'linear', minFilter: 'linear' });

  let model: GltfModel | null = null;
  let modelKey: ModelChoice | 'dropped' = 'lantern';
  let groups: { main: GPUBindGroup; shadow: GPUBindGroup } | null = null;
  const setModel = (m: GltfModel, key: ModelChoice | 'dropped') => {
    model?.destroy();
    model = m;
    modelKey = key;
    groups = {
      main: device.createBindGroup({
        layout: g0,
        entries: [
          { binding: 0, resource: { buffer: frame.buffer } },
          { binding: 1, resource: { buffer: m.transforms } },
          { binding: 2, resource: keyMap.createView() },
          { binding: 3, resource: contactMap.createView() },
          { binding: 4, resource: cmp },
        ],
      }),
      shadow: device.createBindGroup({
        layout: shadowG0,
        entries: [
          { binding: 0, resource: { buffer: frame.buffer } },
          { binding: 1, resource: { buffer: m.transforms } },
        ],
      }),
    };
  };

  // Loads are async; only the most recent request wins.
  let modelRequest = 0;
  const loadModel = async (choice: ModelChoice) => {
    const id = ++modelRequest;
    const m = choice === 'grid' ? await loadGltf(device, materialGridGltf()) : await loadGltf(device, asset('assets/models/lantern.glb'));
    if (id !== modelRequest || disposed) return m.destroy();
    setModel(m, choice);
  };

  let envRequest = 0;
  let hdriCache: ReturnType<typeof parseHdr> | null = null;
  const loadEnvironment = async (choice: EnvChoice) => {
    const id = ++envRequest;
    if (choice === 'hdri') {
      hdriCache ??= parseHdr(await (await fetch(asset('assets/hdri/studio_small_09_1k.hdr'))).arrayBuffer());
      if (id !== envRequest || disposed) return;
      await ibl.setEquirect(hdriCache);
    } else if (choice === 'studio') {
      await ibl.setProcedural({ kind: 'studio' });
    } else {
      const sunset = choice === 'sunset';
      await ibl.setProcedural({ kind: 'sky', sunAzimuth: sunset ? 2.2 : 0.9, sunElevation: sunset ? 0.07 : 0.85, exposure: sunset ? 1.6 : 1 });
    }
  };

  let disposed = false;
  await Promise.all([loadModel(scene.model as ModelChoice), loadEnvironment(scene.environment as EnvChoice)]);
  let currentModel = scene.model;
  let currentEnv = scene.environment;

  // Drag & drop your own .hdr / .glb.
  const onDragOver = (e: DragEvent) => e.preventDefault();
  const onDrop = async (e: DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer?.files[0];
    if (!file) return;
    const buf = await file.arrayBuffer();
    try {
      if (/\.hdr$/i.test(file.name)) {
        envRequest++;
        await ibl.setEquirect(parseHdr(buf));
      } else if (/\.(glb|gltf)$/i.test(file.name)) {
        modelRequest++;
        setModel(await loadGltf(device, buf), 'dropped');
      }
    } catch (err) {
      console.error(err);
      alert(`Could not load ${file.name}: ${err instanceof Error ? err.message : err}`);
    }
  };
  surface.canvas.addEventListener('dragover', onDragOver);
  surface.canvas.addEventListener('drop', onDrop);

  const camera = new OrbitCamera(surface.canvas, { target: [0, 1.2, 0], distance: 6.2, yaw: 0.5, pitch: 0.18, minDistance: 1.5, maxDistance: 20 });
  camera.autoRotate = 0;
  let framedFor: string | null = null;

  return {
    frame({ time, dt, size }) {
      // React to select changes (async; the old asset stays up until the new one is ready).
      if (scene.model !== currentModel) {
        currentModel = scene.model;
        void loadModel(scene.model as ModelChoice);
      }
      if (scene.environment !== currentEnv) {
        currentEnv = scene.environment;
        void loadEnvironment(scene.environment as EnvChoice);
      }
      if (!model || !groups) return;

      const framing = FRAMING[modelKey];
      if (framedFor !== modelKey) {
        framedFor = modelKey;
        camera.setView({ target: [0, framing.height * 0.45, 0], distance: framing.distance });
      }
      camera.autoRotate = scene.autoRotate ? 0.15 : 0;
      camera.update(dt, size.width / size.height);

      ibl.rotation = (scene.rotation * Math.PI) / 180;
      ibl.intensity = scene.intensity;
      ibl.update();

      const root = model.fit(framing.height);
      const h = framing.height;
      const reach = h * 1.6;

      // Key shadow light = the environment's dominant direction, kept above 15°.
      let [lx, ly, lz] = ibl.lightDirection();
      ly = Math.max(ly, 0.26);
      const ldir = vec3.normalize([lx, ly, lz]);
      const center = vec3.fromValues(0, h * 0.5, 0);
      const keyView = mat4.lookAt(vec3.addScaled(center, ldir, 20), center, Math.abs(ldir[1]) > 0.99 ? [0, 0, 1] : [0, 1, 0]);
      const keyViewProj = mat4.multiply(mat4.ortho(-reach, reach, -reach, reach, 1, 40), keyView);
      // Contact map: orthographic, looking straight up from just below the floor.
      const contactRange = h;
      const contactView = mat4.lookAt([0, -1, 0], [0, 0, 0], [0, 0, 1]);
      const contactViewProj = mat4.multiply(mat4.ortho(-reach, reach, -reach, reach, 1, 1 + contactRange), contactView);

      frame
        .set('viewProj', camera.viewProjection)
        .set('invViewProj', mat4.inverse(camera.viewProjection))
        .set('modelRoot', root)
        .set('keyViewProj', keyViewProj)
        .set('contactViewProj', contactViewProj)
        .set('cameraPos', camera.position)
        .set('backgroundBlur', backdrop.blur)
        .set('emissiveStrength', scene.emissive)
        .set('groundSize', h * 6)
        .set('shadowOpacity', backdrop.shadowOpacity)
        .set('shadowSoftness', backdrop.shadowSoftness * h * 0.5)
        .set('keyExtent', reach)
        .set('contactExtent', reach)
        .set('contactRange', contactRange)
        .set('contactRadius', h * 0.06)
        .set('contactFalloff', h * 0.08)
        .set('contactStrength', backdrop.contact)
        .upload();

      const encoder = device.createCommandEncoder({ label: 'studio-ibl' });
      const drawModel = (pass: GPURenderPassEncoder, materials: boolean) => {
        for (const item of model!.items) {
          if (materials) {
            pass.setPipeline(item.material.doubleSided ? modelNone : modelBack);
            pass.setBindGroup(2, item.material.bindGroup);
          }
          pass.setVertexBuffer(0, item.vertexBuffer);
          pass.setIndexBuffer(item.indexBuffer, item.indexFormat);
          pass.drawIndexed(item.indexCount, 1, 0, 0, item.transformIndex);
        }
      };
      for (const [map, pipeline] of [
        [keyMap, keyPipeline],
        [contactMap, contactPipeline],
      ] as const) {
        const sp = encoder.beginRenderPass({
          colorAttachments: [],
          depthStencilAttachment: { view: map.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
        });
        sp.setPipeline(pipeline);
        sp.setBindGroup(0, groups.shadow);
        drawModel(sp, false);
        sp.end();
      }

      const full = [size.width, size.height] as const;
      const color = targets.get('msaa', { size: full, format: HDR_FORMAT, sampleCount: MSAA });
      const depth = targets.get('depth', { size: full, format: DEPTH_FORMAT, sampleCount: MSAA });
      const sceneTarget = post.scene(size);
      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: color.view, resolveTarget: sceneTarget.view, loadOp: 'clear', storeOp: 'discard', clearValue: [0, 0, 0, 0] }],
        depthStencilAttachment: { view: depth.view, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard' },
      });
      pass.setBindGroup(0, groups.main);
      pass.setBindGroup(1, ibl.bindGroup);
      if (backdrop.background === 'env') {
        pass.setPipeline(skyPipeline);
        pass.draw(3);
      }
      drawModel(pass, true);
      pass.setPipeline(groundPipeline);
      pass.setVertexBuffer(0, ground.vertexBuffer);
      pass.setIndexBuffer(ground.indexBuffer, 'uint32');
      pass.drawIndexed(ground.indexCount);
      pass.end();

      post.render(encoder, surface.currentTexture().createView(), {
        time,
        background: hexToLinear(backdrop.pageColor),
      });
      device.queue.submit([encoder.finish()]);
    },
    dispose() {
      disposed = true;
      surface.canvas.removeEventListener('dragover', onDragOver);
      surface.canvas.removeEventListener('drop', onDrop);
      camera.destroy();
      model?.destroy();
      ground.destroy();
      keyMap.destroy();
      contactMap.destroy();
      targets.destroy();
      frame.destroy();
      ibl.destroy();
      post.destroy();
    },
  };
};

export default create;
