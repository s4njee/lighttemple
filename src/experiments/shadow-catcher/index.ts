import { mat4, vec3, type Mat4 } from 'wgpu-matrix';
import brdfWgsl from '../../shaders/brdf.wgsl?raw';
import commonWgsl from '../../shaders/common.wgsl?raw';
import fullscreenWgsl from '../../shaders/fullscreen.wgsl?raw';
import { hexToLinear } from '../../gpu/color';
import { MESH_VERTEX_LAYOUT, MESH_VERTEX_WGSL, sphere, torus, uploadMesh } from '../../gpu/geometry';
import { createShader, wgsl } from '../../gpu/shader';
import { TargetPool } from '../../gpu/targets';
import { UniformBlock } from '../../gpu/uniforms';
import { HDR_FORMAT, PostProcess } from '../../post/post';
import { OrbitCamera } from '../../scene/orbit-camera';
import { PROP_WGSL, PropSet, type Prop } from '../../scene/props';
import type { ExperimentFactory } from '../types';
import { buildBackdrop } from './backdrop';
import shaderWgsl from './shader.wgsl?raw';

const MSAA = 4;
const DEPTH_FORMAT: GPUTextureFormat = 'depth24plus';
const SHADOW_FORMAT: GPUTextureFormat = 'depth32float';
const CONTACT_MAP_SIZE = 256;
/** Camera: vertical field of view, and the page's visible half-height in world units. */
const FOV_Y = (28 * Math.PI) / 180;
const PAGE_HALF_HEIGHT = 1;

type ObjectChoice = 'planet' | 'pebbles';

/** Object compositions, built around the origin; `radius` bounds them, `bottom` is their lowest point. */
interface Composition {
  props: PropSet;
  radius: number;
  bottom: number;
  /** Local transform of each prop (before the drag rotation and the hover offset). */
  local: Mat4[];
}

const create: ExperimentFactory = async (ctx) => {
  const { device, surface, params, reducedMotion } = ctx;

  const scene = params.add(
    {
      object: { value: 'planet', options: { 'Ringed planet': 'planet', Pebbles: 'pebbles' } },
      hover: { value: 0.22, min: 0.005, max: 1, step: 0.005, label: 'Hover height', hint: 'Gap between the object\'s lowest point and the page (1 = half the visible page height).' },
      bob: { value: !reducedMotion, label: 'Float' },
      spin: { value: !reducedMotion, label: 'Auto-rotate' },
    },
    'Scene',
  );
  // Only the gallery has HTML of its own behind the canvas; a library host's page is its own business.
  const html = ctx.backdrop
    ? params.add(
        {
          page: {
            value: 'article',
            options: { 'Gradient & article': 'article', 'Dark article': 'dark', Checkerboard: 'checker' },
            label: 'HTML behind',
            hint: 'Real DOM content behind the transparent canvas. The checkerboard makes any wrong alpha obvious.',
          },
          interaction: {
            value: 'object',
            options: { 'Text selectable (drag the object)': 'object', 'Canvas takes the pointer': 'canvas' },
            hint: 'A canvas on top swallows every pointer event. "Text selectable" sets pointer-events: none on the canvas and puts an invisible handle over the object, so the page underneath stays selectable and clickable.',
          },
        },
        'Page',
      )
    : null;
  const light = params.add(
    {
      azimuth: { value: 55, min: -180, max: 180, step: 1, hint: 'Where the light comes from, around the page (degrees; 90 = above).' },
      elevation: { value: 52, min: 10, max: 89, step: 1, hint: 'Angle above the page plane. Lower = longer shadow.' },
      intensity: { value: 3.5, min: 0, max: 8, step: 0.05 },
      color: { value: '#fff3e3', type: 'color' },
      lightSize: { value: 4, min: 0.2, max: 20, step: 0.1, label: 'Light size (°)', hint: 'Angular diameter of the key light. Sets how fast the penumbra widens with height (PCSS).' },
    },
    'Light',
  );
  const shadow = params.add(
    {
      shadowOpacity: { value: 0.6, min: 0, max: 1, step: 0.01, label: 'Shadow' },
      contact: { value: 0.7, min: 0, max: 1, step: 0.01, label: 'Contact AO' },
      alphaBlend: {
        value: 'linear',
        options: { 'Linear light': 'linear', 'Coverage (plain CSS)': 'coverage' },
        label: 'Alpha',
        hint: 'Browsers blend canvases in sRGB space. "Linear light" remaps shadow alpha so the page darkens as it would in linear light (and as it does over a flat page colour); "Coverage" writes alpha = darkness, which reads darker and harder.',
      },
      taps: { value: 24, options: { '8': 8, '16': 16, '24': 24, '32': 32, '48': 48 }, label: 'Shadow taps', hint: 'PCSS filter taps per page pixel (blocker search uses half). Lower for phones.' },
      mapSize: { value: 1024, options: { '512': 512, '1024': 1024, '2048': 2048 }, label: 'Map size', hint: 'The key shadow map only covers the object, so 1024 is plenty.' },
      pageColor: {
        value: '#efe4d6',
        type: 'color',
        label: 'Page colour',
        hint: 'Tints the light the page bounces back onto the object. When the host canvas is opaque (no HTML behind), the catcher composites over this colour instead.',
      },
      view: { value: 0, options: { Composite: 0, 'Alpha only': 1, 'Premultiplied check': 2, 'Shadow mask': 3 }, hint: 'Alpha and the premultiplied check read back the final image the browser receives. Invalid pixels (rgb > alpha) turn red.' },
    },
    'Shadow & debug',
  );
  const post = await PostProcess.create(ctx, { bloom: 0.02, vignette: 0, grain: 0 });

  // --- Backdrop (gallery only): real HTML behind the transparent canvas -------------------
  const canvas = surface.canvas;
  const stage = canvas.parentElement;
  const backdrop = ctx.backdrop ? buildBackdrop(ctx.backdrop) : null;
  let handle: HTMLDivElement | null = null;
  if (ctx.backdrop && stage) {
    handle = document.createElement('div');
    handle.className = 'catcher-handle';
    handle.setAttribute('aria-label', 'Drag to rotate the object');
    handle.hidden = true; // shown by the "Text selectable" interaction mode
    stage.append(handle);
  }

  // --- Scene ---------------------------------------------------------------------------
  const meshes = {
    sphere: uploadMesh(device, sphere(1, 64, 40), 'sphere'),
    ring: uploadMesh(device, torus(1, 0.075, 24, 160), 'ring'),
  };
  const compose = (props: (Prop & { local: Mat4 })[], radius: number, bottom: number): Composition => {
    const set = new PropSet(device, props, 'catcher-props');
    return { props: set, radius, bottom, local: props.map((p) => p.local) };
  };
  const S = (x: number, y: number, z: number, sx: number, sy = sx, sz = sx) => mat4.scale(mat4.translation([x, y, z]), [sx, sy, sz]);
  const compositions: Record<ObjectChoice, Composition> = {
    planet: compose(
      [
        { mesh: meshes.sphere, color: '#2f6066', roughness: 0.3, local: S(0, 0, 0, 0.17) },
        // A thin gold ring, tilted off the sphere's equator.
        {
          mesh: meshes.ring,
          color: '#f3c27a',
          roughness: 0.28,
          metallic: 1,
          local: mat4.rotateZ(mat4.rotateX(mat4.scaling([0.3, 0.3, 0.3]), 1.15), 0.35),
        },
      ],
      0.32,
      -0.3,
    ),
    pebbles: compose(
      [
        // Three flattened stones at different heights: watch the shadow sharpen as they near the page.
        { mesh: meshes.sphere, color: '#c9c2b8', roughness: 0.7, local: mat4.rotateZ(S(-0.24, -0.06, -0.17, 0.14, 0.1, 0.06), 0.4) },
        { mesh: meshes.sphere, color: '#8f8679', roughness: 0.6, local: mat4.rotateZ(S(0.04, 0.11, 0.02, 0.12, 0.085, 0.055), -0.7) },
        { mesh: meshes.sphere, color: '#e7e1d6', roughness: 0.55, local: mat4.rotateZ(S(0.25, -0.08, 0.2, 0.1, 0.07, 0.045), 1.2) },
      ],
      0.36,
      -0.23,
    ),
  };

  const frame = new UniformBlock(device, 'Frame', {
    viewProj: 'mat4x4f',
    invViewProj: 'mat4x4f',
    keyViewProj: 'mat4x4f',
    contactViewProj: 'mat4x4f',
    cameraPos: 'vec3f',
    keyExtent: 'f32',
    toLight: 'vec3f',
    keyDepthRange: 'f32',
    lightColor: 'vec3f',
    lightTan: 'f32',
    skyColor: 'vec3f',
    shadowOpacity: 'f32',
    bounceColor: 'vec3f',
    contactStrength: 'f32',
    keyTexelWorld: 'f32',
    contactExtent: 'f32',
    contactRange: 'f32',
    contactRadius: 'f32',
    contactFalloff: 'f32',
    taps: 'u32',
    debugView: 'u32',
  });

  const module = await createShader(
    device,
    'shadow-catcher',
    wgsl(commonWgsl, fullscreenWgsl, MESH_VERTEX_WGSL, PROP_WGSL, brdfWgsl, frame.wgslStruct, shaderWgsl),
  );

  // --- Pipelines -------------------------------------------------------------------------
  const mainLayout = device.createBindGroupLayout({
    label: 'catcher-main',
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
    ],
  });
  const shadowLayout = device.createBindGroupLayout({
    label: 'catcher-shadow',
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
    ],
  });
  const layout = (g: GPUBindGroupLayout) => device.createPipelineLayout({ bindGroupLayouts: [g] });
  const depthPipeline = (entryPoint: string) =>
    device.createRenderPipelineAsync({
      label: `catcher:${entryPoint}`,
      layout: layout(shadowLayout),
      vertex: { module, entryPoint, buffers: [MESH_VERTEX_LAYOUT] },
      primitive: { cullMode: 'none' },
      depthStencil: { format: SHADOW_FORMAT, depthWriteEnabled: true, depthCompare: 'less' },
    });
  const [keyPipeline, contactPipeline, objectPipeline, pagePipeline, inspectPipeline] = await Promise.all([
    depthPipeline('vs_key'),
    depthPipeline('vs_contact'),
    device.createRenderPipelineAsync({
      label: 'catcher:object',
      layout: layout(mainLayout),
      vertex: { module, entryPoint: 'vs_object', buffers: [MESH_VERTEX_LAYOUT] },
      fragment: { module, entryPoint: 'fs_object', targets: [{ format: HDR_FORMAT }] },
      primitive: { cullMode: 'back' },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: 'less' },
      multisample: { count: MSAA },
    }),
    device.createRenderPipelineAsync({
      label: 'catcher:page',
      layout: layout(mainLayout),
      vertex: { module, entryPoint: 'vs_fullscreen' },
      fragment: { module, entryPoint: 'fs_page', targets: [{ format: HDR_FORMAT }] },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: 'always' },
      multisample: { count: MSAA },
    }),
    device.createRenderPipelineAsync({
      label: 'catcher:inspect',
      layout: 'auto',
      vertex: { module, entryPoint: 'vs_fullscreen' },
      fragment: { module, entryPoint: 'fs_inspect', targets: [{ format: ctx.format }] },
    }),
  ]);

  // --- Resources -----------------------------------------------------------------------
  const cmp = device.createSampler({ compare: 'less-equal', magFilter: 'linear', minFilter: 'linear' });
  const contactMap = device.createTexture({
    label: 'catcher-contact',
    size: [CONTACT_MAP_SIZE, CONTACT_MAP_SIZE],
    format: SHADOW_FORMAT,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  const targets = new TargetPool(device);

  let keyMap: GPUTexture | null = null;
  let groups: Record<ObjectChoice, { main: GPUBindGroup; shadow: GPUBindGroup }> | null = null;
  const ensureKeyMap = (size: number) => {
    if (keyMap?.width === size && groups) return;
    keyMap?.destroy();
    const map = device.createTexture({
      label: 'catcher-key',
      size: [size, size],
      format: SHADOW_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    keyMap = map;
    const make = (c: Composition) => ({
      main: device.createBindGroup({
        layout: mainLayout,
        entries: [
          { binding: 0, resource: { buffer: frame.buffer } },
          { binding: 1, resource: { buffer: c.props.buffer } },
          { binding: 2, resource: map.createView() },
          { binding: 3, resource: contactMap.createView() },
          { binding: 4, resource: cmp },
        ],
      }),
      shadow: device.createBindGroup({
        layout: shadowLayout,
        entries: [
          { binding: 0, resource: { buffer: frame.buffer } },
          { binding: 1, resource: { buffer: c.props.buffer } },
        ],
      }),
    });
    groups = { planet: make(compositions.planet), pebbles: make(compositions.pebbles) };
  };
  let inspectGroup: { texture: GPUTexture; group: GPUBindGroup } | null = null;

  // --- Interaction -----------------------------------------------------------------------
  // The orbit camera is only an input device here: its yaw/pitch spin the object, while the real
  // camera stays square-on to the page so the plane z = 0 keeps lining up with the HTML.
  const cameraOptions = { yaw: 0.6, pitch: 0.35, minPitch: -1.2, maxPitch: 1.2 };
  let input = new OrbitCamera(canvas, cameraOptions);
  let interaction = 'canvas';
  const setInteraction = (mode: string) => {
    if (mode === interaction) return;
    interaction = mode;
    const passthrough = mode === 'object' && handle !== null;
    const yaw = input.yaw;
    const pitch = input.pitch;
    input.destroy();
    input = new OrbitCamera(passthrough ? handle! : canvas, { ...cameraOptions, yaw, pitch });
    canvas.style.pointerEvents = passthrough ? 'none' : '';
    if (handle) handle.hidden = !passthrough;
  };

  const view = mat4.identity();
  const proj = mat4.identity();
  const viewProj = mat4.identity();
  let bobTime = 0;

  return {
    frame({ time, dt, size }) {
      setInteraction(html?.interaction ?? 'canvas');
      if (backdrop && html) backdrop.setVariant(html.page);
      ensureKeyMap(shadow.mapSize);
      if (scene.bob) bobTime += dt;
      input.autoRotate = scene.spin ? 0.25 : 0;
      input.update(dt, 1);

      // Camera: square-on to the page, framing a fixed page height.
      const aspect = size.width / size.height;
      const distance = PAGE_HALF_HEIGHT / Math.tan(FOV_Y / 2);
      const eye = vec3.fromValues(0, 0, distance);
      mat4.lookAt(eye, [0, 0, 0], [0, 1, 0], view);
      mat4.perspective(FOV_Y, aspect, 0.1, distance * 2, proj);
      mat4.multiply(proj, view, viewProj);

      // Object: spun by the drag input, floating `hover` above the page, a little right of centre.
      const comp = compositions[scene.object as ObjectChoice];
      const hover = scene.hover + (scene.bob ? 0.035 * Math.sin(bobTime * 1.3) : 0);
      const center = vec3.fromValues(Math.min(0.36 * aspect, 0.62), 0.05, hover - comp.bottom);
      const spin = mat4.rotateX(mat4.rotationY(input.yaw), input.pitch);
      const world = mat4.multiply(mat4.translation(center), spin);
      comp.props.props.forEach((p, i) => (p.transform = () => mat4.multiply(world, comp.local[i])));
      comp.props.update(time);

      // Key light: orthographic, fitted to the object; depth runs on to the page behind it.
      const az = (light.azimuth * Math.PI) / 180;
      const el = (light.elevation * Math.PI) / 180;
      const toLight = vec3.fromValues(Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el));
      const r = comp.radius * 1.15;
      const lightDistance = 4;
      const keyView = mat4.lookAt(vec3.addScaled(center, toLight, lightDistance), center, [0, 0, 1]);
      // Far plane: the page point behind the object's lowest corner, plus margin.
      const far = lightDistance + (center[2] + r) / Math.max(Math.sin(el), 0.05) + r;
      const near = lightDistance - r;
      const keyProj = mat4.ortho(-r, r, -r, r, near, far);
      const keyViewProj = mat4.multiply(keyProj, keyView);
      // Contact map: orthographic, looking up at the object from just behind the page.
      const contactRange = 0.5;
      const contactExtent = comp.radius * 1.2;
      const contactView = mat4.lookAt([center[0], center[1], -1], [center[0], center[1], 0], [0, 1, 0]);
      const contactViewProj = mat4.multiply(mat4.ortho(-contactExtent, contactExtent, -contactExtent, contactExtent, 1, 1 + contactRange), contactView);

      const lc = hexToLinear(light.color);
      frame
        .set('viewProj', viewProj)
        .set('invViewProj', mat4.inverse(viewProj))
        .set('keyViewProj', keyViewProj)
        .set('contactViewProj', contactViewProj)
        .set('cameraPos', eye)
        .set('keyExtent', r)
        .set('toLight', toLight)
        .set('keyDepthRange', far - near)
        .set('lightColor', lc.map((c) => c * light.intensity))
        .set('lightTan', Math.tan(((light.lightSize / 2) * Math.PI) / 180))
        .set('skyColor', [0.42, 0.41, 0.4])
        .set('bounceColor', hexToLinear(shadow.pageColor).map((c) => c * 0.3))
        .set('shadowOpacity', shadow.shadowOpacity)
        .set('contactStrength', shadow.contact)
        .set('keyTexelWorld', (2 * r) / shadow.mapSize)
        .set('contactExtent', contactExtent)
        .set('contactRange', contactRange)
        .set('contactRadius', 0.05)
        .set('contactFalloff', 0.06)
        .set('taps', shadow.taps)
        .set('debugView', shadow.view)
        .upload();

      const g = groups![scene.object as ObjectChoice];
      const encoder = device.createCommandEncoder({ label: 'shadow-catcher' });
      for (const [map, pipeline] of [
        [keyMap!, keyPipeline],
        [contactMap, contactPipeline],
      ] as const) {
        const sp = encoder.beginRenderPass({
          label: pipeline.label,
          colorAttachments: [],
          depthStencilAttachment: { view: map.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
        });
        sp.setPipeline(pipeline);
        sp.setBindGroup(0, g.shadow);
        comp.props.draw(sp);
        sp.end();
      }

      const full = [size.width, size.height] as const;
      const color = targets.get('msaa', { size: full, format: HDR_FORMAT, sampleCount: MSAA });
      const depth = targets.get('depth', { size: full, format: DEPTH_FORMAT, sampleCount: MSAA });
      const hdr = post.scene(size);
      const pass = encoder.beginRenderPass({
        label: 'catcher-main',
        colorAttachments: [{ view: color.view, resolveTarget: hdr.view, loadOp: 'clear', storeOp: 'discard', clearValue: [0, 0, 0, 0] }],
        depthStencilAttachment: { view: depth.view, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard' },
      });
      pass.setBindGroup(0, g.main);
      pass.setPipeline(pagePipeline);
      pass.draw(3);
      pass.setPipeline(objectPipeline);
      comp.props.draw(pass);
      pass.end();

      const out = surface.currentTexture().createView();
      const alphaBlend = shadow.alphaBlend as 'linear' | 'coverage';
      const background = hexToLinear(shadow.pageColor);
      if (shadow.view === 3) {
        post.render(encoder, out, { time, raw: true, output: 'opaque' });
      } else if (shadow.view === 1 || shadow.view === 2) {
        // Render the real final image off-screen, then inspect it.
        const final = targets.get('final', { size: full, format: ctx.format });
        post.render(encoder, final.view, { time, background, alphaBlend });
        if (inspectGroup?.texture !== final.texture) {
          inspectGroup = {
            texture: final.texture,
            group: device.createBindGroup({
              layout: inspectPipeline.getBindGroupLayout(0),
              entries: [
                { binding: 0, resource: { buffer: frame.buffer } },
                { binding: 5, resource: final.view },
              ],
            }),
          };
        }
        const ip = encoder.beginRenderPass({ colorAttachments: [{ view: out, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] });
        ip.setPipeline(inspectPipeline);
        ip.setBindGroup(0, inspectGroup.group);
        ip.draw(3);
        ip.end();
      } else {
        post.render(encoder, out, { time, background, alphaBlend });
      }
      device.queue.submit([encoder.finish()]);

      // Keep the drag handle over the object (CSS pixels, relative to the canvas).
      if (handle && !handle.hidden) {
        const c = vec3.transformMat4(center, viewProj);
        const edge = vec3.transformMat4(vec3.add(center, [0, comp.radius, 0]), viewProj);
        const x = (c[0] * 0.5 + 0.5) * size.cssWidth;
        const y = (0.5 - c[1] * 0.5) * size.cssHeight;
        const rad = Math.abs(edge[1] - c[1]) * 0.5 * size.cssHeight;
        handle.style.transform = `translate(${(x - rad).toFixed(1)}px, ${(y - rad).toFixed(1)}px)`;
        handle.style.width = handle.style.height = `${(2 * rad).toFixed(1)}px`;
      }
    },
    dispose() {
      input.destroy();
      canvas.style.pointerEvents = '';
      handle?.remove();
      backdrop?.dispose();
      Object.values(compositions).forEach((c) => c.props.destroy());
      Object.values(meshes).forEach((m) => m.destroy());
      keyMap?.destroy();
      contactMap.destroy();
      targets.destroy();
      frame.destroy();
      post.destroy();
    },
  };
};

export default create;
