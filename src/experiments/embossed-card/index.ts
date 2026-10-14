import { mat4, vec3 } from 'wgpu-matrix';
import brdfWgsl from '../../shaders/brdf.wgsl?raw';
import commonWgsl from '../../shaders/common.wgsl?raw';
import fullscreenWgsl from '../../shaders/fullscreen.wgsl?raw';
import noiseWgsl from '../../shaders/noise.wgsl?raw';
import sdfWgsl from '../../shaders/sdf.wgsl?raw';
import { createShader, wgsl } from '../../gpu/shader';
import { UniformBlock } from '../../gpu/uniforms';
import { Ibl, iblWgsl } from '../../ibl/ibl';
import { HDR_FORMAT, PostProcess } from '../../post/post';
import { TextSdfGenerator, type TextSdf } from '../../sdf/text-sdf';
import { hexToLinear } from '../../gpu/color';
import type { ExperimentFactory } from '../types';
import bakeWgsl from './bake.wgsl?raw';
import shaderWgsl from './shader.wgsl?raw';

/** Standard 3.5" × 2" card, in world units (1 unit ≈ 44 mm). */
const CARD_HALF: [number, number] = [1.0, 1.0 / 1.75];
const CARD_RADIUS = 0.035;
const FOV = (28 * Math.PI) / 180;

interface MaterialPreset {
  albedo: string;
  roughness: number;
  metallic: number;
  micro: number;
  relief: 1 | -1;
  foil: string;
}

const MATERIALS: Record<string, MaterialPreset & { label: string }> = {
  paper: { label: 'Cotton paper (letterpress)', albedo: '#f1ece2', roughness: 0.85, metallic: 0, micro: 0.05, relief: -1, foil: 'ink' },
  metal: { label: 'Brushed steel', albedo: '#c8cbd0', roughness: 0.32, metallic: 1, micro: 0.025, relief: 1, foil: 'none' },
  leather: { label: 'Black leather', albedo: '#1e1a18', roughness: 0.55, metallic: 0, micro: 0.09, relief: -1, foil: 'gold' },
  matte: { label: 'Matte black stock', albedo: '#141414', roughness: 0.75, metallic: 0, micro: 0.02, relief: 1, foil: 'gold' },
};
const MATERIAL_INDEX: Record<string, number> = { paper: 0, metal: 1, leather: 2, matte: 3 };

const FOILS: Record<string, { albedo: string; roughness: number; metallic: number }> = {
  gold: { albedo: '#ffd27a', roughness: 0.22, metallic: 1 },
  silver: { albedo: '#eef0f3', roughness: 0.18, metallic: 1 },
  copper: { albedo: '#f2a27a', roughness: 0.25, metallic: 1 },
  ink: { albedo: '#1d2a4a', roughness: 0.6, metallic: 0 },
};

const create: ExperimentFactory = async (ctx) => {
  const { device, surface, params, pointer, reducedMotion } = ctx;
  const coarse = window.matchMedia('(pointer: coarse)').matches;

  const card = params.add(
    {
      material: {
        value: 'paper',
        options: Object.fromEntries(Object.entries(MATERIALS).map(([k, m]) => [m.label, k])),
        hint: 'Each material comes with its own relief direction and foil default.',
      },
      headline: { value: 'Light Temple', type: 'text', maxLength: 32 },
      subline: { value: 'WEBGPU LIGHTING STUDIO', type: 'text', maxLength: 48 },
      relief: { value: 'auto', options: { 'Material default': 'auto', Emboss: 'emboss', Deboss: 'deboss' } },
      foil: { value: 'auto', options: { 'Material default': 'auto', None: 'none', Gold: 'gold', Silver: 'silver', Copper: 'copper', Ink: 'ink' } },
      depth: { value: 0.6, min: 0.1, max: 2, step: 0.01, label: 'Depth (mm)' },
      bevel: { value: 0.35, min: 0.05, max: 1.5, step: 0.01, label: 'Shoulder (mm)', hint: 'Width of the pressed edge. Real dies are around 0.2–0.5 mm.' },
      border: { value: true },
    },
    'Card',
  );
  const light = params.add(
    {
      lightHeight: { value: 0.45, min: 0.08, max: 2, step: 0.01, label: 'Height', hint: 'Lower = more raking light, deeper relief shadows.' },
      intensity: { value: 1.4, min: 0, max: 6, step: 0.01 },
      color: { value: '#ffe2c4', type: 'color' },
      lightRadius: { value: 0.08, min: 0.005, max: 0.5, step: 0.005, label: 'Radius' },
      environment: { value: 0.35, min: 0, max: 1.5, step: 0.01, hint: 'Studio IBL fill (gives metals something to reflect).' },
    },
    'Light',
  );
  const view = params.add(
    {
      tilt: { value: 32, min: 0, max: 60, step: 1, label: 'Tilt (°)', hint: 'Viewing angle. Parallax occlusion mapping only shows depth off-axis.' },
      lift: { value: 0.03, min: 0, max: 0.3, step: 0.005, hint: 'How far the card floats above the desk.' },
      table: { value: '#3a2a20', type: 'color', label: 'Desk' },
      pomSteps: { value: coarse ? 16 : 32, min: 4, max: 96, step: 1, label: 'POM steps' },
      parallax: { value: !reducedMotion, label: 'Pointer parallax' },
      debug: { value: 0, options: { Lit: 0, Height: 1, Normals: 2, 'Relief shadow': 3 } },
    },
    'View & debug',
  );
  const post = await PostProcess.create(ctx, { bloom: 0.03, vignette: 0.35, grain: 0.025, exposure: 1.1 });

  const ibl = await Ibl.create(device);
  await ibl.setProcedural({ kind: 'studio' });
  const textGen = new TextSdfGenerator(device);

  // --- Bake (height field) -----------------------------------------------------
  const bakeUniforms = new UniformBlock(device, 'Bake', {
    cardHalf: 'vec2f',
    material: 'u32',
    relief: 'f32',
    headRect: 'vec4f',
    subRect: 'vec4f',
    headScalePad: 'vec2f',
    subScalePad: 'vec2f',
    logoPos: 'vec2f',
    bevel: 'f32',
    microAmount: 'f32',
    seed: 'f32',
    border: 'u32',
  });
  const bakeModule = await createShader(
    device,
    'embossed-card-bake',
    wgsl(commonWgsl, noiseWgsl, sdfWgsl, fullscreenWgsl, bakeUniforms.wgslStruct, bakeWgsl),
  );
  const bakePipeline = await device.createRenderPipelineAsync({
    label: 'embossed-card-bake',
    layout: 'auto',
    vertex: { module: bakeModule, entryPoint: 'vs_fullscreen' },
    fragment: { module: bakeModule, entryPoint: 'fs_bake', targets: [{ format: 'rgba16float' }] },
  });
  const texWidth = coarse ? 1536 : 2560;
  const heightTex = device.createTexture({
    label: 'card-height',
    size: [texWidth, Math.round(texWidth / 1.75)],
    format: 'rgba16float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  const linear = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

  const FONT = '"Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif';
  let head: TextSdf | null = null;
  let sub: TextSdf | null = null;
  let textKey = '';
  let bakeKey = '';
  const ensureText = async () => {
    const key = `${card.headline}|${card.subline}`;
    if (key === textKey) return;
    textKey = key;
    const [h, s] = await Promise.all([
      textGen.generate({ text: card.headline.trim() || ' ', family: FONT, weight: 600, size: 200 }),
      textGen.generate({ text: card.subline.trim() || ' ', family: '"Helvetica Neue", Arial, ui-sans-serif, sans-serif', weight: 500, size: 120 }),
    ]);
    if (key !== textKey) {
      h.destroy();
      s.destroy();
      return;
    }
    head?.destroy();
    sub?.destroy();
    head = h;
    sub = s;
    bakeKey = ''; // force a re-bake
  };

  /** Places a text SDF centred at (cx, cy) with a given glyph cap height in card units. */
  const placeText = (t: TextSdf, cx: number, cy: number, capHeight: number, maxWidth: number) => {
    const glyphH = t.height - 2 * t.padding;
    let scale = capHeight / Math.max(glyphH, 1); // card units per texel
    if ((t.width - 2 * t.padding) * scale > maxWidth) scale = maxWidth / (t.width - 2 * t.padding);
    return { rect: [cx, cy, (t.width * scale) / 2, (t.height * scale) / 2], scalePad: [scale, t.padding * scale] };
  };

  const runBake = () => {
    if (!head || !sub) return;
    const preset = MATERIALS[card.material] ?? MATERIALS.paper;
    const relief = card.relief === 'auto' ? preset.relief : card.relief === 'emboss' ? 1 : -1;
    const key = JSON.stringify([card.material, relief, card.bevel, card.border, textKey]);
    if (key === bakeKey) return;
    bakeKey = key;
    const mm = 1 / 44; // world units per millimetre
    const h = placeText(head, 0, -0.02, 0.17, 1.6);
    const s = placeText(sub, 0, -0.28, 0.045, 1.3);
    bakeUniforms
      .set('cardHalf', CARD_HALF)
      .set('material', MATERIAL_INDEX[card.material] ?? 0)
      .set('relief', relief)
      .set('headRect', h.rect)
      .set('headScalePad', h.scalePad)
      .set('subRect', s.rect)
      .set('subScalePad', s.scalePad)
      .set('logoPos', [0, 0.3])
      .set('bevel', card.bevel * mm)
      .set('microAmount', preset.micro)
      .set('seed', 0.37)
      .set('border', card.border ? 1 : 0)
      .upload();
    const group = device.createBindGroup({
      layout: bakePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bakeUniforms.buffer } },
        { binding: 1, resource: head.texture.createView() },
        { binding: 2, resource: sub.texture.createView() },
        { binding: 3, resource: linear },
      ],
    });
    const encoder = device.createCommandEncoder({ label: 'card-bake' });
    const pass = encoder.beginRenderPass({ colorAttachments: [{ view: heightTex.createView(), loadOp: 'clear', storeOp: 'store' }] });
    pass.setPipeline(bakePipeline);
    pass.setBindGroup(0, group);
    pass.draw(3);
    pass.end();
    device.queue.submit([encoder.finish()]);
  };
  await ensureText();
  runBake();

  let textTimer = 0;
  params.onChange(() => {
    clearTimeout(textTimer);
    textTimer = window.setTimeout(() => void ensureText().then(runBake), 200);
  });

  // --- Main pass -----------------------------------------------------------------
  const frame = new UniformBlock(device, 'Frame', {
    invViewProj: 'mat4x4f',
    cameraPos: 'vec3f',
    reliefDepth: 'f32',
    lightPos: 'vec3f',
    lightRadius: 'f32',
    lightColor: 'vec3f',
    envIntensity: 'f32',
    baseAlbedo: 'vec3f',
    baseRoughness: 'f32',
    foilAlbedo: 'vec3f',
    foilRoughness: 'f32',
    tableAlbedo: 'vec3f',
    baseMetallic: 'f32',
    cardHalf: 'vec2f',
    cardLift: 'f32',
    cardRadius: 'f32',
    foilMetallic: 'f32',
    pixelAngle: 'f32',
    foil: 'u32',
    pomSteps: 'u32',
    debugView: 'u32',
  });
  const module = await createShader(
    device,
    'embossed-card',
    wgsl(commonWgsl, noiseWgsl, sdfWgsl, brdfWgsl, fullscreenWgsl, frame.wgslStruct, ibl.wgslStruct, iblWgsl, shaderWgsl),
  );
  const g0 = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
    ],
  });
  const pipeline = await device.createRenderPipelineAsync({
    label: 'embossed-card',
    layout: device.createPipelineLayout({ bindGroupLayouts: [g0, ibl.bindGroupLayout] }),
    vertex: { module, entryPoint: 'vs_fullscreen' },
    fragment: { module, entryPoint: 'fs_main', targets: [{ format: HDR_FORMAT }] },
  });
  const group = device.createBindGroup({
    layout: g0,
    entries: [
      { binding: 0, resource: { buffer: frame.buffer } },
      { binding: 1, resource: heightTex.createView() },
      { binding: 2, resource: linear },
    ],
  });

  let lightXY: [number, number] = [0.3, 0.2];
  let tiltX = 0;
  let tiltY = 0;

  return {
    frame({ time, dt, size }) {
      const aspect = size.width / size.height;
      // Camera: orbit the card centre; the pointer adds a few degrees of parallax.
      const k = 1 - Math.pow(2, -dt / 0.25);
      const [pu, pv] = pointer.inside && view.parallax ? pointer.smoothUv : [0.5, 0.5];
      tiltX += ((pu - 0.5) * 0.12 - tiltX) * k;
      tiltY += ((pv - 0.5) * 0.08 - tiltY) * k;
      const tilt = (view.tilt * Math.PI) / 180 + tiltY;
      // Fit the card (with a margin) in both directions.
      const fitW = (CARD_HALF[0] * 1.3) / (Math.tan(FOV / 2) * aspect);
      const fitH = (CARD_HALF[1] * 1.45) / Math.tan(FOV / 2);
      const dist = Math.max(fitW, fitH);
      const eye = vec3.fromValues(Math.sin(tiltX) * dist, -Math.sin(tilt) * dist, Math.cos(tilt) * Math.cos(tiltX) * dist);
      const viewM = mat4.lookAt(eye, [0, 0, 0], [0, 1, 0]);
      const proj = mat4.perspective(FOV, aspect, 0.05, 50);
      const viewProj = mat4.multiply(proj, viewM);
      const inv = mat4.inverse(viewProj);

      // Light: where the pointer ray meets the card plane, raised by `lightHeight`.
      const idle = !pointer.inside || pointer.idle > 4;
      let target: [number, number];
      if (idle) {
        target = [Math.sin(time * 0.37) * 0.9, Math.sin(time * 0.53) * 0.45];
      } else {
        const ndc = [pointer.uv[0] * 2 - 1, 1 - pointer.uv[1] * 2];
        const near = vec3.transformMat4([ndc[0], ndc[1], 0], inv);
        const far = vec3.transformMat4([ndc[0], ndc[1], 1], inv);
        const d = vec3.subtract(far, near);
        const t = (view.lift - near[2]) / d[2];
        target = [near[0] + d[0] * t, near[1] + d[1] * t];
      }
      const lk = 1 - Math.pow(2, -dt / (idle ? 0.8 : 0.07));
      lightXY = [lightXY[0] + (target[0] - lightXY[0]) * lk, lightXY[1] + (target[1] - lightXY[1]) * lk];

      const preset = MATERIALS[card.material] ?? MATERIALS.paper;
      const foilName = card.foil === 'auto' ? preset.foil : card.foil;
      const foil = FOILS[foilName];
      const lc = hexToLinear(light.color);
      const li = light.intensity;

      frame
        .set('invViewProj', inv)
        .set('cameraPos', eye)
        .set('reliefDepth', card.depth / 44)
        .set('lightPos', [lightXY[0], lightXY[1], view.lift + light.lightHeight])
        .set('lightRadius', light.lightRadius)
        .set('lightColor', [lc[0] * li, lc[1] * li, lc[2] * li])
        .set('envIntensity', light.environment)
        .set('baseAlbedo', hexToLinear(preset.albedo))
        .set('baseRoughness', preset.roughness)
        .set('baseMetallic', preset.metallic)
        .set('foilAlbedo', foil ? hexToLinear(foil.albedo) : [0, 0, 0])
        .set('foilRoughness', foil?.roughness ?? 0.5)
        .set('foilMetallic', foil?.metallic ?? 0)
        .set('foil', foil ? 1 : 0)
        .set('tableAlbedo', hexToLinear(view.table))
        .set('cardHalf', CARD_HALF)
        .set('cardLift', view.lift)
        .set('cardRadius', CARD_RADIUS)
        .set('pixelAngle', (2 * Math.tan(FOV / 2)) / size.height)
        .set('pomSteps', view.pomSteps)
        .set('debugView', view.debug)
        .upload();
      ibl.update();

      const encoder = device.createCommandEncoder({ label: 'embossed-card' });
      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: post.scene(size).view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.setBindGroup(1, ibl.bindGroup);
      pass.draw(3);
      pass.end();
      post.render(encoder, surface.currentTexture().createView(), { time, raw: view.debug !== 0 });
      device.queue.submit([encoder.finish()]);
    },
    dispose() {
      clearTimeout(textTimer);
      head?.destroy();
      sub?.destroy();
      textGen.destroy();
      heightTex.destroy();
      bakeUniforms.destroy();
      frame.destroy();
      ibl.destroy();
      post.destroy();
    },
  };
};

export default create;
