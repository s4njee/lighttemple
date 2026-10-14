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
import shaderWgsl from './shader.wgsl?raw';

const MAX_LETTERS = 48;
/** Texels per CSS pixel in the text SDF (2 = crisp at DPR 2). */
const OVERSAMPLE = 2;

const create: ExperimentFactory = async (ctx) => {
  const { device, surface, params, reducedMotion } = ctx;
  const canvas = surface.canvas;
  const stage = canvas.parentElement!;

  const type = params.add(
    {
      material: { value: 0, options: { 'Alternating metals': 0, 'Glossy colours': 1, 'Brushed gold': 2 } },
      depth: { value: 0.18, min: 0.02, max: 0.6, step: 0.01, label: 'Extrusion', hint: 'Relative to the font size.' },
      bevel: { value: 0.025, min: 0, max: 0.08, step: 0.001, hint: 'Edge rounding, relative to the font size.' },
      tilt: { value: 0.35, min: 0, max: 1.2, step: 0.01, hint: 'How far the pointer skews the oblique projection (front faces stay put).' },
    },
    'Type',
  );
  const light = params.add(
    {
      lightHeight: { value: 0.45, min: 0.05, max: 2, step: 0.01, label: 'Light height', hint: 'Above the page, in canvas heights.' },
      intensity: { value: 2.4, min: 0, max: 10, step: 0.05 },
      color: { value: '#fff0dc', type: 'color' },
      softness: { value: 0.06, min: 0.005, max: 0.3, step: 0.005, hint: 'Light size: wider penumbrae.' },
      environment: { value: 0.5, min: 0, max: 2, step: 0.01 },
      wall: { value: '#d8d0c4', type: 'color' },
      view: { value: 0, options: { Lit: 0, Normals: 1, 'Shadow × AO': 2 } },
      showDom: { value: false, label: 'Show DOM text', hint: 'Reveal the real (normally transparent) heading to check alignment.' },
    },
    'Light',
  );
  const post = await PostProcess.create(ctx, { bloom: 0.04, vignette: 0.3, grain: 0.02, exposure: 1.05 });

  // --- The real, selectable, editable heading (ink is transparent) ---------------
  const heading = document.createElement('h1');
  heading.className = 'lit-type-dom';
  heading.contentEditable = 'plaintext-only';
  if (heading.contentEditable !== 'plaintext-only') heading.contentEditable = 'true';
  heading.spellcheck = false;
  heading.textContent = 'Light Temple';
  heading.setAttribute('aria-label', 'Editable headline, rendered as lit 3D letters');
  heading.addEventListener('keydown', (e) => e.key === 'Enter' && e.preventDefault());
  stage.append(heading);

  // Pointer over the whole stage (the heading sits above the canvas).
  let pointer: { x: number; y: number; inside: boolean; idle: number } = { x: 0.5, y: 0.3, inside: false, idle: Infinity };
  const onMove = (e: PointerEvent) => {
    const r = canvas.getBoundingClientRect();
    pointer = { x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height, inside: true, idle: 0 };
  };
  const onLeave = () => (pointer.inside = false);
  stage.addEventListener('pointermove', onMove);
  stage.addEventListener('pointerleave', onLeave);

  const ibl = await Ibl.create(device);
  await ibl.setProcedural({ kind: 'studio' });
  const textGen = new TextSdfGenerator(device);

  const u = new UniformBlock(device, 'LitType', {
    textRect: 'vec4f',
    light: 'vec3f',
    textScale: 'f32',
    lightColor: 'vec3f',
    depth: 'f32',
    wallColor: 'vec3f',
    bevel: 'f32',
    shift: 'vec2f',
    lightRadius: 'f32',
    lightRange: 'f32',
    envIntensity: 'f32',
    materialMode: 'u32',
    letterCount: 'u32',
    debugView: 'u32',
  });
  const letterBuffer = device.createBuffer({ size: MAX_LETTERS * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const letterData = new Float32Array(MAX_LETTERS * 4);

  const module = await createShader(
    device,
    'lit-type',
    wgsl(commonWgsl, noiseWgsl, sdfWgsl, brdfWgsl, fullscreenWgsl, u.wgslStruct, ibl.wgslStruct, iblWgsl, shaderWgsl),
  );
  const g0 = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    ],
  });
  const pipeline = await device.createRenderPipelineAsync({
    label: 'lit-type',
    layout: device.createPipelineLayout({ bindGroupLayouts: [g0, ibl.bindGroupLayout] }),
    vertex: { module, entryPoint: 'vs_fullscreen' },
    fragment: { module, entryPoint: 'fs_main', targets: [{ format: HDR_FORMAT }] },
  });
  const linear = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

  // --- Text SDF from the heading's own computed font --------------------------------
  let text: TextSdf | null = null;
  let group: GPUBindGroup | null = null;
  let textKey = '';
  let cssFontSize = 0;
  let disposed = false;
  const syncText = async () => {
    const style = getComputedStyle(heading);
    const size = parseFloat(style.fontSize);
    const content = (heading.textContent ?? '').replace(/\s+/g, ' ').trim() || ' ';
    const key = `${content}|${style.fontFamily}|${style.fontWeight}|${size}`;
    if (key === textKey) return;
    textKey = key;
    const t = await textGen.generate({
      text: content,
      family: style.fontFamily,
      weight: style.fontWeight,
      size: Math.min(320, size * OVERSAMPLE),
    });
    if (disposed || key !== textKey) return t.destroy();
    text?.destroy();
    text = t;
    cssFontSize = size;
    group = device.createBindGroup({
      layout: g0,
      entries: [
        { binding: 0, resource: { buffer: u.buffer } },
        { binding: 1, resource: t.texture.createView() },
        { binding: 2, resource: linear },
        { binding: 3, resource: { buffer: letterBuffer } },
      ],
    });
  };
  await syncText();
  let timer = 0;
  const scheduleSync = () => {
    clearTimeout(timer);
    timer = window.setTimeout(() => void syncText(), 120);
  };
  heading.addEventListener('input', scheduleSync);
  const ro = new ResizeObserver(scheduleSync);
  ro.observe(heading);

  let lightXY: [number, number] = [0.5, 0.3];
  let shift: [number, number] = [0, 0];
  const range = document.createRange();

  return {
    frame({ time, dt, size }) {
      if (!text || !group) return;
      pointer.idle += dt;
      heading.classList.toggle('reveal', light.showDom);
      const { width: w, height: h } = size;
      const cr = canvas.getBoundingClientRect();
      const pxPerCss = w / cr.width;

      // Where does the DOM put the text? Range rects are the content area: top = baseline − ascent.
      range.selectNodeContents(heading);
      const rr = range.getBoundingClientRect();
      const k = text.size / cssFontSize; // texels per CSS px
      const penX = (rr.left - cr.left) * pxPerCss;
      const baseline = (rr.top - cr.top + text.fontAscent / k) * pxPerCss;
      const s = pxPerCss / k; // canvas px per texel
      const cx = penX + (text.width / 2 - text.originX) * s;
      const cy = baseline + (text.height / 2 - text.baseline) * s;
      const fontPx = cssFontSize * pxPerCss;

      // Measure where the DOM actually put each visible character, and store a
      // per-letter shift from the SDF's layout to the DOM's (rounding, kerning).
      const node = heading.firstChild;
      const domChars: DOMRect[] = [];
      if (node && node.nodeType === Node.TEXT_NODE) {
        const str = node.textContent ?? '';
        for (let i = 0; i < str.length && domChars.length < MAX_LETTERS; i++) {
          if (!str[i].trim()) continue;
          range.setStart(node, i);
          range.setEnd(node, i + 1);
          domChars.push(range.getBoundingClientRect());
        }
      }
      const letters = text.letters.slice(0, MAX_LETTERS);
      letters.forEach((l, i) => {
        const texX0 = penX + (l.start - text!.originX) * s;
        const dom = domChars[i];
        const x0 = dom ? (dom.left - cr.left) * pxPerCss : texX0;
        const x1 = dom ? (dom.right - cr.left) * pxPerCss : penX + (l.end - text!.originX) * s;
        letterData.set([x0, x1, x0 - texX0, (i * 0.13 + 0.02) % 1], i * 4);
      });
      device.queue.writeBuffer(letterBuffer, 0, letterData);

      // Light follows the pointer; tilt skews away from it.
      const idle = !pointer.inside || pointer.idle > 4;
      const target: [number, number] = idle
        ? [0.5 + 0.32 * Math.sin(time * 0.33), 0.35 + 0.2 * Math.sin(time * 0.47)]
        : [pointer.x, pointer.y];
      const lk = 1 - Math.pow(2, -dt / (idle ? 0.9 : 0.08));
      lightXY = [lightXY[0] + (target[0] - lightXY[0]) * lk, lightXY[1] + (target[1] - lightXY[1]) * lk];
      const tilt = reducedMotion ? type.tilt * 0.3 : type.tilt;
      const shiftTarget: [number, number] = [(0.5 - lightXY[0]) * tilt, (0.5 - lightXY[1]) * tilt];
      const sk = 1 - Math.pow(2, -dt / 0.15);
      shift = [shift[0] + (shiftTarget[0] - shift[0]) * sk, shift[1] + (shiftTarget[1] - shift[1]) * sk];

      const lc = hexToLinear(light.color);
      const li = light.intensity;
      u.set('textRect', [cx, cy, (text.width * s) / 2, (text.height * s) / 2])
        .set('textScale', s)
        .set('light', [lightXY[0] * w, lightXY[1] * h, light.lightHeight * h])
        .set('lightColor', [lc[0] * li, lc[1] * li, lc[2] * li])
        .set('lightRadius', light.softness)
        .set('lightRange', h * 0.9)
        .set('depth', type.depth * fontPx)
        .set('bevel', type.bevel * fontPx)
        .set('shift', shift)
        .set('wallColor', hexToLinear(light.wall))
        .set('envIntensity', light.environment)
        .set('materialMode', type.material)
        .set('letterCount', letters.length)
        .set('debugView', light.view)
        .upload();
      ibl.update();

      const encoder = device.createCommandEncoder({ label: 'lit-type' });
      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: post.scene(size).view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.setBindGroup(1, ibl.bindGroup);
      pass.draw(3);
      pass.end();
      post.render(encoder, surface.currentTexture().createView(), { time, raw: light.view !== 0 });
      device.queue.submit([encoder.finish()]);
    },
    dispose() {
      disposed = true;
      clearTimeout(timer);
      ro.disconnect();
      heading.remove();
      stage.removeEventListener('pointermove', onMove);
      stage.removeEventListener('pointerleave', onLeave);
      text?.destroy();
      textGen.destroy();
      letterBuffer.destroy();
      u.destroy();
      ibl.destroy();
      post.destroy();
    },
  };
};

export default create;
