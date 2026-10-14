import commonWgsl from '../../shaders/common.wgsl?raw';
import fullscreenWgsl from '../../shaders/fullscreen.wgsl?raw';
import noiseWgsl from '../../shaders/noise.wgsl?raw';
import sdfWgsl from '../../shaders/sdf.wgsl?raw';
import { generateMipmaps } from '../../gpu/mipmaps';
import { createShader, wgsl } from '../../gpu/shader';
import { UniformBlock } from '../../gpu/uniforms';
import { HDR_FORMAT, PostProcess } from '../../post/post';
import { TextSdfGenerator, type TextSdf } from '../../sdf/text-sdf';
import { hexToLinear } from '../../gpu/color';
import type { ExperimentFactory } from '../types';
import shaderWgsl from './shader.wgsl?raw';

const MAX_LETTERS = 32;
const FONTS = {
  script: '"Snell Roundhand", "Segoe Script", "Brush Script MT", "Lucida Handwriting", cursive',
  sans: '"Avenir Next", "Helvetica Neue", Arial, ui-sans-serif, sans-serif',
} as const;

const create: ExperimentFactory = async (ctx) => {
  const { device, surface, params, reducedMotion } = ctx;

  const sign = params.add(
    {
      text: { value: 'open late', type: 'text', maxLength: MAX_LETTERS },
      font: { value: 'script', options: { Script: 'script', Sans: 'sans' } },
      style: { value: 0, options: { 'Outline tube': 0, 'Solid letters': 1 } },
      colorText: { value: '#ff3d8b', type: 'color', label: 'Text colour' },
      colorFrame: { value: '#38d6ff', type: 'color', label: 'Frame colour' },
      colorIcon: { value: '#ffd23d', type: 'color', label: 'Moon colour' },
      tube: { value: 3.2, min: 1, max: 10, step: 0.1, label: 'Tube (px)' },
      power: { value: 4, min: 0, max: 16, step: 0.1 },
    },
    'Sign',
  );
  const env = params.add(
    {
      flicker: { value: reducedMotion ? 0 : 0.6, min: 0, max: 1, step: 0.01, hint: 'Random dropouts and hum. One letter is "faulty" and misbehaves more.' },
      spill: { value: 0.22, min: 0, max: 1, step: 0.01, hint: 'How much the tubes light the wall.' },
      wall: { value: 0, options: { Brick: 0, Concrete: 1 } },
      bump: { value: 6, min: 0, max: 20, step: 0.1 },
      ambient: { value: 0.02, min: 0, max: 0.3, step: 0.005 },
    },
    'Street',
  );
  const post = await PostProcess.create(ctx, { bloom: 0.12, bloomRadius: 1.3, vignette: 0.45, grain: 0.035 });

  const textGen = new TextSdfGenerator(device);
  const makeText = () =>
    textGen.generate({ text: sign.text.trim() || ' ', family: FONTS[sign.font as keyof typeof FONTS] ?? FONTS.script, weight: sign.font === 'script' ? 400 : 300, size: 200 });
  let text: TextSdf = await makeText();

  const u = new UniformBlock(device, 'Neon', {
    textRect: 'vec4f',
    frameRect: 'vec4f',
    iconPos: 'vec4f',
    colorText: 'vec3f',
    textScale: 'f32',
    colorFrame: 'vec3f',
    tube: 'f32',
    colorIcon: 'vec3f',
    power: 'f32',
    resolution: 'vec2f',
    time: 'f32',
    flicker: 'f32',
    spill: 'f32',
    bump: 'f32',
    ambient: 'f32',
    emissionScale: 'f32',
    style: 'u32',
    wallType: 'u32',
    letterCount: 'u32',
  });
  const letterBuffer = device.createBuffer({ size: MAX_LETTERS * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const letterData = new Float32Array(MAX_LETTERS * 4);

  const module = await createShader(device, 'neon-sign', wgsl(commonWgsl, noiseWgsl, sdfWgsl, fullscreenWgsl, u.wgslStruct, shaderWgsl));
  const g0 = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: {} },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    ],
  });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [g0] });
  const [emissionPipeline, compositePipeline] = await Promise.all([
    device.createRenderPipelineAsync({
      label: 'neon:emission',
      layout,
      vertex: { module, entryPoint: 'vs_fullscreen' },
      fragment: { module, entryPoint: 'fs_emission', targets: [{ format: HDR_FORMAT }] },
    }),
    device.createRenderPipelineAsync({
      label: 'neon:composite',
      layout,
      vertex: { module, entryPoint: 'vs_fullscreen' },
      fragment: { module, entryPoint: 'fs_composite', targets: [{ format: HDR_FORMAT }] },
    }),
  ]);
  const linear = device.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear' });
  // A 1×1 placeholder bound while the emission pass writes the real texture
  // (a texture can't be both the render target and a sampled input).
  const dummy = device.createTexture({ size: [1, 1], format: HDR_FORMAT, usage: GPUTextureUsage.TEXTURE_BINDING });

  let emission: GPUTexture | null = null;
  let groups: { emission: GPUBindGroup; composite: GPUBindGroup } | null = null;
  const rebuild = () => {
    if (!emission) return;
    const make = (emissionView: GPUTextureView) =>
      device.createBindGroup({
        layout: g0,
        entries: [
          { binding: 0, resource: { buffer: u.buffer } },
          { binding: 1, resource: text.texture.createView() },
          { binding: 2, resource: linear },
          { binding: 3, resource: emissionView },
          { binding: 4, resource: { buffer: letterBuffer } },
        ],
      });
    groups = { emission: make(dummy.createView()), composite: make(emission.createView()) };
  };

  let textKey = `${sign.text}|${sign.font}`;
  let timer = 0;
  params.onChange(() => {
    clearTimeout(timer);
    timer = window.setTimeout(async () => {
      const key = `${sign.text}|${sign.font}`;
      if (key === textKey) return;
      textKey = key;
      const t = await makeText();
      text.destroy();
      text = t;
      rebuild();
    }, 200);
  });

  return {
    frame({ time, size }) {
      const { width: w, height: h } = size;
      const ew = Math.max(1, Math.ceil(w / 2));
      const eh = Math.max(1, Math.ceil(h / 2));
      if (emission?.width !== ew || emission.height !== eh) {
        emission?.destroy();
        emission = device.createTexture({
          label: 'neon-emission',
          size: [ew, eh],
          format: HDR_FORMAT,
          mipLevelCount: 8,
          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        rebuild();
      }

      // Layout: text centred, frame around it, moon at the frame's top-right corner.
      const glyphW = text.width - 2 * text.padding;
      const glyphH = text.height - 2 * text.padding;
      const scale = Math.min((0.62 * w) / Math.max(glyphW, 1), (0.3 * h) / Math.max(glyphH, 1)); // px per texel
      const cx = w * 0.5;
      const cy = h * 0.5;
      const frameHalf = [(glyphW * scale) / 2 + 0.07 * h, (glyphH * scale) / 2 + 0.07 * h];
      const tube = sign.tube * Math.max(1, h / 800);

      // Letter extents in px + flicker phase; one deterministic "faulty" letter.
      const letters = text.letters.slice(0, MAX_LETTERS);
      const faulty = letters.length > 2 ? Math.floor(letters.length * 0.6) : -1;
      letters.forEach((l, i) => {
        const x0 = cx + (l.start - text.width / 2) * scale;
        const x1 = cx + (l.end - text.width / 2) * scale;
        letterData.set([x0, x1, (0.13 + i * 0.618) % 1, i === faulty ? 1 : 0], i * 4);
      });
      device.queue.writeBuffer(letterBuffer, 0, letterData);

      u.set('textRect', [cx, cy, (text.width * scale) / 2, (text.height * scale) / 2])
        .set('textScale', scale)
        .set('frameRect', [cx, cy, frameHalf[0], frameHalf[1]])
        .set('iconPos', [cx + frameHalf[0] - 0.02 * h, cy - frameHalf[1] - 0.005 * h, 0.05 * h, 0])
        .set('colorText', hexToLinear(sign.colorText))
        .set('colorFrame', hexToLinear(sign.colorFrame))
        .set('colorIcon', hexToLinear(sign.colorIcon))
        .set('tube', tube)
        .set('power', sign.power)
        .set('resolution', [w, h])
        .set('time', time)
        .set('flicker', env.flicker)
        .set('spill', env.spill)
        .set('bump', env.bump)
        .set('ambient', env.ambient)
        .set('emissionScale', w / ew)
        .set('style', sign.style)
        .set('wallType', env.wall)
        .set('letterCount', letters.length)
        .upload();

      const encoder = device.createCommandEncoder({ label: 'neon-sign' });
      const ep = encoder.beginRenderPass({
        colorAttachments: [{ view: emission!.createView({ baseMipLevel: 0, mipLevelCount: 1 }), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
      });
      ep.setPipeline(emissionPipeline);
      ep.setBindGroup(0, groups!.emission);
      ep.draw(3);
      ep.end();
      generateMipmaps(device, emission!, encoder);

      const cp = encoder.beginRenderPass({
        colorAttachments: [{ view: post.scene(size).view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
      });
      cp.setPipeline(compositePipeline);
      cp.setBindGroup(0, groups!.composite);
      cp.draw(3);
      cp.end();
      post.render(encoder, surface.currentTexture().createView(), { time });
      device.queue.submit([encoder.finish()]);
    },
    dispose() {
      clearTimeout(timer);
      text.destroy();
      textGen.destroy();
      emission?.destroy();
      dummy.destroy();
      letterBuffer.destroy();
      u.destroy();
      post.destroy();
    },
  };
};

export default create;
