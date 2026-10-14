import commonWgsl from '../../shaders/common.wgsl?raw';
import fullscreenWgsl from '../../shaders/fullscreen.wgsl?raw';
import sdfWgsl from '../../shaders/sdf.wgsl?raw';
import { generateMipmaps, mipLevelCount } from '../../gpu/mipmaps';
import { createShader, wgsl } from '../../gpu/shader';
import { UniformBlock } from '../../gpu/uniforms';
import { HDR_FORMAT, PostProcess } from '../../post/post';
import { TextSdfGenerator, type TextSdf } from '../../sdf/text-sdf';
import { hexToLinear } from '../../gpu/color';
import type { ExperimentFactory } from '../types';
import shaderWgsl from './shader.wgsl?raw';

/** Panel layout in fractions of the canvas: centre x/y, half width/height (of the shorter side). */
interface Panel {
  cx: number;
  cy: number;
  hw: number;
  hh: number;
  radius: number;
}

const DEFAULT_PANELS: Panel[] = [
  { cx: 0.42, cy: 0.5, hw: 0.36, hh: 0.27, radius: 0.06 }, // main card (front)
  { cx: 0.74, cy: 0.3, hw: 0.17, hh: 0.12, radius: 0.05 }, // small widget
  { cx: 0.73, cy: 0.74, hw: 0.2, hh: 0.065, radius: 0.065 }, // pill button (back)
];

const create: ExperimentFactory = async (ctx) => {
  const { device, surface, params, pointer, reducedMotion } = ctx;

  const glass = params.add(
    {
      frost: { value: 3.2, min: 0, max: 6, step: 0.05, hint: 'Blur, as a mip level of the background (each step doubles the radius).' },
      refraction: { value: 26, min: 0, max: 120, step: 1, label: 'Refraction (px)' },
      bevel: { value: 26, min: 2, max: 80, step: 1, label: 'Bevel (px)' },
      dispersion: { value: 0.18, min: 0, max: 0.6, step: 0.01, hint: 'Per-channel refraction difference: colour fringes at the edges.' },
      tint: { value: '#e8efff', type: 'color' },
      lift: { value: 0.35, min: 0, max: 1, step: 0.01, hint: 'How much the frosted glass brightens what is behind it.' },
      rim: { value: 1, min: 0, max: 3, step: 0.01, label: 'Rim light' },
    },
    'Glass',
  );
  const scene = params.add(
    {
      headline: { value: 'glass', type: 'text', maxLength: 16 },
      lightHeight: { value: 260, min: 20, max: 1200, step: 5, label: 'Light height (px)' },
      lightColor: { value: '#fff3e6', type: 'color', label: 'Light colour' },
      shadowStrength: { value: 0.5, min: 0, max: 1, step: 0.01, label: 'Shadow' },
      animate: { value: !reducedMotion },
    },
    'Scene',
  );
  params.button('Reset panel positions', () => {
    panels = DEFAULT_PANELS.map((p) => ({ ...p }));
  });
  const post = await PostProcess.create(ctx, { bloom: 0.08, vignette: 0.3, grain: 0.03 });

  const textGen = new TextSdfGenerator(device);
  const FONT = '"Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif';
  let text: TextSdf = await textGen.generate({ text: scene.headline, family: FONT, weight: 700, size: 260 });

  const u = new UniformBlock(device, 'Glass', {
    panel0: 'vec4f',
    panel1: 'vec4f',
    panel2: 'vec4f',
    radii: 'vec4f',
    tint: 'vec4f',
    textRect: 'vec4f',
    light: 'vec3f',
    textScale: 'f32',
    lightColor: 'vec3f',
    time: 'f32',
    resolution: 'vec2f',
    frost: 'f32',
    bevel: 'f32',
    refraction: 'f32',
    dispersion: 'f32',
    rim: 'f32',
    shadowOffset: 'f32',
    shadowBlur: 'f32',
    shadowStrength: 'f32',
  });
  const module = await createShader(device, 'glass-panels', wgsl(commonWgsl, sdfWgsl, fullscreenWgsl, u.wgslStruct, shaderWgsl));
  const [bgPipeline, glassPipeline] = await Promise.all([
    device.createRenderPipelineAsync({
      label: 'glass:background',
      layout: 'auto',
      vertex: { module, entryPoint: 'vs_fullscreen' },
      fragment: { module, entryPoint: 'fs_background', targets: [{ format: HDR_FORMAT }] },
    }),
    device.createRenderPipelineAsync({
      label: 'glass:composite',
      layout: 'auto',
      vertex: { module, entryPoint: 'vs_fullscreen' },
      fragment: { module, entryPoint: 'fs_glass', targets: [{ format: HDR_FORMAT }] },
    }),
  ]);
  const linear = device.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', addressModeU: 'mirror-repeat', addressModeV: 'mirror-repeat' });

  let bgTex: GPUTexture | null = null;
  let bgGroup: GPUBindGroup | null = null;
  let glassGroup: GPUBindGroup | null = null;
  const rebuildGroups = () => {
    bgGroup = device.createBindGroup({
      layout: bgPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: u.buffer } },
        { binding: 1, resource: text.texture.createView() },
        { binding: 2, resource: linear },
      ],
    });
    if (bgTex) {
      glassGroup = device.createBindGroup({
        layout: glassPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: u.buffer } },
          { binding: 2, resource: linear },
          { binding: 3, resource: bgTex.createView() },
        ],
      });
    }
  };
  const ensureBg = (w: number, h: number) => {
    if (bgTex?.width === w && bgTex.height === h) return;
    bgTex?.destroy();
    bgTex = device.createTexture({
      label: 'glass-background',
      size: [w, h],
      format: HDR_FORMAT,
      mipLevelCount: Math.min(9, mipLevelCount(w, h)),
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    rebuildGroups();
  };
  rebuildGroups();

  let headlineKey = scene.headline;
  let textTimer = 0;
  params.onChange(() => {
    clearTimeout(textTimer);
    textTimer = window.setTimeout(async () => {
      if (scene.headline === headlineKey) return;
      headlineKey = scene.headline;
      const t = await textGen.generate({ text: scene.headline.trim() || ' ', family: FONT, weight: 700, size: 260 });
      text.destroy();
      text = t;
      rebuildGroups();
    }, 200);
  });

  // Dragging: grab the front-most panel under the pointer.
  let panels = DEFAULT_PANELS.map((p) => ({ ...p }));
  let drag: { index: number; dx: number; dy: number } | null = null;
  let light: [number, number] = [0.5, 0.3];
  let animTime = 0;

  const pxPanel = (p: Panel, w: number, h: number) => {
    const s = Math.min(w, h);
    return [p.cx * w, p.cy * h, p.hw * s, p.hh * s, p.radius * s] as const;
  };

  return {
    frame({ time, dt, size }) {
      const { width: w, height: h } = size;
      if (scene.animate) animTime += dt;
      ensureBg(w, h);

      const [pu, pv] = pointer.uv;
      if (pointer.down) {
        if (!drag) {
          const hit = panels.findIndex((p) => {
            const [cx, cy, hw, hh] = pxPanel(p, w, h);
            return Math.abs(pu * w - cx) < hw && Math.abs(pv * h - cy) < hh;
          });
          if (hit >= 0) drag = { index: hit, dx: panels[hit].cx - pu, dy: panels[hit].cy - pv };
        }
        if (drag) {
          panels[drag.index].cx = Math.min(1, Math.max(0, pu + drag.dx));
          panels[drag.index].cy = Math.min(1, Math.max(0, pv + drag.dy));
        }
      } else {
        drag = null;
      }

      // Light follows the pointer; drifts when idle.
      const idle = !pointer.inside || pointer.idle > 4;
      const target: [number, number] = idle ? [0.5 + 0.35 * Math.sin(time * 0.3), 0.35 + 0.25 * Math.sin(time * 0.41)] : [pu, pv];
      const k = 1 - Math.pow(2, -dt / (idle ? 0.8 : 0.06));
      light = [light[0] + (target[0] - light[0]) * k, light[1] + (target[1] - light[1]) * k];

      const px = panels.map((p) => pxPanel(p, w, h));
      const s = Math.min(w, h) / 900; // scale px-based params with the canvas
      const glyphW = text.width - 2 * text.padding;
      const scale = (0.9 * w) / Math.max(glyphW, 1); // canvas px per text texel
      const tint = hexToLinear(glass.tint);
      const lc = hexToLinear(scene.lightColor);

      u.set('panel0', px[0].slice(0, 4))
        .set('panel1', px[1].slice(0, 4))
        .set('panel2', px[2].slice(0, 4))
        .set('radii', [px[0][4], px[1][4], px[2][4], 0])
        .set('tint', [...tint, glass.lift])
        .set('textRect', [w * 0.5, h * 0.5, (text.width * scale) / 2, (text.height * scale) / 2])
        .set('textScale', scale)
        .set('light', [light[0] * w, light[1] * h, scene.lightHeight * s])
        .set('lightColor', lc)
        .set('time', animTime)
        .set('resolution', [w, h])
        .set('frost', glass.frost)
        .set('bevel', glass.bevel * s)
        .set('refraction', glass.refraction * s)
        .set('dispersion', glass.dispersion)
        .set('rim', glass.rim)
        .set('shadowOffset', 22 * s)
        .set('shadowBlur', 40 * s)
        .set('shadowStrength', scene.shadowStrength)
        .upload();

      const encoder = device.createCommandEncoder({ label: 'glass-panels' });
      const bgPass = encoder.beginRenderPass({
        colorAttachments: [{ view: bgTex!.createView({ baseMipLevel: 0, mipLevelCount: 1 }), loadOp: 'clear', storeOp: 'store' }],
      });
      bgPass.setPipeline(bgPipeline);
      bgPass.setBindGroup(0, bgGroup!);
      bgPass.draw(3);
      bgPass.end();
      generateMipmaps(device, bgTex!, encoder);

      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: post.scene(size).view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
      });
      pass.setPipeline(glassPipeline);
      pass.setBindGroup(0, glassGroup!);
      pass.draw(3);
      pass.end();
      post.render(encoder, surface.currentTexture().createView(), { time });
      device.queue.submit([encoder.finish()]);
    },
    dispose() {
      clearTimeout(textTimer);
      text.destroy();
      textGen.destroy();
      bgTex?.destroy();
      u.destroy();
      post.destroy();
    },
  };
};

export default create;
