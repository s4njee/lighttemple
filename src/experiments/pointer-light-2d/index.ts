import commonWgsl from '../../shaders/common.wgsl?raw';
import fullscreenWgsl from '../../shaders/fullscreen.wgsl?raw';
import sdfWgsl from '../../shaders/sdf.wgsl?raw';
import { createShader, wgsl } from '../../gpu/shader';
import { HDR_FORMAT, PostProcess } from '../../post/post';
import { UniformBlock } from '../../gpu/uniforms';
import { TextSdfGenerator, type TextSdf } from '../../sdf/text-sdf';
import { hexToLinear } from '../../gpu/color';
import type { ExperimentFactory } from '../types';
import shaderWgsl from './shader.wgsl?raw';

const create: ExperimentFactory = async (ctx) => {
  const { device, surface: canvas, params, pointer, reducedMotion } = ctx;
  // Touch-first devices are usually phones/tablets: start with a lighter budget.
  const coarse = window.matchMedia('(pointer: coarse)').matches;
  const content = params.add(
    {
      scene: { value: 1, options: { Headline: 1, Shapes: 0 } },
      text: { value: 'Light Temple', type: 'text', maxLength: 40, hint: 'Rendered to a distance field at runtime, so any text or font works.' },
      font: { value: 'serif', options: { Serif: 'serif', Sans: 'sans' } },
    },
    'Scene',
  );
  const light = params.add(
    {
      intensity: { value: 1.6, min: 0, max: 6, step: 0.01 },
      color: { value: '#ffd2a1', type: 'color' },
      height: { value: 0.35, min: 0.05, max: 1.5, step: 0.01, hint: 'Height of the light above the floor. Lower = longer shadows.' },
      radius: { value: 0.06, min: 0.002, max: 0.3, step: 0.001, hint: 'Physical size of the light. Bigger = softer penumbrae.' },
      smoothing: { value: 0.08, min: 0, max: 0.4, step: 0.01, hint: 'Pointer follow half-life in seconds.' },
    },
    'Pointer light',
  );
  const fill = params.add(
    {
      fillIntensity: { value: 0.5, min: 0, max: 4, step: 0.01, label: 'Intensity' },
      fillColor: { value: '#6f8cff', type: 'color', label: 'Color' },
      orbit: { value: true, hint: 'Slowly orbit the fill light around the scene.' },
    },
    'Fill light',
  );
  const surface = params.add(
    {
      floor: { value: '#c9c2b8', type: 'color' },
      shapes: { value: '#ece6dc', type: 'color' },
      shapeHeight: { value: 0.12, min: 0, max: 0.6, step: 0.005, hint: 'Extrusion height. Taller shapes cast longer shadows.' },
      bevel: { value: 0.03, min: 0.001, max: 0.15, step: 0.001 },
      gloss: { value: 0.25, min: 0, max: 1, step: 0.01 },
      ambient: { value: 0.05, min: 0, max: 0.4, step: 0.005 },
      animate: { value: !reducedMotion },
    },
    'Surface',
  );
  const debug = params.add(
    {
      steps: { value: coarse ? 32 : 48, min: 8, max: 128, step: 1, hint: 'Max shadow ray-march steps per light (rays also stop once they clear every shape).' },
      view: { value: 0, options: { Lit: 0, 'Shadow mask': 1, 'Distance field': 2, Normals: 3 } },
    },
    'Quality & debug',
  );
  const post = await PostProcess.create(ctx, { bloom: 0.06, vignette: 0.3, grain: 0.04 });

  const uniforms = new UniformBlock(device, 'Params', {
    resolution: 'vec2f',
    time: 'f32',
    sceneTime: 'f32',
    light0: 'vec4f',
    light0Color: 'vec4f',
    light1: 'vec4f',
    light1Color: 'vec4f',
    ambient: 'vec4f',
    floorColor: 'vec4f',
    shapeColor: 'vec4f',
    shapeHeight: 'f32',
    bevel: 'f32',
    steps: 'u32',
    debugView: 'u32',
    sceneMode: 'u32',
    textScale: 'f32',
    textPad: 'f32',
    textRect: 'vec4f',
  });

  const module = await createShader(
    device,
    'pointer-light-2d',
    wgsl(commonWgsl, fullscreenWgsl, sdfWgsl, uniforms.wgslStruct, shaderWgsl),
  );

  const pipeline = await device.createRenderPipelineAsync({
    label: 'pointer-light-2d',
    layout: 'auto',
    vertex: { module, entryPoint: 'vs_fullscreen' },
    fragment: { module, entryPoint: 'fs_main', targets: [{ format: HDR_FORMAT }] },
    primitive: { topology: 'triangle-list' },
  });

  // Headline: text -> SDF texture, regenerated (debounced) when the text or font changes.
  const FONTS = {
    serif: '"Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif',
    sans: 'ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif',
  } as const;
  const textGen = new TextSdfGenerator(device);
  const textSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  let textSdf: TextSdf;
  let bindGroup: GPUBindGroup;
  let textKey = '';
  let textRequest = 0;
  const buildText = async () => {
    const key = `${content.font}|${content.text}`;
    if (key === textKey) return;
    textKey = key;
    const id = ++textRequest;
    const sdf = await textGen.generate({ text: content.text.trim() || ' ', family: FONTS[content.font as keyof typeof FONTS], size: 160 });
    if (id !== textRequest || disposed) return sdf.destroy();
    const old = textSdf;
    textSdf = sdf;
    bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniforms.buffer } },
        { binding: 1, resource: sdf.texture.createView() },
        { binding: 2, resource: textSampler },
      ],
    });
    old?.destroy();
  };
  let disposed = false;
  await buildText();
  let textTimer = 0;
  params.onChange(() => {
    clearTimeout(textTimer);
    textTimer = window.setTimeout(() => void buildText(), 200);
  });

  let sceneTime = 0;
  // Where the light rests when nobody is pointing at the canvas.
  let lightPos: [number, number] = [0.15, 0.1];

  return {
    frame({ time, dt, size }) {
      const aspect = size.width / size.height;
      // Fit the headline texture (glyphs + padding) to ~90% of the width, at most ~0.36 world units tall.
      let textHalf = [0.9 * aspect, (0.9 * aspect * textSdf.height) / textSdf.width];
      if (textHalf[1] > 0.36) textHalf = [(0.36 * textSdf.width) / textSdf.height, 0.36];
      if (surface.animate) sceneTime += dt;

      // Pointer uv -> world (y up, y in [-1,1]). Drift on a slow Lissajous when idle.
      const idle = !pointer.inside || pointer.idle > 4;
      const target: [number, number] = idle
        ? [Math.sin(time * 0.31) * 0.6 * aspect, 0.1 + Math.sin(time * 0.47) * 0.45]
        : [(pointer.uv[0] - 0.5) * 2 * aspect, (0.5 - pointer.uv[1]) * 2];
      const halfLife = idle ? 0.9 : light.smoothing;
      const k = halfLife > 0 ? 1 - Math.pow(2, -dt / halfLife) : 1;
      lightPos = [lightPos[0] + (target[0] - lightPos[0]) * k, lightPos[1] + (target[1] - lightPos[1]) * k];

      const fillAngle = fill.orbit && surface.animate ? sceneTime * 0.25 : 2.2;
      const lc = hexToLinear(light.color);
      const fc = hexToLinear(fill.fillColor);
      // Lights are specified in "lux at 1 unit"; scale so the defaults read well at typical heights.
      const li = light.intensity * 0.6;
      const fi = fill.fillIntensity * 0.6;

      uniforms
        .set('resolution', [size.width, size.height])
        .set('time', time)
        .set('sceneTime', sceneTime)
        .set('light0', [lightPos[0], lightPos[1], light.height, light.radius])
        .set('light0Color', [lc[0] * li, lc[1] * li, lc[2] * li, 0])
        .set('light1', [Math.cos(fillAngle) * 1.1 * aspect, Math.sin(fillAngle) * 0.9, 0.6, 0.12])
        .set('light1Color', [fc[0] * fi, fc[1] * fi, fc[2] * fi, 0])
        .set('ambient', [...hexToLinear(surface.floor).map((c) => c * surface.ambient), 0])
        .set('floorColor', [...hexToLinear(surface.floor), 0])
        .set('shapeColor', [...hexToLinear(surface.shapes), surface.gloss])
        .set('shapeHeight', surface.shapeHeight)
        .set('bevel', surface.bevel)
        .set('sceneMode', content.scene)
        .set('textScale', (2 * textHalf[0]) / textSdf.width)
        .set('textRect', [0, 0.02, textHalf[0], textHalf[1]])
        .set('textPad', (textSdf.padding * 2 * textHalf[0]) / textSdf.width)
        .set('steps', debug.steps)
        .set('debugView', debug.view)
        .upload();

      const encoder = device.createCommandEncoder({ label: 'pointer-light-2d' });
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          { view: post.scene(size).view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] },
        ],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(3);
      pass.end();
      post.render(encoder, canvas.currentTexture().createView(), { time, raw: debug.view !== 0 });
      device.queue.submit([encoder.finish()]);
    },
    dispose() {
      disposed = true;
      clearTimeout(textTimer);
      textSdf.destroy();
      textGen.destroy();
      post.destroy();
      uniforms.destroy();
    },
  };
};

export default create;
