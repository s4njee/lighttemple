import { mat4, vec3 } from 'wgpu-matrix';
import brdfWgsl from '../../shaders/brdf.wgsl?raw';
import commonWgsl from '../../shaders/common.wgsl?raw';
import fullscreenWgsl from '../../shaders/fullscreen.wgsl?raw';
import noiseWgsl from '../../shaders/noise.wgsl?raw';
import opticsWgsl from '../../shaders/optics.wgsl?raw';
import sdfWgsl from '../../shaders/sdf.wgsl?raw';
import { createShader, wgsl } from '../../gpu/shader';
import { UniformBlock } from '../../gpu/uniforms';
import { Ibl, iblWgsl } from '../../ibl/ibl';
import { HDR_FORMAT, PostProcess } from '../../post/post';
import { TextSdfGenerator, type TextSdf } from '../../sdf/text-sdf';
import { hexToLinear } from '../../gpu/color';
import type { ExperimentFactory } from '../types';
import shaderWgsl from './shader.wgsl?raw';

/** Trading card proportions (63 × 88 mm). */
const CARD_HALF: [number, number] = [0.63, 0.88];
const FOV = (30 * Math.PI) / 180;

/** Critically-damped-ish spring for a satisfying tilt that slightly overshoots. */
class Spring {
  x = 0;
  v = 0;
  constructor(
    public stiffness = 120,
    public damping = 14,
  ) {}
  step(target: number, dt: number): number {
    // Semi-implicit Euler in small sub-steps for stability at low frame rates.
    const n = Math.max(1, Math.ceil(dt / (1 / 240)));
    const h = dt / n;
    for (let i = 0; i < n; i++) {
      this.v += (this.stiffness * (target - this.x) - this.damping * this.v) * h;
      this.x += this.v * h;
    }
    return this.x;
  }
}

const create: ExperimentFactory = async (ctx) => {
  const { device, surface, params, pointer, reducedMotion } = ctx;

  const holo = params.add(
    {
      title: { value: 'LIGHT TEMPLE', type: 'text', maxLength: 24 },
      holoMode: { value: 1, options: { 'Frame & title': 0, 'Reverse holo': 1, 'Full art': 2 }, label: 'Foil layout' },
      holoIntensity: { value: 1.2, min: 0, max: 4, step: 0.01, label: 'Rainbow', hint: 'Diffraction-grating strength.' },
      grooveSpacing: { value: 1500, min: 600, max: 4000, step: 10, label: 'Groove spacing (nm)', hint: 'Finer grooves spread the rainbow over wider angles.' },
      glitter: { value: 0.18, min: 0, max: 1, step: 0.01, hint: 'Fraction of tiny cells holding a randomly oriented glitter fleck.' },
      filmThickness: { value: 420, min: 100, max: 1200, step: 5, label: 'Film (nm)', hint: 'Thin-film thickness: shifts the oil-slick colours.' },
      filmStrength: { value: 0.55, min: 0, max: 1, step: 0.01, label: 'Film strength' },
      anisotropy: { value: 0.75, min: -0.95, max: 0.95, step: 0.01, hint: 'Radially brushed foil: highlights stretch into streaks.' },
      foilRoughness: { value: 0.22, min: 0.03, max: 0.8, step: 0.01, label: 'Foil roughness' },
      inkCover: { value: 0.55, min: 0, max: 1, step: 0.01, label: 'Ink over foil' },
    },
    'Foil',
  );
  const motion = params.add(
    {
      maxTilt: { value: 22, min: 0, max: 45, step: 1, label: 'Max tilt (°)' },
      springiness: { value: 0.5, min: 0, max: 1, step: 0.01, hint: '0 = heavy and damped, 1 = bouncy.' },
      lightAngle: { value: 35, min: -90, max: 90, step: 1, label: 'Light angle (°)' },
      intensity: { value: 9, min: 0, max: 40, step: 0.1, label: 'Light intensity' },
      environment: { value: 0.6, min: 0, max: 2, step: 0.01 },
    },
    'Motion & light',
  );
  const post = await PostProcess.create(ctx, { bloom: 0.06, vignette: 0.35, grain: 0.03 });

  // Device tilt (phones). iOS needs a permission prompt triggered by a user gesture.
  let gyro: { beta: number; gamma: number } | null = null;
  const onOrientation = (e: DeviceOrientationEvent) => {
    if (e.beta === null || e.gamma === null) return;
    gyro = { beta: e.beta, gamma: e.gamma };
  };
  let gyroBase: { beta: number; gamma: number } | null = null;
  params.button('Use device tilt (gyro)', async () => {
    const DOE = DeviceOrientationEvent as unknown as { requestPermission?: () => Promise<string> };
    try {
      if (DOE.requestPermission && (await DOE.requestPermission()) !== 'granted') return;
      window.addEventListener('deviceorientation', onOrientation);
      gyroBase = null;
    } catch (err) {
      console.warn('Device orientation unavailable', err);
    }
  });

  const ibl = await Ibl.create(device);
  await ibl.setProcedural({ kind: 'studio' });

  const textGen = new TextSdfGenerator(device);
  let title: TextSdf = await textGen.generate({ text: holo.title, family: '"Helvetica Neue", Arial, ui-sans-serif, sans-serif', weight: 800, size: 140 });

  const frame = new UniformBlock(device, 'Frame', {
    invViewProj: 'mat4x4f',
    cardX: 'vec4f',
    cardY: 'vec4f',
    cardZ: 'vec4f',
    titleRect: 'vec4f',
    cameraPos: 'vec3f',
    envIntensity: 'f32',
    lightPos: 'vec3f',
    lightRadius: 'f32',
    lightColor: 'vec3f',
    pixelAngle: 'f32',
    glowColor: 'vec3f',
    titleScale: 'f32',
    cardHalf: 'vec2f',
    cardRadius: 'f32',
    backdropZ: 'f32',
    holoMode: 'u32',
    glitterScale: 'f32',
    glitter: 'f32',
    filmThickness: 'f32',
    filmStrength: 'f32',
    foilRoughness: 'f32',
    anisotropy: 'f32',
    grooveSpacing: 'f32',
    holoIntensity: 'f32',
    inkCover: 'f32',
    foilAmount: 'f32',
  });
  const module = await createShader(
    device,
    'holo-card',
    wgsl(commonWgsl, noiseWgsl, sdfWgsl, brdfWgsl, opticsWgsl, fullscreenWgsl, frame.wgslStruct, ibl.wgslStruct, iblWgsl, shaderWgsl),
  );
  const g0 = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
    ],
  });
  const pipeline = await device.createRenderPipelineAsync({
    label: 'holo-card',
    layout: device.createPipelineLayout({ bindGroupLayouts: [g0, ibl.bindGroupLayout] }),
    vertex: { module, entryPoint: 'vs_fullscreen' },
    fragment: { module, entryPoint: 'fs_main', targets: [{ format: HDR_FORMAT }] },
  });
  const linear = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  const makeGroup = () =>
    device.createBindGroup({
      layout: g0,
      entries: [
        { binding: 0, resource: { buffer: frame.buffer } },
        { binding: 1, resource: title.texture.createView() },
        { binding: 2, resource: linear },
      ],
    });
  let group = makeGroup();

  let titleKey = holo.title;
  let titleTimer = 0;
  params.onChange(() => {
    clearTimeout(titleTimer);
    titleTimer = window.setTimeout(async () => {
      if (holo.title === titleKey) return;
      titleKey = holo.title;
      const t = await textGen.generate({ text: holo.title.trim() || ' ', family: '"Helvetica Neue", Arial, ui-sans-serif, sans-serif', weight: 800, size: 140 });
      title.destroy();
      title = t;
      group = makeGroup();
    }, 200);
  });

  const rx = new Spring();
  const ry = new Spring();

  return {
    frame({ time, dt, size }) {
      const aspect = size.width / size.height;
      rx.damping = ry.damping = 26 - 18 * motion.springiness;
      const max = (motion.maxTilt * Math.PI) / 180;

      // Target tilt: device orientation, else the pointer, else a slow idle sway.
      let tx: number;
      let ty: number;
      if (gyro) {
        gyroBase ??= { ...gyro };
        tx = Math.max(-1, Math.min(1, (gyro.beta - gyroBase.beta) / 25)) * max;
        ty = Math.max(-1, Math.min(1, (gyro.gamma - gyroBase.gamma) / 25)) * max;
      } else if (pointer.inside && pointer.idle < 3) {
        tx = (pointer.smoothUv[1] - 0.5) * 2 * max;
        ty = (pointer.smoothUv[0] - 0.5) * 2 * max;
      } else {
        const s = reducedMotion ? 0.25 : 1;
        tx = Math.sin(time * 0.6) * max * 0.45 * s;
        ty = Math.sin(time * 0.43 + 1) * max * 0.6 * s;
      }
      const ax = rx.step(tx, dt);
      const ay = ry.step(ty, dt);
      const rot = mat4.multiply(mat4.rotationY(ay), mat4.rotationX(ax));

      const fitW = (CARD_HALF[0] * 1.5) / (Math.tan(FOV / 2) * aspect);
      const fitH = (CARD_HALF[1] * 1.3) / Math.tan(FOV / 2);
      const dist = Math.max(fitW, fitH);
      const eye = vec3.fromValues(0, 0, dist);
      const viewProj = mat4.multiply(mat4.perspective(FOV, aspect, 0.05, 50), mat4.lookAt(eye, [0, 0, 0], [0, 1, 0]));

      const la = (motion.lightAngle * Math.PI) / 180;
      const lightPos = [Math.sin(la) * 2.2, 1.8, Math.cos(la) * 2.2 + 0.4];
      const ti = motion.intensity;
      const glyphH = title.height - 2 * title.padding;
      let scale = 0.11 / Math.max(glyphH, 1);
      if ((title.width - 2 * title.padding) * scale > 1.0) scale = 1.0 / (title.width - 2 * title.padding);

      frame
        .set('invViewProj', mat4.inverse(viewProj))
        .set('cardX', [rot[0], rot[1], rot[2], 0])
        .set('cardY', [rot[4], rot[5], rot[6], 0])
        .set('cardZ', [rot[8], rot[9], rot[10], 0])
        .set('titleRect', [0, -CARD_HALF[1] + 0.15, (title.width * scale) / 2, (title.height * scale) / 2])
        .set('titleScale', scale)
        .set('cameraPos', eye)
        .set('envIntensity', motion.environment)
        .set('lightPos', lightPos)
        .set('lightRadius', 0.15)
        .set('lightColor', [ti, ti * 0.96, ti * 0.92])
        .set('pixelAngle', (2 * Math.tan(FOV / 2)) / size.height)
        .set('glowColor', hexToLinear('#3a1a4a').map((c) => c * 0.25))
        .set('cardHalf', CARD_HALF)
        .set('cardRadius', 0.05)
        .set('backdropZ', -0.9)
        .set('holoMode', holo.holoMode)
        .set('glitterScale', 420)
        .set('glitter', holo.glitter)
        .set('filmThickness', holo.filmThickness)
        .set('filmStrength', holo.filmStrength)
        .set('foilRoughness', holo.foilRoughness)
        .set('anisotropy', holo.anisotropy)
        .set('grooveSpacing', holo.grooveSpacing)
        .set('holoIntensity', holo.holoIntensity)
        .set('inkCover', holo.inkCover)
        .set('foilAmount', 1)
        .upload();
      ibl.update();

      const encoder = device.createCommandEncoder({ label: 'holo-card' });
      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: post.scene(size).view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.setBindGroup(1, ibl.bindGroup);
      pass.draw(3);
      pass.end();
      post.render(encoder, surface.currentTexture().createView(), { time });
      device.queue.submit([encoder.finish()]);
    },
    dispose() {
      clearTimeout(titleTimer);
      window.removeEventListener('deviceorientation', onOrientation);
      title.destroy();
      textGen.destroy();
      frame.destroy();
      ibl.destroy();
      post.destroy();
    },
  };
};

export default create;
