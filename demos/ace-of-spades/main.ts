/**
 * Ace of Spades: a standalone demo page built on the Light Temple engine.
 * A holographic ace of spades floating in black, with orbit controls.
 */

import { mat4 } from 'wgpu-matrix';
import brdfWgsl from '../../src/shaders/brdf.wgsl?raw';
import commonWgsl from '../../src/shaders/common.wgsl?raw';
import fullscreenWgsl from '../../src/shaders/fullscreen.wgsl?raw';
import noiseWgsl from '../../src/shaders/noise.wgsl?raw';
import opticsWgsl from '../../src/shaders/optics.wgsl?raw';
import sdfWgsl from '../../src/shaders/sdf.wgsl?raw';
import { getGpu } from '../../src/gpu/context';
import { generateMipmaps, mipLevelCount } from '../../src/gpu/mipmaps';
import { createShader, wgsl } from '../../src/gpu/shader';
import { Surface } from '../../src/gpu/surface';
import { UniformBlock } from '../../src/gpu/uniforms';
import { Ibl, iblWgsl } from '../../src/ibl/ibl';
import { HDR_FORMAT, PostProcess } from '../../src/post/post';
import { OrbitCamera } from '../../src/scene/orbit-camera';
import { TextSdfGenerator, type TextSdf } from '../../src/sdf/text-sdf';
import aceWgsl from './ace.wgsl?raw';
import emblemUrl from './eye-emblem.png';
import jostUrl from './fonts/jost-latin.woff2';

const CARD_HALF: [number, number] = [0.63, 0.88];
const FOV = (32 * Math.PI) / 180;
// Playing-card indices are a condensed Roman; a regular serif squeezed horizontally reads the same.
const SERIF = '"Times New Roman", Times, "Liberation Serif", "Nimbus Roman", Georgia, serif';
// The mark is set in Jost (SIL OFL, bundled), a geometric sans close to Supra / Futura.
const GEOMETRIC = 'Jost, Futura, "Century Gothic", ui-sans-serif, sans-serif';
// Pure #FF0000: the mark is composited after tonemapping (see `red_ink` in ace.wgsl).
const MARK_RED: [number, number, number] = [1, 0, 0];

declare global {
  interface Window {
    __ace?: { frames: number; error?: string };
  }
}

function fail(message: string): void {
  window.__ace = { frames: 0, error: message };
  const el = document.getElementById('fallback')!;
  el.textContent = message;
  el.hidden = false;
}

/**
 * Places a text SDF so its glyphs are `capHeight` tall, centred at (cx, cy) in card
 * units. `squeeze` < 1 narrows the glyphs (distances use the smaller scale, so the
 * edges stay anti-aliased correctly).
 */
function place(t: TextSdf, cx: number, cy: number, capHeight: number, squeeze = 1) {
  const sy = capHeight / Math.max(t.baseline - t.padding, 1); // card units per texel (caps)
  const sx = sy * squeeze;
  // Centre on the ink: horizontally on the glyph box, vertically on the cap height.
  const inkCx = t.width / 2;
  const inkCy = (t.padding + t.baseline) / 2;
  const rx = cx + (t.width / 2 - inkCx) * sx;
  const ry = cy - (t.height / 2 - inkCy) * sy;
  return { rect: [rx, ry, (t.width * sx) / 2, (t.height * sy) / 2], scalePad: [Math.min(sx, sy), t.padding * Math.min(sx, sy)] };
}

async function main(): Promise<void> {
  const canvas = document.getElementById('stage') as HTMLCanvasElement;
  const gpu = await getGpu((info) => info.reason !== 'destroyed' && fail(`The GPU device was lost (${info.message}). Reload to try again.`));
  const { device } = gpu;
  const format = gpu.presentationFormat;
  const surface = new Surface(canvas, device, format, { maxPixelRatio: 2 });
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

  const post = await PostProcess.create({ device, format }, { bloom: 0.05, bloomRadius: 1.2, vignette: 0, grain: 0, exposure: 1.6 });
  const ibl = await Ibl.create(device);
  await ibl.setProcedural({ kind: 'studio' });

  const textGen = new TextSdfGenerator(device);
  try {
    document.fonts.add(await new FontFace('Jost', `url(${jostUrl}) format('woff2')`, { weight: '100 900' }).load());
  } catch (err) {
    console.warn('Jost failed to load; using a fallback geometric sans.', err);
  }
  const [indexText, markText, emblemImage] = await Promise.all([
    textGen.generate({ text: 'A', family: SERIF, weight: 400, size: 220 }),
    textGen.generate({ text: 'mk7s', family: GEOMETRIC, weight: 600, size: 160 }),
    fetch(emblemUrl)
      .then((r) => r.blob())
      .then((b) => createImageBitmap(b, { colorSpaceConversion: 'none' })),
  ]);
  // The eye emblem is stored as masks (R = foil strokes, G = white plate), so it stays linear.
  const emblemTex = device.createTexture({
    label: 'eye-emblem',
    size: [emblemImage.width, emblemImage.height],
    format: 'rgba8unorm',
    mipLevelCount: mipLevelCount(emblemImage.width, emblemImage.height),
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  device.queue.copyExternalImageToTexture({ source: emblemImage }, { texture: emblemTex }, [emblemImage.width, emblemImage.height]);
  generateMipmaps(device, emblemTex);
  // Corner index like the reference: a tall condensed "A" with a solid pip beneath.
  const indexPos: [number, number] = [-CARD_HALF[0] + 0.115, CARD_HALF[1] - 0.17];
  const index = place(indexText, indexPos[0], indexPos[1], 0.15, 0.72);
  const indexPip: [number, number] = [indexPos[0], indexPos[1] - 0.17];
  // The eye emblem takes the club's place in the big spade; tiny red type sits just under it.
  const emblemHalfW = 0.21 * 0.44;
  const emblemHalfH = (emblemHalfW * emblemImage.height) / emblemImage.width;
  const emblemRect = [0, 0.27 * 0.44, emblemHalfW, emblemHalfH];
  const mark = place(markText, 0, 0.045 * 0.44, 0.02);

  const u = new UniformBlock(device, 'Ace', {
    invViewProj: 'mat4x4f',
    light0: 'vec4f',
    light0Color: 'vec4f',
    light1: 'vec4f',
    light1Color: 'vec4f',
    light2: 'vec4f',
    light2Color: 'vec4f',
    indexRect: 'vec4f',
    markRect: 'vec4f',
    emblemRect: 'vec4f',
    cameraPos: 'vec3f',
    pixelAngle: 'f32',
    cardHalf: 'vec2f',
    cardRadius: 'f32',
    halfThickness: 'f32',
    indexScalePad: 'vec2f',
    indexPip: 'vec2f',
    markScalePad: 'vec2f',
    envIntensity: 'f32',
    pearl: 'f32',
    glitter: 'f32',
    glitterScale: 'f32',
    filmThickness: 'f32',
    filmStrength: 'f32',
    foilRoughness: 'f32',
    anisotropy: 'f32',
    grooveSpacing: 'f32',
    holoIntensity: 'f32',
    emblemTexel: 'f32',
  });

  const module = await createShader(
    device,
    'ace-of-spades',
    wgsl(commonWgsl, noiseWgsl, sdfWgsl, brdfWgsl, opticsWgsl, fullscreenWgsl, u.wgslStruct, ibl.wgslStruct, iblWgsl, aceWgsl),
  );
  const g0 = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {} },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: {} },
    ],
  });
  const pipeline = await device.createRenderPipelineAsync({
    label: 'ace-of-spades',
    layout: device.createPipelineLayout({ bindGroupLayouts: [g0, ibl.bindGroupLayout] }),
    vertex: { module, entryPoint: 'vs_fullscreen' },
    fragment: { module, entryPoint: 'fs_main', targets: [{ format: HDR_FORMAT }] },
  });
  const group = device.createBindGroup({
    layout: g0,
    entries: [
      { binding: 0, resource: { buffer: u.buffer } },
      { binding: 1, resource: indexText.texture.createView() },
      { binding: 2, resource: markText.texture.createView() },
      { binding: 3, resource: device.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear' }) },
      { binding: 4, resource: emblemTex.createView() },
    ],
  });

  // Orbit controls: drag to orbit (all the way round to the back), wheel / pinch to zoom.
  const camera = new OrbitCamera(canvas, {
    target: [0, 0, 0],
    distance: 3.6,
    yaw: -0.35,
    pitch: 0.12,
    minDistance: 1.1,
    maxDistance: 8,
    minPitch: -1.35,
    maxPitch: 1.35,
    fovY: FOV,
    // Snappy: the card tracks the pointer closely, with just enough easing to stay smooth.
    smoothing: 0.025,
    rotateSpeed: 1.35,
    zoomSpeed: 1.6,
  });
  camera.autoRotate = reducedMotion ? 0 : 0.12;

  // Hide the hint after the first interaction.
  const hint = document.getElementById('hint')!;
  const dismiss = () => hint.classList.add('gone');
  canvas.addEventListener('pointerdown', dismiss, { once: true });
  canvas.addEventListener('wheel', dismiss, { once: true, passive: true });

  u.set('indexRect', index.rect)
    .set('indexScalePad', index.scalePad)
    .set('indexPip', indexPip)
    .set('markRect', mark.rect)
    .set('markScalePad', mark.scalePad)
    .set('emblemRect', emblemRect)
    .set('emblemTexel', (2 * emblemHalfW) / emblemImage.width)
    .set('cardHalf', CARD_HALF)
    .set('cardRadius', 0.055)
    .set('halfThickness', 0.006)
    .set('envIntensity', 0.8)
    .set('pearl', 0.0)
    .set('glitter', 0.16)
    .set('glitterScale', 380)
    .set('filmThickness', 430)
    .set('filmStrength', 0.6)
    .set('foilRoughness', 0.2)
    .set('anisotropy', 0.75)
    .set('grooveSpacing', 1500)
    .set('holoIntensity', 1.6);

  window.__ace = { frames: 0 };
  let last = performance.now();
  let time = 0;
  const tick = (now: number) => {
    requestAnimationFrame(tick);
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    time += dt;
    const { size } = surface;
    camera.update(dt, size.width / size.height);
    ibl.update();

    // Lights drift slowly so the foil keeps shimmering even when nobody touches it.
    const drift = reducedMotion ? 0 : time * 0.25;
    u.set('invViewProj', mat4.inverse(camera.viewProjection))
      .set('cameraPos', camera.position)
      .set('pixelAngle', (2 * Math.tan(FOV / 2)) / size.height)
      .set('light0', [1.6 * Math.cos(drift), 2.0, 2.6, 0])
      .set('light0Color', [14, 13.2, 12.2, 0])
      .set('light1', [-2.4, -0.6 + 0.8 * Math.sin(drift * 0.7), 1.8, 0])
      .set('light1Color', [5, 6, 8, 0])
      .set('light2', [0.6, 1.6, -2.6, 0])
      .set('light2Color', [10, 9.5, 9, 0])
      .upload();

    const encoder = device.createCommandEncoder({ label: 'ace' });
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: post.scene(size).view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.setBindGroup(1, ibl.bindGroup);
    pass.draw(3);
    pass.end();
    post.render(encoder, surface.currentTexture().createView(), { time, background: MARK_RED });
    device.queue.submit([encoder.finish()]);
    window.__ace!.frames++;
  };
  requestAnimationFrame(tick);
}

main().catch((err) => {
  console.error(err);
  fail(
    err instanceof Error && err.name === 'WebGpuUnavailableError'
      ? `${err.message}`
      : `Something went wrong: ${err instanceof Error ? err.message : String(err)}`,
  );
});
