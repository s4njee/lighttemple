import commonWgsl from '../../shaders/common.wgsl?raw';
import fullscreenWgsl from '../../shaders/fullscreen.wgsl?raw';
import sdfWgsl from '../../shaders/sdf.wgsl?raw';
import { createShader, wgsl } from '../../gpu/shader';
import { UniformBlock } from '../../gpu/uniforms';
import { HDR_FORMAT, PostProcess, TONEMAP } from '../../post/post';
import { TextSdfGenerator, type TextSdf } from '../../sdf/text-sdf';
import { hexToLinear } from '../../ui/params';
import type { ExperimentFactory } from '../types';
import shaderWgsl from './shader.wgsl?raw';

const MAX_TILES = 10;
const TILE_FLOATS = 16;

const PALETTES: Record<string, { background: string; tiles: string[]; glyph: string }> = {
  sunset: { background: '#f4a259', tiles: ['#e76f51', '#2a9d8f', '#264653', '#e9c46a', '#8ab17d'], glyph: '#fdf6ec' },
  mint: { background: '#2ec4b6', tiles: ['#ff9f1c', '#e71d36', '#011627', '#fdfffc', '#ffbf69'], glyph: '#fdfffc' },
  midnight: { background: '#1d2440', tiles: ['#5e60ce', '#48bfe3', '#ff6b6b', '#ffd166', '#06d6a0'], glyph: '#fdfdfd' },
};

const create: ExperimentFactory = async (ctx) => {
  const { device, surface, params, pointer } = ctx;

  const design = params.add(
    {
      word: { value: 'SHADOW', type: 'text', maxLength: MAX_TILES },
      palette: { value: 'sunset', options: { Sunset: 'sunset', Mint: 'mint', Midnight: 'midnight' } },
      shape: { value: 0, options: { 'Rounded square': 0, Circle: 1 } },
      angle: { value: 45, min: -180, max: 180, step: 1, label: 'Angle (°)', hint: 'Used when the pointer is away; otherwise the light sits at the pointer.' },
      length: { value: 0.9, min: 0.05, max: 3, step: 0.01, label: 'Length', hint: 'In canvas heights; large values feel "infinite".' },
      fade: { value: 0.65, min: 0, max: 1, step: 0.01, hint: '0 = solid flat shadow, 1 = fades to nothing at full length.' },
      strength: { value: 0.7, min: 0, max: 1, step: 0.01, label: 'Darkness' },
    },
    'Design',
  );
  // Flat design wants exact colours: no tonemapping, no bloom, no vignette.
  const post = await PostProcess.create(ctx, { tonemap: TONEMAP.None, bloom: 0, vignette: 0, grain: 0.015 });

  const textGen = new TextSdfGenerator(device);
  const FONT = '"Helvetica Neue", Arial, ui-sans-serif, system-ui, sans-serif';
  // Letters are rasterised with spaces between them, so each tile's slice of the
  // SDF never contains its neighbours.
  const spaced = (word: string) => [...(word.trim() || ' ')].join('  ');
  let text: TextSdf = await textGen.generate({ text: spaced(design.word), family: FONT, weight: 800, size: 220 });

  const u = new UniformBlock(device, 'Flat', {
    background: 'vec3f',
    count: 'u32',
    shadowTint: 'vec3f',
    length: 'f32',
    glyphColor: 'vec3f',
    fade: 'f32',
    dir: 'vec2f',
    circles: 'f32',
    shadowStrength: 'f32',
  });
  const tileBuffer = device.createBuffer({ label: 'tiles', size: MAX_TILES * TILE_FLOATS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const tileData = new Float32Array(MAX_TILES * TILE_FLOATS);

  const module = await createShader(device, 'long-shadows', wgsl(commonWgsl, sdfWgsl, fullscreenWgsl, u.wgslStruct, shaderWgsl));
  const pipeline = await device.createRenderPipelineAsync({
    label: 'long-shadows',
    layout: 'auto',
    vertex: { module, entryPoint: 'vs_fullscreen' },
    fragment: { module, entryPoint: 'fs_main', targets: [{ format: HDR_FORMAT }] },
  });
  const linear = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  const makeGroup = () =>
    device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: u.buffer } },
        { binding: 1, resource: { buffer: tileBuffer } },
        { binding: 2, resource: text.texture.createView() },
        { binding: 3, resource: linear },
      ],
    });
  let group = makeGroup();

  let wordKey = design.word;
  let timer = 0;
  params.onChange(() => {
    clearTimeout(timer);
    timer = window.setTimeout(async () => {
      if (design.word === wordKey) return;
      wordKey = design.word;
      const t = await textGen.generate({ text: spaced(design.word), family: FONT, weight: 800, size: 220 });
      text.destroy();
      text = t;
      group = makeGroup();
    }, 200);
  });

  let dir: [number, number] = [Math.SQRT1_2, Math.SQRT1_2];

  return {
    frame({ time, dt, size }) {
      const { width: w, height: h } = size;
      const palette = PALETTES[design.palette] ?? PALETTES.sunset;
      const letters = text.letters.slice(0, MAX_TILES);
      const n = Math.max(1, letters.length);

      // Lay the tiles out in a centred row.
      const gap = 0.22;
      const tile = Math.min((0.86 * w) / (n + (n - 1) * gap), 0.34 * h);
      const total = n * tile + (n - 1) * gap * tile;
      const glyphH = text.baseline - text.padding; // ascent in texels (caps)
      const scale = (tile * 0.52) / Math.max(glyphH, 1); // px per texel
      letters.forEach((l, i) => {
        const o = i * TILE_FLOATS;
        const cx = (w - total) / 2 + tile / 2 + i * tile * (1 + gap);
        const cy = h * 0.5;
        // Extend each slice halfway into the gap on either side.
        const left = i > 0 ? (l.start - letters[i - 1].end) / 2 : text.padding;
        const right = i < letters.length - 1 ? (letters[i + 1].start - l.end) / 2 : text.padding;
        const u0 = (l.start - left) / text.width;
        const u1 = (l.end + right) / text.width;
        // Offset from the region's centre to the glyph's visual centre (cap height middle).
        const visualY = (text.padding + text.baseline) / 2;
        tileData.set([cx, cy, tile / 2, tile / 2], o);
        tileData.set([u0, 0, u1, 1], o + 4);
        tileData.set([...hexToLinear(palette.tiles[i % palette.tiles.length]), tile * 0.22], o + 8);
        tileData.set([scale, 0, visualY - text.height / 2, 0], o + 12);
      });
      device.queue.writeBuffer(tileBuffer, 0, tileData);

      // Light at the pointer => shadows point away from it; otherwise the set angle.
      let target: [number, number];
      if (pointer.inside && pointer.idle < 4) {
        const dx = w / 2 - pointer.uv[0] * w;
        const dy = h / 2 - pointer.uv[1] * h;
        const len = Math.hypot(dx, dy) || 1;
        target = [dx / len, dy / len];
      } else {
        const a = (design.angle * Math.PI) / 180;
        target = [Math.cos(a), Math.sin(a)];
      }
      const k = 1 - Math.pow(2, -dt / 0.12);
      dir = [dir[0] + (target[0] - dir[0]) * k, dir[1] + (target[1] - dir[1]) * k];
      const dl = Math.hypot(dir[0], dir[1]) || 1;

      u.set('background', hexToLinear(palette.background))
        .set('count', letters.length)
        .set('shadowTint', [0.5, 0.46, 0.55])
        .set('length', design.length * h)
        .set('glyphColor', hexToLinear(palette.glyph))
        .set('fade', design.fade)
        .set('dir', [dir[0] / dl, dir[1] / dl])
        .set('circles', design.shape)
        .set('shadowStrength', design.strength)
        .upload();

      const encoder = device.createCommandEncoder({ label: 'long-shadows' });
      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: post.scene(size).view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.draw(3);
      pass.end();
      post.render(encoder, surface.currentTexture().createView(), { time });
      device.queue.submit([encoder.finish()]);
    },
    dispose() {
      clearTimeout(timer);
      text.destroy();
      textGen.destroy();
      tileBuffer.destroy();
      u.destroy();
      post.destroy();
    },
  };
};

export default create;
