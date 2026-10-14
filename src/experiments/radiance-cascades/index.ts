import commonWgsl from '../../shaders/common.wgsl?raw';
import fullscreenWgsl from '../../shaders/fullscreen.wgsl?raw';
import sdfWgsl from '../../shaders/sdf.wgsl?raw';
import { createShader, wgsl } from '../../gpu/shader';
import { UniformBlock, uniformStruct } from '../../gpu/uniforms';
import { HDR_FORMAT, PostProcess } from '../../post/post';
import { JumpFlood } from '../../sdf/jump-flood';
import { hexToLinear } from '../../gpu/color';
import type { ExperimentFactory } from '../types';
import shaderWgsl from './shader.wgsl?raw';

const SCENE_FORMAT: GPUTextureFormat = 'rgba16float';
const PASS_SCHEMA = { index: 'u32', pad0: 'u32', pad1: 'u32', pad2: 'u32' } as const;

const create: ExperimentFactory = async (ctx) => {
  const { device, surface, params, pointer } = ctx;
  const coarse = window.matchMedia('(pointer: coarse)').matches;

  const paint = params.add(
    {
      tool: { value: 'light', options: { 'Paint light': 'light', 'Paint wall': 'wall', Erase: 'erase' }, hint: 'Drag on the canvas to paint.' },
      brushColor: { value: '#ffb36b', type: 'color', label: 'Light colour' },
      brushIntensity: { value: 2.5, min: 0.1, max: 10, step: 0.1, label: 'Brightness' },
      brushSize: { value: 10, min: 2, max: 60, step: 1, label: 'Brush (px)' },
      cursorLight: { value: true, label: 'Cursor light', hint: 'A lamp that follows the pointer (while not painting).' },
      cursorColor: { value: '#fff1d6', type: 'color', label: 'Cursor colour' },
    },
    'Paint',
  );
  const gi = params.add(
    {
      scale: { value: coarse ? 0.25 : 0.5, options: { 'Quarter res': 0.25, 'Half res': 0.5, 'Full res': 1 }, label: 'GI resolution' },
      interval: { value: 1.5, min: 0.5, max: 8, step: 0.1, label: 'Base interval', hint: 'Ray length of cascade 0 in probe texels; each cascade is 4× longer.' },
      sky: { value: '#0d1424', type: 'color', label: 'Ambient sky' },
      floor: { value: 0.85, min: 0, max: 1, step: 0.01, label: 'Floor albedo' },
      wall: { value: '#1b1b20', type: 'color' },
      view: { value: 0, options: { Lit: 0, 'Distance field': 1, 'Raw fluence': 2, 'Reference (64 rays)': 3 } },
    },
    'Cascades',
  );
  params.button('Reset scene', () => (reseed = true));
  params.button('Clear', () => (clear = true));
  const post = await PostProcess.create(ctx, { bloom: 0.08, vignette: 0.25, grain: 0.02 });

  const rc = new UniformBlock(device, 'Rc', {
    orb: 'vec4f',
    giSize: 'vec2f',
    cascadeCount: 'u32',
    interval0: 'f32',
    orbColor: 'vec3f',
    displayScale: 'f32',
    sky: 'vec3f',
    floorAlbedo: 'f32',
    wallColor: 'vec3f',
    giStrength: 'f32',
    seedIntensity: 'f32',
    debugView: 'u32',
  });
  const brush = new UniformBlock(device, 'Brush', { value: 'vec4f', a: 'vec2f', b: 'vec2f', radius: 'f32' });
  const module = await createShader(
    device,
    'radiance-cascades',
    wgsl(commonWgsl, sdfWgsl, fullscreenWgsl, rc.wgslStruct, uniformStruct('RcPass', PASS_SCHEMA), brush.wgslStruct, shaderWgsl),
  );
  const make = (entryPoint: string, format: GPUTextureFormat) =>
    device.createRenderPipelineAsync({
      label: `rc:${entryPoint}`,
      layout: 'auto',
      vertex: { module, entryPoint: 'vs_fullscreen' },
      fragment: { module, entryPoint, targets: [{ format }] },
    });
  const [cascadePipeline, displayPipeline, brushPipeline, seedPipeline] = await Promise.all([
    make('fs_cascade', 'rgba16float'),
    make('fs_display', HDR_FORMAT),
    make('fs_brush', SCENE_FORMAT),
    make('fs_seed', SCENE_FORMAT),
  ]);
  const jfa = new JumpFlood(device);

  // Per-cascade uniforms (just the index), allocated for the maximum count.
  const MAX_CASCADES = 8;
  const passBuffers = Array.from({ length: MAX_CASCADES }, (_, i) => {
    const b = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(b, 0, new Uint32Array([i, 0, 0, 0]));
    return b;
  });

  interface Resources {
    giW: number;
    giH: number;
    cascades: number;
    scene: GPUTexture;
    sdf: GPUTexture;
    tex: [GPUTexture, GPUTexture];
    cascadeGroups: GPUBindGroup[];
    displayGroup: GPUBindGroup;
    seedGroup: GPUBindGroup;
    brushGroup: GPUBindGroup;
  }
  let res: Resources | null = null;
  let reseed = true;
  let clear = false;
  let sceneDirty = true;

  const build = (canvasW: number, canvasH: number, scale: number): Resources => {
    const w0 = Math.max(16, Math.round(canvasW * scale));
    const h0 = Math.max(16, Math.round(canvasH * scale));
    const l0 = gi.interval;
    // Enough cascades for the last interval to span the diagonal.
    const diag = Math.hypot(w0, h0);
    let cascades = 1;
    while (cascades < MAX_CASCADES && (l0 * (4 ** cascades - 1)) / 3 < diag) cascades++;
    // Every cascade must tile the GI area with whole probes.
    const align = 2 ** (cascades - 1);
    const giW = Math.ceil(w0 / align) * align;
    const giH = Math.ceil(h0 / align) * align;
    const tex2 = (w: number, h: number, label: string, usage: number) =>
      device.createTexture({ label, size: [w, h], format: 'rgba16float', usage });
    const RT = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
    const scene = tex2(giW, giH, 'rc-scene', RT);
    const sdf = tex2(giW, giH, 'rc-sdf', GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING);
    const tex: [GPUTexture, GPUTexture] = [tex2(giW * 2, giH * 2, 'rc-cascade-a', RT), tex2(giW * 2, giH * 2, 'rc-cascade-b', RT)];
    // Cascade n writes tex[n % 2] and reads the merged cascade n+1 from the other.
    const cascadeGroups = Array.from({ length: cascades }, (_, n) =>
      device.createBindGroup({
        layout: cascadePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: rc.buffer } },
          { binding: 1, resource: scene.createView() },
          { binding: 2, resource: sdf.createView() },
          { binding: 3, resource: tex[(n + 1) % 2].createView() },
          { binding: 4, resource: { buffer: passBuffers[n] } },
        ],
      }),
    );
    const displayGroup = device.createBindGroup({
      layout: displayPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: rc.buffer } },
        { binding: 1, resource: scene.createView() },
        { binding: 2, resource: sdf.createView() },
        { binding: 3, resource: tex[0].createView() },
      ],
    });
    const seedGroup = device.createBindGroup({
      layout: seedPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: rc.buffer } }],
    });
    const brushGroup = device.createBindGroup({
      layout: brushPipeline.getBindGroupLayout(0),
      entries: [{ binding: 5, resource: { buffer: brush.buffer } }],
    });
    return { giW, giH, cascades, scene, sdf, tex, cascadeGroups, displayGroup, seedGroup, brushGroup };
  };
  const destroyRes = (r: Resources | null) => {
    r?.scene.destroy();
    r?.sdf.destroy();
    r?.tex.forEach((t) => t.destroy());
  };

  let lastBrush: [number, number] | null = null;
  let orb: [number, number] = [0.5, 0.5];
  let resKey = '';

  return {
    frame({ time, dt, size }) {
      // (Re)build when the canvas, GI scale or interval change. The painting is
      // resolution-dependent, so a rebuild re-seeds the starter scene.
      const key = `${size.width}x${size.height}@${gi.scale}:${gi.interval}`;
      if (key !== resKey) {
        resKey = key;
        destroyRes(res);
        res = build(size.width, size.height, gi.scale);
        reseed = true;
      }
      const r = res!;
      const scale = gi.scale; // GI texels per canvas pixel (the GI area may extend past the canvas)

      const encoder = device.createCommandEncoder({ label: 'radiance-cascades' });
      const fullscreen = (pipeline: GPURenderPipeline, group: GPUBindGroup, target: GPUTexture, load: GPULoadOp) => {
        const pass = encoder.beginRenderPass({
          colorAttachments: [{ view: target.createView(), loadOp: load, storeOp: 'store', clearValue: [0, 0, 0, 0] }],
        });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, group);
        pass.draw(3);
        pass.end();
      };

      // Orb (cursor light), in GI texels.
      const painting = pointer.down && pointer.inside;
      const idle = !pointer.inside || pointer.idle > 4;
      const target: [number, number] = idle
        ? [r.giW * (0.5 + 0.3 * Math.sin(time * 0.35)), r.giH * (0.55 + 0.25 * Math.sin(time * 0.5))]
        : [pointer.uv[0] * size.width * scale, pointer.uv[1] * size.height * scale];
      const k = 1 - Math.pow(2, -dt / (idle ? 0.6 : 0.04));
      orb = [orb[0] + (target[0] - orb[0]) * k, orb[1] + (target[1] - orb[1]) * k];
      const oc = hexToLinear(paint.cursorColor);

      rc.set('orb', [orb[0], orb[1], 7 * scale * 2, paint.cursorLight && !painting ? 1 : 0])
        .set('giSize', [r.giW, r.giH])
        .set('cascadeCount', r.cascades)
        .set('interval0', gi.interval)
        .set('orbColor', oc.map((c) => c * 3))
        .set('displayScale', scale)
        .set('sky', hexToLinear(gi.sky))
        .set('floorAlbedo', gi.floor)
        .set('wallColor', hexToLinear(gi.wall))
        .set('giStrength', 1)
        .set('seedIntensity', 1)
        .set('debugView', gi.view)
        .upload();

      if (reseed || clear) {
        if (clear) {
          const pass = encoder.beginRenderPass({
            colorAttachments: [{ view: r.scene.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }],
          });
          pass.end();
        } else {
          fullscreen(seedPipeline, r.seedGroup, r.scene, 'clear');
        }
        reseed = clear = false;
        sceneDirty = true;
      }

      // Painting: a capsule from the previous pointer position to the current one.
      if (painting) {
        const p: [number, number] = [pointer.uv[0] * size.width * scale, pointer.uv[1] * size.height * scale];
        const from = lastBrush ?? p;
        const lc = hexToLinear(paint.brushColor);
        const value =
          paint.tool === 'light'
            ? [lc[0] * paint.brushIntensity, lc[1] * paint.brushIntensity, lc[2] * paint.brushIntensity, 1]
            : paint.tool === 'wall'
              ? [0, 0, 0, 1]
              : [0, 0, 0, 0];
        brush.set('value', value).set('a', from).set('b', p).set('radius', paint.brushSize * scale).upload();
        fullscreen(brushPipeline, r.brushGroup, r.scene, 'load');
        lastBrush = p;
        sceneDirty = true;
      } else {
        lastBrush = null;
      }

      if (sceneDirty) {
        jfa.encode(encoder, r.scene, r.sdf);
        sceneDirty = false;
      }

      // Cascades, top-down: each merges with the one above it.
      for (let n = r.cascades - 1; n >= 0; n--) {
        fullscreen(cascadePipeline, r.cascadeGroups[n], r.tex[n % 2], 'clear');
      }

      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: post.scene(size).view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
      });
      pass.setPipeline(displayPipeline);
      pass.setBindGroup(0, r.displayGroup);
      pass.draw(3);
      pass.end();
      post.render(encoder, surface.currentTexture().createView(), { time, raw: gi.view !== 0 });
      device.queue.submit([encoder.finish()]);
    },
    dispose() {
      destroyRes(res);
      jfa.destroy();
      passBuffers.forEach((b) => b.destroy());
      rc.destroy();
      brush.destroy();
      post.destroy();
    },
  };
};

export default create;
