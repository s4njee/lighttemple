import { mat4, vec3, type Mat4 } from 'wgpu-matrix';
import brdfWgsl from '../../shaders/brdf.wgsl?raw';
import commonWgsl from '../../shaders/common.wgsl?raw';
import fullscreenWgsl from '../../shaders/fullscreen.wgsl?raw';
import noiseWgsl from '../../shaders/noise.wgsl?raw';
import { hexToLinear } from '../../gpu/color';
import { MESH_VERTEX_LAYOUT, MESH_VERTEX_WGSL, sphere, torus, uploadMesh } from '../../gpu/geometry';
import { createShader, wgsl } from '../../gpu/shader';
import { TargetPool } from '../../gpu/targets';
import { UniformBlock } from '../../gpu/uniforms';
import { HDR_FORMAT, PostProcess } from '../../post/post';
import { TemporalAA } from '../../post/taa';
import { OrbitCamera } from '../../scene/orbit-camera';
import { PROP_WGSL, PropSet, type Prop } from '../../scene/props';
import { cylinder, lathe, roomShell, subdividedBox } from '../../scene/shapes';
import type { ExperimentFactory } from '../types';
import shaderWgsl from './shader.wgsl?raw';

const MSAA_SAMPLES = 4;
const DEPTH_FORMAT: GPUTextureFormat = 'depth24plus';
const SHADOW_FORMAT: GPUTextureFormat = 'depth32float';
const SHADOW_NEAR = 0.05;
const SHADOW_FAR = 12;

// The room: x across, y up, z toward the default camera. Floor at y = 0.
const ROOM = { w: 7, h: 3.2, d: 5.6 };
const WINDOW = [1.5, 1.8, 0.6, 0.72] as const; // centre x, y; half width, height (back wall)
const RUG = [-0.3, -0.35, 1.45, 1.0] as const; // centre x, z; half extents

const MAT = { plain: 0, room: 1, flame: 2, rug: 3, painting: 4 } as const;

/**
 * Cube-face view rotations (rows of the 3×3), in the order +X, −X, +Y, −Y, +Z, −Z.
 * Each maps a direction d onto the face so that view x / −z and y / −z land where the
 * cube-sampling convention (sc, tc on the major axis) will look for it. They are mirror
 * images (determinant −1), so the shadow pass draws with culling off.
 */
const CUBE_ROWS: [number, number, number][][] = [
  [[0, 0, -1], [0, 1, 0], [-1, 0, 0]],
  [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
  [[1, 0, 0], [0, 0, -1], [0, -1, 0]],
  [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
  [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
  [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
];

function cubeFaceView(rows: [number, number, number][], p: ArrayLike<number>): Mat4 {
  const m = mat4.identity();
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) m[j * 4 + i] = rows[i][j];
    m[12 + i] = -(rows[i][0] * p[0] + rows[i][1] * p[1] + rows[i][2] * p[2]);
  }
  return m;
}

/** Deterministic PRNG so the bookshelf looks the same on every visit. */
function mulberry32(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const create: ExperimentFactory = async (ctx) => {
  const { device, surface, params, reducedMotion } = ctx;
  const coarse = window.matchMedia('(pointer: coarse)').matches;

  const lantern = params.add(
    {
      intensity: { value: 5, min: 0, max: 20, step: 0.05 },
      color: { value: '#ffae5c', type: 'color' },
      posX: { value: -0.3, min: -3, max: 3, step: 0.01, label: 'Position x' },
      height: { value: 1.7, min: 0.7, max: 2.6, step: 0.01, hint: 'Height of the flame; the lantern hangs from the ceiling on a chain.' },
      posZ: { value: -0.45, min: -2.4, max: 2.4, step: 0.01, label: 'Position z' },
      radius: {
        value: 0.012,
        min: 0.002,
        max: 0.12,
        step: 0.001,
        label: 'Flame radius',
        hint: 'Size of the light (m). Sets the PCSS penumbra and keeps inverse-square falloff finite near the flame.',
      },
      flicker: { value: reducedMotion ? 0 : 0.6, min: 0, max: 1, step: 0.01, hint: 'Candle flicker: intensity and a tiny flame wobble.' },
      swing: { value: reducedMotion ? 0 : 4, min: 0, max: 15, step: 0.1, label: 'Swing (°)', hint: 'Pendulum amplitude of the hanging lantern.' },
    },
    'Lantern',
  );
  const shadow = params.add(
    {
      technique: {
        value: 'cube',
        options: { 'Cube map (6 passes)': 'cube', 'Dual paraboloid (2 passes)': 'dp' },
        hint: 'Dual paraboloid uses the same memory in two square maps, but bends straight edges (needs tessellation) and has a seam at flame height.',
      },
      filterMode: {
        value: 3,
        options: { Off: 0, 'Hard (2×2 HW)': 1, 'PCF (Poisson)': 2, PCSS: 3 },
        label: 'Filter',
      },
      mapSize: {
        value: coarse ? 512 : 1024,
        options: { '256': 256, '512': 512, '1024': 1024, '2048': 2048 },
        label: 'Face size',
        hint: 'Texels per cube face. The paraboloid maps get the same total memory (√3 × wider).',
      },
      pcfRadius: { value: 2, min: 0, max: 8, step: 0.1, label: 'PCF radius', hint: 'Poisson kernel radius in shadow-map texels.' },
      depthBias: { value: 1.5, min: 0, max: 8, step: 0.1, label: 'Depth bias', hint: 'Along the light ray, in shadow-map texels at the receiver.' },
      normalOffset: { value: 1, min: 0, max: 6, step: 0.1, label: 'Normal offset', hint: 'In shadow-map texels at the receiver.' },
    },
    'Shadows',
  );
  const look = params.add(
    {
      wall: { value: '#d6c7ad', type: 'color' },
      floor: { value: '#ffffff', type: 'color', label: 'Floor tint' },
      moonlight: { value: 0.035, min: 0, max: 0.3, step: 0.001, hint: 'Cool ambient from the window.' },
      bounce: { value: 0.12, min: 0, max: 1, step: 0.01, hint: 'A flat, warm ambient that scales with the lantern: a stand-in for indirect light.' },
      antialiasing: {
        value: 'msaa',
        options: { 'MSAA 4×': 'msaa', TAA: 'taa', Off: 'off' },
        hint: 'TAA also accumulates the per-pixel rotated Poisson taps over frames, so soft shadows come out smooth.',
      },
      view: { value: 0, options: { Lit: 0, Visibility: 1, 'Cube face / hemisphere': 2, 'Light distance': 3 } },
      showMap: { value: false, label: 'Show shadow map', hint: 'The cube map unfolded into a cross (or the two paraboloid discs), as distance from the flame.' },
      animate: { value: !reducedMotion },
    },
    'Look & debug',
  );
  const post = await PostProcess.create(ctx, { exposure: 1.1, bloom: 0.06, vignette: 0.35, grain: 0.025 });
  const taa = await TemporalAA.create(device);

  // --- Scene ------------------------------------------------------------------
  const meshes = {
    room: uploadMesh(device, roomShell(ROOM.w, ROOM.h, ROOM.d, 0.5), 'room'),
    box: uploadMesh(device, subdividedBox(1, 1, 1, 3), 'box'),
    sphere: uploadMesh(device, sphere(1, 32, 20), 'sphere'),
    ring: uploadMesh(device, torus(0.04, 0.008, 12, 32), 'ring'),
    candle: uploadMesh(device, cylinder(0.038, 0.13, 24), 'candle'),
    base: uploadMesh(device, cylinder(0.16, 0.035, 32), 'lantern-base'),
    cap: uploadMesh(device, lathe([[0, 0], [0.18, 0], [0.17, 0.02], [0.04, 0.11], [0.025, 0.13], [0, 0.13]], { segments: 32 }), 'cap'),
    flame: uploadMesh(device, lathe([[0, -0.03], [0.012, -0.022], [0.016, -0.004], [0.011, 0.018], [0.004, 0.04], [0, 0.05]], { segments: 16, creaseAngle: 2 }), 'flame'),
    vase: uploadMesh(
      device,
      lathe([[0, 0], [0.06, 0], [0.09, 0.06], [0.1, 0.15], [0.07, 0.25], [0.04, 0.3], [0.045, 0.34], [0.055, 0.36], [0, 0.36]], { creaseAngle: 1.2 }),
      'vase',
    ),
  };

  // Light state, updated every frame before the props are evaluated.
  const lightPos = vec3.fromValues(0, 1.7, 0);
  let lanternMatrix = mat4.identity();
  let chainLength = 1;

  const T = (x: number, y: number, z: number) => mat4.translation([x, y, z]);
  const box = (x: number, y: number, z: number, sx: number, sy: number, sz: number, rotY = 0): Mat4 =>
    mat4.scale(mat4.rotateY(T(x, y, z), rotY), [sx, sy, sz]);

  const props: Prop[] = [];
  const add = (p: Prop) => props.push(p);
  // Room + plinth.
  add({ mesh: meshes.room, color: '#ffffff', roughness: 0.85, extra: MAT.room });
  add({ mesh: meshes.box, color: '#231b16', roughness: 0.7, transform: () => box(0, -0.151, 0, ROOM.w + 0.3, 0.3, ROOM.d + 0.3) });
  add({ mesh: meshes.box, color: '#7a2c22', roughness: 0.95, extra: MAT.rug, transform: () => box(RUG[0], 0.006, RUG[1], RUG[2] * 2, 0.012, RUG[3] * 2) });

  // Table, with a vase of dry stems, apples and a book.
  const wood = '#5e3d27';
  const tx = -0.3;
  const tz = -0.5;
  add({ mesh: meshes.box, color: wood, roughness: 0.55, transform: () => box(tx, 0.76, tz, 1.7, 0.06, 0.95) });
  for (const [sx, sz] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
    add({ mesh: meshes.box, color: wood, roughness: 0.6, transform: () => box(tx + sx * 0.75, 0.365, tz + sz * 0.4, 0.07, 0.73, 0.07) });
  }
  add({ mesh: meshes.vase, color: '#2e5866', roughness: 0.22, transform: () => T(tx - 0.45, 0.79, tz - 0.15) });
  for (const [a, b, len] of [[0.25, 0.3, 0.55], [-0.2, 1.9, 0.5], [0.1, 4.0, 0.62]]) {
    // Thin stems leaning out of the vase: delicate, sharp-to-soft shadows on the wall.
    add({
      mesh: meshes.box,
      color: '#6b5a3e',
      roughness: 0.8,
      transform: () => {
        const m = mat4.rotateY(T(tx - 0.45, 1.1, tz - 0.15), b);
        return mat4.scale(mat4.translate(mat4.rotateZ(m, a), [0, len / 2 - 0.05, 0]), [0.008, len, 0.008]);
      },
    });
  }
  for (const [x, z] of [[0.05, -0.35], [0.16, -0.42], [0.1, -0.26]]) {
    add({ mesh: meshes.sphere, color: '#8c2a1c', roughness: 0.35, transform: () => mat4.uniformScale(T(tx + x, 0.835, tz + z), 0.045) });
  }
  add({ mesh: meshes.box, color: '#3b4a5a', roughness: 0.7, transform: () => box(tx + 0.45, 0.81, tz + 0.15, 0.24, 0.04, 0.32, 0.4) });

  // Two ladder-back chairs: the slats throw striped shadows.
  const chair = (cx: number, cz: number, rotY: number) => {
    const c = '#4a3020';
    const at = (x: number, y: number, z: number, sx: number, sy: number, sz: number) => () =>
      mat4.scale(mat4.translate(mat4.rotateY(T(cx, 0, cz), rotY), [x, y, z]), [sx, sy, sz]);
    add({ mesh: meshes.box, color: c, roughness: 0.6, transform: at(0, 0.46, 0, 0.46, 0.045, 0.46) });
    for (const [sx, sz] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
      add({ mesh: meshes.box, color: c, roughness: 0.6, transform: at(sx * 0.19, 0.23, sz * 0.19, 0.04, 0.46, 0.04) });
    }
    for (const sx of [-1, 1]) add({ mesh: meshes.box, color: c, roughness: 0.6, transform: at(sx * 0.19, 0.72, -0.21, 0.04, 0.52, 0.04) });
    add({ mesh: meshes.box, color: c, roughness: 0.6, transform: at(0, 0.95, -0.21, 0.42, 0.06, 0.03) });
    for (const y of [0.62, 0.78]) add({ mesh: meshes.box, color: c, roughness: 0.6, transform: at(0, y, -0.21, 0.38, 0.035, 0.02) });
  };
  chair(tx + 0.1, tz + 0.75, Math.PI);
  chair(tx - 1.15, tz, Math.PI / 2);

  // Bookshelf against the back wall.
  {
    const sx = -2.3;
    const sz = -ROOM.d / 2 + 0.17;
    const c = '#3d2a1d';
    for (const x of [-0.58, 0.58]) add({ mesh: meshes.box, color: c, roughness: 0.6, transform: () => box(sx + x, 1.0, sz, 0.04, 2.0, 0.32) });
    add({ mesh: meshes.box, color: c, roughness: 0.6, transform: () => box(sx, 2.0, sz, 1.2, 0.04, 0.32) });
    const shelves = [0.08, 0.56, 1.04, 1.52];
    for (const y of shelves) add({ mesh: meshes.box, color: c, roughness: 0.6, transform: () => box(sx, y, sz, 1.14, 0.03, 0.3) });
    const rnd = mulberry32(7);
    const palette = ['#6e2a24', '#2c4a3a', '#8a6b3a', '#273548', '#5a3d5c', '#b39a72', '#3c3c3c'];
    for (const y of shelves) {
      let x = -0.54;
      while (x < 0.5) {
        const w = 0.03 + rnd() * 0.04;
        const h = 0.24 + rnd() * 0.16;
        if (rnd() < 0.12) {
          x += 0.08; // gaps let light through between books
          continue;
        }
        const tilt = x > 0.42 ? -0.25 : 0;
        const bx = sx + x + w / 2;
        const color = palette[Math.floor(rnd() * palette.length)];
        add({
          mesh: meshes.box,
          color,
          roughness: 0.7,
          transform: () => mat4.scale(mat4.rotateZ(T(bx, y + 0.015 + h / 2, sz + 0.02), tilt), [w, h, 0.22]),
        });
        x += w + 0.004;
      }
    }
  }

  // A chest of drawers under the window, and a painting on the left wall.
  add({ mesh: meshes.box, color: '#4b3324', roughness: 0.55, transform: () => box(WINDOW[0], 0.42, -ROOM.d / 2 + 0.24, 1.2, 0.84, 0.46) });
  add({ mesh: meshes.box, color: '#c9b38a', roughness: 0.4, transform: () => box(WINDOW[0] - 0.3, 0.92, -ROOM.d / 2 + 0.25, 0.16, 0.16, 0.16, 0.5) });
  add({ mesh: meshes.box, color: '#8a6a3a', roughness: 0.35, metallic: 0.8, transform: () => box(-ROOM.w / 2 + 0.015, 1.75, 0.6, 0.03, 0.72, 0.92) });
  add({ mesh: meshes.box, color: '#ffffff', roughness: 0.8, extra: MAT.painting, transform: () => box(-ROOM.w / 2 + 0.017, 1.75, 0.6, 0.034, 0.6, 0.8) });

  // The lantern, in its own frame with the flame at the origin.
  const lanternPart = (p: Omit<Prop, 'transform'>, local: Mat4) => add({ ...p, transform: () => mat4.multiply(lanternMatrix, local) });
  const iron = { color: '#2a2724', roughness: 0.45, metallic: 0.9 };
  lanternPart({ mesh: meshes.base, ...iron }, T(0, -0.215, 0));
  lanternPart({ mesh: meshes.candle, color: '#efe6d2', roughness: 0.6 }, T(0, -0.18, 0));
  lanternPart({ mesh: meshes.flame, color: '#ffffff', roughness: 1, extra: MAT.flame, castsShadow: false }, mat4.identity());
  for (const [x, z] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
    lanternPart({ mesh: meshes.box, ...iron }, mat4.scale(T(x * 0.135, 0, z * 0.135), [0.016, 0.42, 0.016]));
  }
  for (const [x, z] of [[0, -1], [0, 1], [-1, 0], [1, 0]]) {
    lanternPart({ mesh: meshes.box, ...iron }, mat4.scale(T(x * 0.135, 0, z * 0.135), [0.008, 0.42, 0.008]));
  }
  for (const y of [-0.11, 0.12]) {
    for (const s of [-1, 1]) {
      lanternPart({ mesh: meshes.box, ...iron }, mat4.scale(T(0, y, s * 0.135), [0.27, 0.01, 0.01]));
      lanternPart({ mesh: meshes.box, ...iron }, mat4.scale(T(s * 0.135, y, 0), [0.01, 0.01, 0.27]));
    }
  }
  lanternPart({ mesh: meshes.cap, ...iron }, T(0, 0.21, 0));
  lanternPart({ mesh: meshes.ring, ...iron }, mat4.rotateX(T(0, 0.38, 0), Math.PI / 2));
  add({
    mesh: meshes.box,
    ...iron,
    transform: () => mat4.multiply(lanternMatrix, mat4.scale(T(0, (0.42 + chainLength) / 2, 0), [0.012, Math.max(0.01, chainLength - 0.42), 0.012])),
  });

  const propSet = new PropSet(device, props, 'lantern-room');

  // --- Uniforms & pipelines -----------------------------------------------------
  const frame = new UniformBlock(device, 'Frame', {
    viewProj: 'mat4x4f',
    cameraPos: 'vec3f',
    lightRadius: 'f32',
    lightPos: 'vec3f',
    shadowNear: 'f32',
    lightColor: 'vec4f',
    ambientColor: 'vec4f',
    bounceColor: 'vec4f',
    wallTint: 'vec4f',
    floorTint: 'vec4f',
    window: 'vec4f',
    rug: 'vec4f',
    shadowFar: 'f32',
    mapSize: 'f32',
    technique: 'u32',
    filterMode: 'u32',
    pcfRadius: 'f32',
    depthBias: 'f32',
    normalOffset: 'f32',
    debugView: 'u32',
    noiseOffset: 'f32',
    windowGlow: 'f32',
    flameGlow: 'f32',
  });
  const FACE_SCHEMA = { viewProj: 'mat4x4f', lightPos: 'vec3f', hemisphere: 'f32', near: 'f32', far: 'f32' } as const;
  // Six cube faces, then the two paraboloid hemispheres (+y, −y).
  const faces = Array.from({ length: 8 }, () => new UniformBlock(device, 'Face', FACE_SCHEMA));

  const module = await createShader(
    device,
    'point-light-shadows',
    wgsl(commonWgsl, fullscreenWgsl, noiseWgsl, brdfWgsl, MESH_VERTEX_WGSL, PROP_WGSL, frame.wgslStruct, faces[0].wgslStruct, shaderWgsl),
  );

  const bindGroupLayout = device.createBindGroupLayout({
    label: 'lantern-room',
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth', viewDimension: 'cube' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth', viewDimension: '2d-array' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
      { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'non-filtering' } },
    ],
  });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
  const shadowBindGroupLayout = device.createBindGroupLayout({
    label: 'lantern-shadow',
    entries: [
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 6, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
    ],
  });
  const shadowLayout = device.createPipelineLayout({ bindGroupLayouts: [shadowBindGroupLayout] });

  const makeScenePipelines = (count: number) =>
    Promise.all([
      device.createRenderPipelineAsync({
        label: `lantern-main-x${count}`,
        layout,
        vertex: { module, entryPoint: 'vs_main', buffers: [MESH_VERTEX_LAYOUT] },
        fragment: { module, entryPoint: 'fs_main', targets: [{ format: HDR_FORMAT }] },
        primitive: { topology: 'triangle-list', cullMode: 'back' },
        depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: 'less' },
        multisample: { count },
      }),
      device.createRenderPipelineAsync({
        label: `lantern-overlay-x${count}`,
        layout,
        vertex: { module, entryPoint: 'vs_fullscreen' },
        fragment: { module, entryPoint: 'fs_overlay', targets: [{ format: HDR_FORMAT }] },
        depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: 'always' },
        multisample: { count },
      }),
    ]);
  const [cubePipeline, dpPipeline, msaaPipelines, singlePipelines] = await Promise.all([
    device.createRenderPipelineAsync({
      label: 'lantern-cube-shadow',
      layout: shadowLayout,
      vertex: { module, entryPoint: 'vs_cube_shadow', buffers: [MESH_VERTEX_LAYOUT] },
      // The face views are mirror images, and thin casters need both sides anyway.
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: SHADOW_FORMAT, depthWriteEnabled: true, depthCompare: 'less' },
    }),
    device.createRenderPipelineAsync({
      label: 'lantern-dp-shadow',
      layout: shadowLayout,
      vertex: { module, entryPoint: 'vs_dp_shadow', buffers: [MESH_VERTEX_LAYOUT] },
      fragment: { module, entryPoint: 'fs_dp_shadow', targets: [] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: SHADOW_FORMAT, depthWriteEnabled: true, depthCompare: 'less' },
    }),
    makeScenePipelines(MSAA_SAMPLES),
    makeScenePipelines(1),
  ]);

  const compareSampler = device.createSampler({
    label: 'lantern-shadow-compare',
    compare: 'less-equal',
    magFilter: 'linear',
    minFilter: 'linear',
  });
  const pointSampler = device.createSampler({ label: 'lantern-shadow-point' });
  const shadowBindGroups = faces.map((f) =>
    device.createBindGroup({
      layout: shadowBindGroupLayout,
      entries: [
        { binding: 1, resource: { buffer: propSet.buffer } },
        { binding: 6, resource: { buffer: f.buffer } },
      ],
    }),
  );

  // --- Shadow maps: only the active technique gets real memory -------------------
  const shadowTexture = (label: string, size: number, layers: number) =>
    device.createTexture({
      label,
      size: [size, size, layers],
      format: SHADOW_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
  let cubeMap = shadowTexture('shadow-cube-dummy', 1, 6);
  let dpMap = shadowTexture('shadow-dp-dummy', 1, 2);
  let mapKey = '';
  let mainBindGroup: GPUBindGroup | null = null;
  const ensureShadowMaps = (technique: string, size: number) => {
    const key = `${technique}:${size}`;
    if (key === mapKey) return;
    mapKey = key;
    cubeMap.destroy();
    dpMap.destroy();
    // Same memory either way: two paraboloid maps of √3·N cover what six N² faces do.
    const dpSize = Math.round((size * Math.sqrt(3)) / 16) * 16;
    cubeMap = technique === 'cube' ? shadowTexture('shadow-cube', size, 6) : shadowTexture('shadow-cube-dummy', 1, 6);
    dpMap = technique === 'dp' ? shadowTexture('shadow-dp', dpSize, 2) : shadowTexture('shadow-dp-dummy', 1, 2);
    mainBindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: frame.buffer } },
        { binding: 1, resource: { buffer: propSet.buffer } },
        { binding: 2, resource: cubeMap.createView({ dimension: 'cube' }) },
        { binding: 3, resource: dpMap.createView({ dimension: '2d-array' }) },
        { binding: 4, resource: compareSampler },
        { binding: 5, resource: pointSampler },
      ],
    });
  };

  const targets = new TargetPool(device);
  const camera = new OrbitCamera(surface.canvas, {
    target: [-0.2, 1.1, -0.3],
    distance: 9.5,
    yaw: 0.62,
    pitch: 0.42,
    minPitch: 0.05,
    maxPitch: 1.3,
    minDistance: 3,
    maxDistance: 20,
  });
  let animTime = 0;

  return {
    frame({ time, dt, size }) {
      if (look.animate) animTime += dt;
      camera.update(dt, size.width / size.height);
      ensureShadowMaps(shadow.technique, shadow.mapSize);

      // Lantern: a pendulum hanging from the ceiling above the chosen spot.
      const t = animTime;
      chainLength = ROOM.h - lantern.height;
      const g = 9.81;
      const omega = Math.sqrt(g / chainLength);
      const amp = (lantern.swing * Math.PI) / 180;
      const pivot = T(lantern.posX, ROOM.h, lantern.posZ);
      const swingX = amp * Math.sin(omega * t);
      const swingZ = amp * 0.45 * Math.sin(omega * 0.93 * t + 1.1);
      lanternMatrix = mat4.translate(mat4.rotateZ(mat4.rotateX(pivot, swingX), swingZ), [0, -chainLength, 0]);
      // Flicker: a sum of incommensurate sines in brightness, and a wobble of the flame itself.
      const fl = lantern.flicker;
      const wave = 0.5 * Math.sin(t * 7.3) + 0.3 * Math.sin(t * 13.1 + 1.3) + 0.2 * Math.sin(t * 23.7 + 0.7);
      const gust = Math.max(0, Math.sin(t * 0.9) * Math.sin(t * 2.3 + 0.4)) ** 4;
      const flick = 1 + fl * (0.12 * wave - 0.35 * gust);
      const wobble = [0.006 * fl * Math.sin(t * 5.1), 0, 0.006 * fl * Math.cos(t * 6.7)];
      vec3.transformMat4(wobble, lanternMatrix, lightPos);
      propSet.update(t);

      const aa = look.antialiasing;
      const samples = aa === 'msaa' ? MSAA_SAMPLES : 1;
      const [mainPipeline, overlayPipeline] = samples > 1 ? msaaPipelines : singlePipelines;
      const fullSize = [size.width, size.height] as const;
      const scene = post.scene(size);
      const msaaColor = samples > 1 ? targets.get('msaa', { size: fullSize, format: HDR_FORMAT, sampleCount: samples }) : null;
      const depth = targets.get(`depth-x${samples}`, {
        size: fullSize,
        format: DEPTH_FORMAT,
        sampleCount: samples,
        usage: samples > 1 ? GPUTextureUsage.RENDER_ATTACHMENT : GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      if (aa !== 'taa') taa.reset();
      const viewProj = aa === 'taa' ? taa.jitter(camera.viewProjection, size) : camera.viewProjection;

      const lc = hexToLinear(lantern.color);
      const power = lantern.intensity * flick;
      const moon = [0.45, 0.6, 1.0].map((c) => c * look.moonlight);
      const bounce = lc.map((c) => c * power * look.bounce * 0.05);
      frame
        .set('viewProj', viewProj)
        .set('cameraPos', camera.position)
        .set('lightRadius', lantern.radius)
        .set('lightPos', lightPos)
        .set('shadowNear', SHADOW_NEAR)
        .set('lightColor', [lc[0] * power, lc[1] * power, lc[2] * power, 0])
        .set('ambientColor', [...moon, 0])
        .set('bounceColor', [...bounce, 0])
        .set('wallTint', [...hexToLinear(look.wall), 0])
        .set('floorTint', [...hexToLinear(look.floor), 0])
        .set('window', WINDOW)
        .set('rug', RUG)
        .set('shadowFar', SHADOW_FAR)
        .set('mapSize', shadow.technique === 'cube' ? shadow.mapSize : dpMap.width / 2)
        .set('technique', shadow.technique === 'cube' ? 0 : 1)
        .set('filterMode', shadow.filterMode)
        .set('pcfRadius', shadow.pcfRadius)
        .set('depthBias', shadow.depthBias)
        .set('normalOffset', shadow.normalOffset)
        .set('debugView', look.view)
        .set('noiseOffset', aa === 'taa' ? taa.frameNoiseOffset : 0)
        .set('windowGlow', 1)
        .set('flameGlow', 2.5)
        .upload();

      const proj = mat4.perspective(Math.PI / 2, 1, SHADOW_NEAR, SHADOW_FAR);
      faces.forEach((f, i) => {
        f.set('lightPos', lightPos).set('near', SHADOW_NEAR).set('far', SHADOW_FAR);
        if (i < 6) f.set('viewProj', mat4.multiply(proj, cubeFaceView(CUBE_ROWS[i], lightPos)));
        else f.set('hemisphere', i === 6 ? 1 : -1);
        f.upload();
      });

      const encoder = device.createCommandEncoder({ label: 'point-light-shadows' });

      // 1. Shadow map: six cube faces, or two paraboloid hemispheres.
      if (shadow.filterMode !== 0 || look.showMap) {
        const cube = shadow.technique === 'cube';
        const map = cube ? cubeMap : dpMap;
        for (let i = 0; i < (cube ? 6 : 2); i++) {
          const pass = encoder.beginRenderPass({
            label: cube ? `shadow-cube-${i}` : `shadow-dp-${i}`,
            colorAttachments: [],
            depthStencilAttachment: {
              view: map.createView({ dimension: '2d', baseArrayLayer: i, arrayLayerCount: 1 }),
              depthClearValue: 1,
              depthLoadOp: 'clear',
              depthStoreOp: 'store',
            },
          });
          pass.setPipeline(cube ? cubePipeline : dpPipeline);
          pass.setBindGroup(0, shadowBindGroups[cube ? i : 6 + i]);
          propSet.draw(pass, { castersOnly: true });
          pass.end();
        }
      }

      // 2. Main pass.
      const mp = encoder.beginRenderPass({
        label: 'lantern-main',
        colorAttachments: [
          msaaColor
            ? { view: msaaColor.view, resolveTarget: scene.view, loadOp: 'clear', storeOp: 'discard', clearValue: [0, 0, 0, 0] }
            : { view: scene.view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] },
        ],
        depthStencilAttachment: {
          view: depth.view,
          depthClearValue: 1,
          depthLoadOp: 'clear',
          depthStoreOp: aa === 'taa' ? 'store' : 'discard',
        },
      });
      mp.setPipeline(mainPipeline);
      mp.setBindGroup(0, mainBindGroup!);
      propSet.draw(mp);
      if (look.showMap) {
        const cube = shadow.technique === 'cube';
        const h = Math.round(Math.min(size.height * 0.36, size.width * (cube ? 0.27 : 0.18)));
        const w = cube ? Math.round((h * 4) / 3) : h * 2;
        const m = Math.round(16 * size.pixelRatio);
        mp.setViewport(m, size.height - h - m, w, h, 0, 1);
        mp.setPipeline(overlayPipeline);
        mp.draw(3);
      }
      mp.end();

      const source =
        aa === 'taa' ? taa.resolve(encoder, { color: scene.texture, depth: depth.texture, viewProj: camera.viewProjection }) : undefined;
      post.render(encoder, surface.currentTexture().createView(), {
        time,
        raw: look.view !== 0,
        background: hexToLinear('#0f0d0c'),
        source,
      });
      device.queue.submit([encoder.finish()]);
    },
    dispose() {
      camera.destroy();
      cubeMap.destroy();
      dpMap.destroy();
      targets.destroy();
      post.destroy();
      taa.destroy();
      propSet.destroy();
      frame.destroy();
      faces.forEach((f) => f.destroy());
      Object.values(meshes).forEach((m) => m.destroy());
    },
  };
};

export default create;
