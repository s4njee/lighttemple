/**
 * Minimal glTF 2.0 loader: enough to drop a hero object into an experiment.
 *
 * Supports .glb and .gltf (embedded data URIs or relative files), node
 * hierarchies (matrix or TRS), indexed and non-indexed triangle primitives,
 * float / normalized-integer attributes, and the core metal/roughness material
 * (base colour, metal-rough, normal, occlusion and emissive textures, with
 * sRGB-correct formats and generated mipmaps). Alpha MASK is honoured; BLEND
 * renders as opaque. No skinning, morph targets, animation, cameras or extensions.
 *
 * GPU layout produced (see gltf.wgsl):
 *   vertices: position f32x3 | normal f32x3 | uv f32x2 | tangent f32x4  (48 bytes)
 *             tangent.w == 0 means "no tangents": the shader derives a frame from screen derivatives.
 *   transforms: storage array<NodeTransform> indexed by DrawItem.transformIndex
 *   materials: one bind group per material (Gltf.materialBindGroupLayout)
 */

import { mat4, quat, vec3, type Mat4 } from 'wgpu-matrix';
import { uniformStruct, UniformBlock } from '../gpu/uniforms';
import { generateMipmaps, mipLevelCount } from '../gpu/mipmaps';
import gltfWgslBody from './gltf.wgsl?raw';

export const GLTF_VERTEX_STRIDE = 48;

export const GLTF_VERTEX_LAYOUT: GPUVertexBufferLayout = {
  arrayStride: GLTF_VERTEX_STRIDE,
  attributes: [
    { shaderLocation: 0, offset: 0, format: 'float32x3' },
    { shaderLocation: 1, offset: 12, format: 'float32x3' },
    { shaderLocation: 2, offset: 24, format: 'float32x2' },
    { shaderLocation: 3, offset: 32, format: 'float32x4' },
  ],
};

const MATERIAL_PARAMS = {
  baseColorFactor: 'vec4f',
  emissiveFactor: 'vec3f',
  normalScale: 'f32',
  metallicFactor: 'f32',
  roughnessFactor: 'f32',
  occlusionStrength: 'f32',
  alphaCutoff: 'f32',
} as const;

/** Prepend to shaders that use glTF meshes/materials (after common.wgsl). */
export const gltfWgsl = `${uniformStruct('GltfMaterialParams', MATERIAL_PARAMS)}\n\n${gltfWgslBody}`;

export interface DrawItem {
  vertexBuffer: GPUBuffer;
  indexBuffer: GPUBuffer;
  indexFormat: GPUIndexFormat;
  indexCount: number;
  material: GltfMaterial;
  /** Index into `GltfModel.transforms` (pass as firstInstance). */
  transformIndex: number;
}

export interface GltfMaterial {
  name: string;
  bindGroup: GPUBindGroup;
  doubleSided: boolean;
  alphaMode: 'OPAQUE' | 'MASK' | 'BLEND';
}

export interface GltfModel {
  items: DrawItem[];
  /** array<NodeTransform>: model matrix + normal matrix per node instance. */
  transforms: GPUBuffer;
  /** World-space bounds of the whole scene. */
  bounds: { min: [number, number, number]; max: [number, number, number] };
  /** Root matrix that centres the model on the ground plane and scales it to `height`. */
  fit(height: number): Mat4;
  destroy(): void;
}

// --- glTF JSON subset ----------------------------------------------------------

interface GltfJson {
  scene?: number;
  scenes?: { nodes?: number[] }[];
  nodes?: {
    mesh?: number;
    children?: number[];
    matrix?: number[];
    translation?: number[];
    rotation?: number[];
    scale?: number[];
  }[];
  meshes?: { primitives: GltfPrimitive[] }[];
  accessors?: {
    bufferView?: number;
    byteOffset?: number;
    componentType: number;
    normalized?: boolean;
    count: number;
    type: string;
    min?: number[];
    max?: number[];
    sparse?: unknown;
  }[];
  bufferViews?: { buffer: number; byteOffset?: number; byteLength: number; byteStride?: number }[];
  buffers?: { uri?: string; byteLength: number }[];
  images?: { uri?: string; bufferView?: number; mimeType?: string }[];
  samplers?: { magFilter?: number; minFilter?: number; wrapS?: number; wrapT?: number }[];
  textures?: { source?: number; sampler?: number }[];
  materials?: {
    name?: string;
    pbrMetallicRoughness?: {
      baseColorFactor?: number[];
      baseColorTexture?: { index: number };
      metallicFactor?: number;
      roughnessFactor?: number;
      metallicRoughnessTexture?: { index: number };
    };
    normalTexture?: { index: number; scale?: number };
    occlusionTexture?: { index: number; strength?: number };
    emissiveTexture?: { index: number };
    emissiveFactor?: number[];
    alphaMode?: 'OPAQUE' | 'MASK' | 'BLEND';
    alphaCutoff?: number;
    doubleSided?: boolean;
  }[];
  extensionsRequired?: string[];
}

interface GltfPrimitive {
  attributes: Record<string, number>;
  indices?: number;
  material?: number;
  mode?: number;
}

const COMPONENTS: Record<string, number> = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };
const GL_REPEAT = 10497;
const GL_MIRRORED = 33648;
const GL_CLAMP = 33071;

// --- Loading -------------------------------------------------------------------

export async function loadGltf(device: GPUDevice, source: string | ArrayBuffer, baseUrl?: string): Promise<GltfModel> {
  let data: ArrayBuffer;
  let base = baseUrl ?? '';
  if (typeof source === 'string') {
    const res = await fetch(source);
    if (!res.ok) throw new Error(`Failed to fetch ${source}: ${res.status}`);
    data = await res.arrayBuffer();
    base = baseUrl ?? new URL('.', new URL(source, location.href)).href;
  } else {
    data = source;
  }

  let json: GltfJson;
  let glbBin: ArrayBuffer | null = null;
  const magic = new DataView(data).getUint32(0, true);
  if (magic === 0x46546c67 /* 'glTF' */) {
    ({ json, bin: glbBin } = parseGlb(data));
  } else {
    json = JSON.parse(new TextDecoder().decode(data));
  }
  if (json.extensionsRequired?.length) {
    throw new Error(`Unsupported required glTF extensions: ${json.extensionsRequired.join(', ')}`);
  }

  const buffers = await Promise.all(
    (json.buffers ?? []).map(async (b, i) => {
      if (b.uri === undefined) {
        if (i === 0 && glbBin) return glbBin;
        throw new Error(`glTF buffer ${i} has no data`);
      }
      const res = await fetch(new URL(b.uri, base || location.href));
      return res.arrayBuffer();
    }),
  );

  const resources: { destroy(): void }[] = [];
  const textures = new TextureCache(device, json, buffers, base, resources);
  const fallback = createFallbackTextures(device, resources);
  const materialLayout = getMaterialBindGroupLayout(device);

  // Materials.
  const defaultSampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    addressModeU: 'repeat',
    addressModeV: 'repeat',
    maxAnisotropy: 8,
  });
  const materials: GltfMaterial[] = await Promise.all(
    (json.materials ?? [{}]).map(async (m) => {
      const pbr = m.pbrMetallicRoughness ?? {};
      const params = new UniformBlock(device, 'GltfMaterialParams', MATERIAL_PARAMS);
      resources.push(params);
      params
        .set('baseColorFactor', pbr.baseColorFactor ?? [1, 1, 1, 1])
        .set('emissiveFactor', m.emissiveFactor ?? [0, 0, 0])
        .set('normalScale', m.normalTexture?.scale ?? 1)
        .set('metallicFactor', pbr.metallicFactor ?? 1)
        .set('roughnessFactor', pbr.roughnessFactor ?? 1)
        .set('occlusionStrength', m.occlusionTexture?.strength ?? 1)
        .set('alphaCutoff', m.alphaMode === 'MASK' ? (m.alphaCutoff ?? 0.5) : -1)
        .upload();
      const [base, mr, normal, occlusion, emissive] = await Promise.all([
        textures.get(pbr.baseColorTexture?.index, true),
        textures.get(pbr.metallicRoughnessTexture?.index, false),
        textures.get(m.normalTexture?.index, false),
        textures.get(m.occlusionTexture?.index, false),
        textures.get(m.emissiveTexture?.index, true),
      ]);
      // glTF allows a sampler per texture; we use the base-colour one for the whole material.
      const sampler = textures.sampler(pbr.baseColorTexture?.index) ?? defaultSampler;
      return {
        name: m.name ?? 'material',
        doubleSided: m.doubleSided ?? false,
        alphaMode: m.alphaMode ?? 'OPAQUE',
        bindGroup: device.createBindGroup({
          label: `gltf-material:${m.name ?? ''}`,
          layout: materialLayout,
          entries: [
            { binding: 0, resource: { buffer: params.buffer } },
            { binding: 1, resource: sampler },
            { binding: 2, resource: (base ?? fallback.white).createView() },
            { binding: 3, resource: (mr ?? fallback.white).createView() },
            { binding: 4, resource: (normal ?? fallback.normal).createView() },
            { binding: 5, resource: (occlusion ?? fallback.white).createView() },
            { binding: 6, resource: (emissive ?? fallback.white).createView() },
          ],
        }),
      };
    }),
  );

  // Node traversal -> world matrices + draw items.
  const accessor = (index: number) => readAccessor(json, buffers, index);
  const meshCache = new Map<number, { primitives: Omit<DrawItem, 'transformIndex'>[]; localMin: number[]; localMax: number[] }>();
  const items: DrawItem[] = [];
  const worldMatrices: Mat4[] = [];
  const bmin = [Infinity, Infinity, Infinity];
  const bmax = [-Infinity, -Infinity, -Infinity];

  const buildMesh = (meshIndex: number) => {
    const cached = meshCache.get(meshIndex);
    if (cached) return cached;
    const prims: Omit<DrawItem, 'transformIndex'>[] = [];
    const lmin = [Infinity, Infinity, Infinity];
    const lmax = [-Infinity, -Infinity, -Infinity];
    for (const p of json.meshes![meshIndex].primitives) {
      if ((p.mode ?? 4) !== 4) continue; // triangles only
      const pos = accessor(p.attributes.POSITION);
      const count = pos.count;
      const indices = p.indices !== undefined ? accessor(p.indices).data : Uint32Array.from({ length: count }, (_, i) => i);
      const normals = p.attributes.NORMAL !== undefined ? accessor(p.attributes.NORMAL).data : computeNormals(pos.data, indices);
      const uvs = p.attributes.TEXCOORD_0 !== undefined ? accessor(p.attributes.TEXCOORD_0).data : null;
      const tangents = p.attributes.TANGENT !== undefined ? accessor(p.attributes.TANGENT).data : null;

      const v = new Float32Array(count * 12);
      for (let i = 0; i < count; i++) {
        const o = i * 12;
        for (let k = 0; k < 3; k++) {
          const x = pos.data[i * 3 + k];
          v[o + k] = x;
          lmin[k] = Math.min(lmin[k], x);
          lmax[k] = Math.max(lmax[k], x);
          v[o + 3 + k] = normals[i * 3 + k];
        }
        if (uvs) {
          v[o + 6] = uvs[i * 2];
          v[o + 7] = uvs[i * 2 + 1];
        }
        if (tangents) for (let k = 0; k < 4; k++) v[o + 8 + k] = tangents[i * 4 + k];
      }
      const vertexBuffer = device.createBuffer({ label: 'gltf-vertices', size: v.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(vertexBuffer, 0, v);
      const index32 = count > 65535;
      const idx = index32 ? Uint32Array.from(indices) : Uint16Array.from(indices);
      // writeBuffer sizes must be multiples of 4 bytes.
      const indexBuffer = device.createBuffer({
        label: 'gltf-indices',
        size: Math.ceil(idx.byteLength / 4) * 4,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      });
      const padded = new Uint8Array(indexBuffer.size);
      padded.set(new Uint8Array(idx.buffer, idx.byteOffset, idx.byteLength));
      device.queue.writeBuffer(indexBuffer, 0, padded);
      resources.push(vertexBuffer, indexBuffer);
      prims.push({
        vertexBuffer,
        indexBuffer,
        indexFormat: index32 ? 'uint32' : 'uint16',
        indexCount: idx.length,
        material: materials[p.material ?? 0] ?? materials[0],
      });
    }
    const built = { primitives: prims, localMin: lmin, localMax: lmax };
    meshCache.set(meshIndex, built);
    return built;
  };

  const visit = (nodeIndex: number, parent: Mat4) => {
    const node = json.nodes![nodeIndex];
    let local: Mat4;
    if (node.matrix) {
      local = new Float32Array(node.matrix) as Mat4; // column-major, same as wgpu-matrix
    } else {
      const t = mat4.translation(node.translation ?? [0, 0, 0]);
      const r = mat4.fromQuat(quat.fromValues(...((node.rotation ?? [0, 0, 0, 1]) as [number, number, number, number])));
      const s = mat4.scaling(node.scale ?? [1, 1, 1]);
      local = mat4.multiply(mat4.multiply(t, r), s);
    }
    const world = mat4.multiply(parent, local);
    if (node.mesh !== undefined) {
      const mesh = buildMesh(node.mesh);
      const transformIndex = worldMatrices.length;
      worldMatrices.push(world);
      for (const p of mesh.primitives) items.push({ ...p, transformIndex });
      // Transform the 8 corners of the local AABB into world space.
      for (let c = 0; c < 8; c++) {
        const corner = [0, 1, 2].map((k) => ((c >> k) & 1 ? mesh.localMax[k] : mesh.localMin[k]));
        const w = vec3.transformMat4(corner, world);
        for (let k = 0; k < 3; k++) {
          bmin[k] = Math.min(bmin[k], w[k]);
          bmax[k] = Math.max(bmax[k], w[k]);
        }
      }
    }
    for (const child of node.children ?? []) visit(child, world);
  };
  const scene = json.scenes?.[json.scene ?? 0];
  const roots = scene?.nodes ?? (json.nodes ?? []).map((_, i) => i);
  for (const r of roots) visit(r, mat4.identity());
  if (items.length === 0) throw new Error('glTF contains no triangle meshes');

  // NodeTransform = { model: mat4x4f, normal: mat4x4f }
  const transformData = new Float32Array(worldMatrices.length * 32);
  worldMatrices.forEach((m, i) => {
    transformData.set(m, i * 32);
    transformData.set(mat4.transpose(mat4.inverse(m)), i * 32 + 16);
  });
  const transforms = device.createBuffer({
    label: 'gltf-transforms',
    size: transformData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(transforms, 0, transformData);
  resources.push(transforms);

  const bounds = { min: bmin as [number, number, number], max: bmax as [number, number, number] };
  return {
    items,
    transforms,
    bounds,
    fit(height: number) {
      const size = bmax[1] - bmin[1] || 1;
      const s = height / size;
      const cx = (bmin[0] + bmax[0]) / 2;
      const cz = (bmin[2] + bmax[2]) / 2;
      return mat4.multiply(mat4.scaling([s, s, s]), mat4.translation([-cx, -bmin[1], -cz]));
    },
    destroy() {
      for (const r of resources) r.destroy();
    },
  };
}

function parseGlb(data: ArrayBuffer): { json: GltfJson; bin: ArrayBuffer | null } {
  const view = new DataView(data);
  if (view.getUint32(4, true) !== 2) throw new Error('Only glTF 2.0 .glb files are supported');
  let offset = 12;
  let json: GltfJson | null = null;
  let bin: ArrayBuffer | null = null;
  while (offset < data.byteLength) {
    const length = view.getUint32(offset, true);
    const type = view.getUint32(offset + 4, true);
    const chunk = data.slice(offset + 8, offset + 8 + length);
    if (type === 0x4e4f534a /* JSON */) json = JSON.parse(new TextDecoder().decode(chunk));
    else if (type === 0x004e4942 /* BIN */) bin = chunk;
    offset += 8 + length;
  }
  if (!json) throw new Error('.glb has no JSON chunk');
  return { json, bin };
}

/** Reads an accessor into a tightly packed Float32Array (or Uint32Array for indices). */
function readAccessor(json: GltfJson, buffers: ArrayBuffer[], index: number): { data: Float32Array | Uint32Array; count: number } {
  const a = json.accessors![index];
  if (a.sparse) throw new Error('Sparse accessors are not supported');
  const n = COMPONENTS[a.type];
  const out = a.componentType === 5126 || a.normalized ? new Float32Array(a.count * n) : new Uint32Array(a.count * n);
  if (a.bufferView === undefined) return { data: out, count: a.count }; // all zeros per spec
  const bv = json.bufferViews![a.bufferView];
  const dv = new DataView(buffers[bv.buffer], (bv.byteOffset ?? 0) + (a.byteOffset ?? 0));
  const size = { 5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4 }[a.componentType];
  if (!size) throw new Error(`Unsupported accessor componentType ${a.componentType}`);
  const stride = bv.byteStride ?? size * n;
  const read = (o: number): number => {
    switch (a.componentType) {
      case 5126: return dv.getFloat32(o, true);
      case 5125: return dv.getUint32(o, true);
      case 5123: return a.normalized ? dv.getUint16(o, true) / 65535 : dv.getUint16(o, true);
      case 5122: return a.normalized ? Math.max(dv.getInt16(o, true) / 32767, -1) : dv.getInt16(o, true);
      case 5121: return a.normalized ? dv.getUint8(o) / 255 : dv.getUint8(o);
      default: return a.normalized ? Math.max(dv.getInt8(o) / 127, -1) : dv.getInt8(o);
    }
  };
  for (let i = 0; i < a.count; i++) for (let k = 0; k < n; k++) out[i * n + k] = read(i * stride + k * size);
  return { data: out, count: a.count };
}

function computeNormals(pos: Float32Array | Uint32Array, idx: Float32Array | Uint32Array): Float32Array {
  const n = new Float32Array(pos.length);
  for (let i = 0; i + 2 < idx.length; i += 3) {
    const [a, b, c] = [idx[i] * 3, idx[i + 1] * 3, idx[i + 2] * 3];
    const e1 = [pos[b] - pos[a], pos[b + 1] - pos[a + 1], pos[b + 2] - pos[a + 2]];
    const e2 = [pos[c] - pos[a], pos[c + 1] - pos[a + 1], pos[c + 2] - pos[a + 2]];
    const f = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    for (const v of [a, b, c]) for (let k = 0; k < 3; k++) n[v + k] += f[k];
  }
  for (let i = 0; i < n.length; i += 3) {
    const l = Math.hypot(n[i], n[i + 1], n[i + 2]) || 1;
    n[i] /= l;
    n[i + 1] /= l;
    n[i + 2] /= l;
  }
  return n;
}

class TextureCache {
  private cache = new Map<string, Promise<GPUTexture | null>>();
  private samplers = new Map<number, GPUSampler>();

  constructor(
    private readonly device: GPUDevice,
    private readonly json: GltfJson,
    private readonly buffers: ArrayBuffer[],
    private readonly base: string,
    private readonly resources: { destroy(): void }[],
  ) {}

  get(textureIndex: number | undefined, srgb: boolean): Promise<GPUTexture | null> {
    if (textureIndex === undefined) return Promise.resolve(null);
    const source = this.json.textures?.[textureIndex]?.source;
    if (source === undefined) return Promise.resolve(null);
    const key = `${source}:${srgb}`;
    let p = this.cache.get(key);
    if (!p) {
      p = this.load(source, srgb);
      this.cache.set(key, p);
    }
    return p;
  }

  sampler(textureIndex: number | undefined): GPUSampler | null {
    const si = textureIndex === undefined ? undefined : this.json.textures?.[textureIndex]?.sampler;
    if (si === undefined) return null;
    let s = this.samplers.get(si);
    if (!s) {
      const d = this.json.samplers![si];
      const wrap = (w?: number): GPUAddressMode =>
        w === GL_CLAMP ? 'clamp-to-edge' : w === GL_MIRRORED ? 'mirror-repeat' : (w ?? GL_REPEAT) === GL_REPEAT ? 'repeat' : 'repeat';
      const nearestMag = d.magFilter === 9728;
      s = this.device.createSampler({
        magFilter: nearestMag ? 'nearest' : 'linear',
        minFilter: 'linear',
        mipmapFilter: 'linear',
        addressModeU: wrap(d.wrapS),
        addressModeV: wrap(d.wrapT),
        maxAnisotropy: nearestMag ? 1 : 8,
      });
      this.samplers.set(si, s);
    }
    return s;
  }

  private async load(imageIndex: number, srgb: boolean): Promise<GPUTexture | null> {
    const img = this.json.images?.[imageIndex];
    if (!img) return null;
    let blob: Blob;
    if (img.bufferView !== undefined) {
      const bv = this.json.bufferViews![img.bufferView];
      const bytes = new Uint8Array(this.buffers[bv.buffer], bv.byteOffset ?? 0, bv.byteLength);
      blob = new Blob([bytes.slice()], { type: img.mimeType ?? 'image/png' });
    } else if (img.uri) {
      blob = await (await fetch(new URL(img.uri, this.base || location.href))).blob();
    } else {
      return null;
    }
    const bitmap = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
    const texture = this.device.createTexture({
      label: `gltf-image-${imageIndex}${srgb ? '-srgb' : ''}`,
      size: [bitmap.width, bitmap.height],
      format: srgb ? 'rgba8unorm-srgb' : 'rgba8unorm',
      mipLevelCount: mipLevelCount(bitmap.width, bitmap.height),
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.device.queue.copyExternalImageToTexture({ source: bitmap }, { texture }, [bitmap.width, bitmap.height]);
    bitmap.close();
    generateMipmaps(this.device, texture);
    this.resources.push(texture);
    return texture;
  }
}

function createFallbackTextures(device: GPUDevice, resources: { destroy(): void }[]) {
  const make = (rgba: number[], label: string) => {
    const t = device.createTexture({ label, size: [1, 1], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    device.queue.writeTexture({ texture: t }, new Uint8Array(rgba), { bytesPerRow: 4 }, [1, 1]);
    resources.push(t);
    return t;
  };
  return { white: make([255, 255, 255, 255], 'gltf-white'), normal: make([128, 128, 255, 255], 'gltf-flat-normal') };
}

const layouts = new WeakMap<GPUDevice, GPUBindGroupLayout>();

/** Bind group layout for glTF materials (bind it at group 2; see gltf.wgsl). */
export function getMaterialBindGroupLayout(device: GPUDevice): GPUBindGroupLayout {
  let layout = layouts.get(device);
  if (!layout) {
    const tex = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility: GPUShaderStage.FRAGMENT, texture: {} });
    layout = device.createBindGroupLayout({
      label: 'gltf-material',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        tex(2),
        tex(3),
        tex(4),
        tex(5),
        tex(6),
      ],
    });
    layouts.set(device, layout);
  }
  return layout;
}
