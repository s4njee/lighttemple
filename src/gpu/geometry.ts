/**
 * Procedural meshes with a single interleaved layout:
 *   @location(0) position: vec3f, @location(1) normal: vec3f, @location(2) uv: vec2f
 */

export interface MeshData {
  vertices: Float32Array<ArrayBuffer>;
  indices: Uint32Array<ArrayBuffer>;
}

export interface Mesh {
  vertexBuffer: GPUBuffer;
  indexBuffer: GPUBuffer;
  indexCount: number;
  destroy(): void;
}

export const VERTEX_STRIDE = 8 * 4;

export const MESH_VERTEX_LAYOUT: GPUVertexBufferLayout = {
  arrayStride: VERTEX_STRIDE,
  attributes: [
    { shaderLocation: 0, offset: 0, format: 'float32x3' },
    { shaderLocation: 1, offset: 12, format: 'float32x3' },
    { shaderLocation: 2, offset: 24, format: 'float32x2' },
  ],
};

/** Matching WGSL vertex input struct. */
export const MESH_VERTEX_WGSL = /* wgsl */ `
struct VertexIn {
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
};`;

export function uploadMesh(device: GPUDevice, data: MeshData, label = 'mesh'): Mesh {
  const vertexBuffer = device.createBuffer({
    label: `${label}:vertices`,
    size: data.vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertexBuffer, 0, data.vertices);
  const indexBuffer = device.createBuffer({
    label: `${label}:indices`,
    size: data.indices.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(indexBuffer, 0, data.indices);
  return {
    vertexBuffer,
    indexBuffer,
    indexCount: data.indices.length,
    destroy() {
      vertexBuffer.destroy();
      indexBuffer.destroy();
    },
  };
}

/** XZ plane facing +Y, centred on the origin. */
export function plane(size = 1, segments = 1): MeshData {
  const v: number[] = [];
  const idx: number[] = [];
  for (let z = 0; z <= segments; z++) {
    for (let x = 0; x <= segments; x++) {
      const u = x / segments;
      const w = z / segments;
      v.push((u - 0.5) * size, 0, (w - 0.5) * size, 0, 1, 0, u, w);
    }
  }
  const row = segments + 1;
  for (let z = 0; z < segments; z++) {
    for (let x = 0; x < segments; x++) {
      const a = z * row + x;
      idx.push(a, a + row, a + 1, a + 1, a + row, a + row + 1);
    }
  }
  return { vertices: new Float32Array(v), indices: new Uint32Array(idx) };
}

/** Axis-aligned box with flat-shaded faces. */
export function box(sx = 1, sy = 1, sz = 1): MeshData {
  const hx = sx / 2;
  const hy = sy / 2;
  const hz = sz / 2;
  // normal, tangent u, tangent v (u x v = n)
  const faces: [number[], number[], number[]][] = [
    [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
    [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
    [[0, 1, 0], [1, 0, 0], [0, 0, -1]],
    [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
    [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
    [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
  ];
  const v: number[] = [];
  const idx: number[] = [];
  for (const [n, t, b] of faces) {
    const base = v.length / 8;
    for (const [s, r] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      const p = [0, 1, 2].map((i) => (n[i] + s * t[i] + r * b[i]) * [hx, hy, hz][i]);
      v.push(...p, ...n, (s + 1) / 2, (r + 1) / 2);
    }
    idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }
  return { vertices: new Float32Array(v), indices: new Uint32Array(idx) };
}

export function sphere(radius = 0.5, widthSegments = 48, heightSegments = 32): MeshData {
  const v: number[] = [];
  const idx: number[] = [];
  for (let y = 0; y <= heightSegments; y++) {
    const vv = y / heightSegments;
    const theta = vv * Math.PI;
    for (let x = 0; x <= widthSegments; x++) {
      const u = x / widthSegments;
      const phi = u * Math.PI * 2;
      const nx = -Math.cos(phi) * Math.sin(theta);
      const ny = Math.cos(theta);
      const nz = Math.sin(phi) * Math.sin(theta);
      v.push(nx * radius, ny * radius, nz * radius, nx, ny, nz, u, vv);
    }
  }
  const row = widthSegments + 1;
  for (let y = 0; y < heightSegments; y++) {
    for (let x = 0; x < widthSegments; x++) {
      const a = y * row + x;
      const b = a + row;
      if (y !== 0) idx.push(a, b, a + 1);
      if (y !== heightSegments - 1) idx.push(a + 1, b, b + 1);
    }
  }
  return { vertices: new Float32Array(v), indices: new Uint32Array(idx) };
}

export function torus(radius = 0.5, tube = 0.2, radialSegments = 32, tubularSegments = 96): MeshData {
  const v: number[] = [];
  const idx: number[] = [];
  for (let j = 0; j <= radialSegments; j++) {
    for (let i = 0; i <= tubularSegments; i++) {
      const u = (i / tubularSegments) * Math.PI * 2;
      const t = (j / radialSegments) * Math.PI * 2;
      const cx = Math.cos(u) * radius;
      const cz = Math.sin(u) * radius;
      const px = (radius + tube * Math.cos(t)) * Math.cos(u);
      const py = tube * Math.sin(t);
      const pz = (radius + tube * Math.cos(t)) * Math.sin(u);
      const nx = px - cx;
      const ny = py;
      const nz = pz - cz;
      const len = Math.hypot(nx, ny, nz) || 1;
      v.push(px, py, pz, nx / len, ny / len, nz / len, i / tubularSegments, j / radialSegments);
    }
  }
  const row = tubularSegments + 1;
  for (let j = 1; j <= radialSegments; j++) {
    for (let i = 1; i <= tubularSegments; i++) {
      const a = row * j + i - 1;
      const b = row * (j - 1) + i - 1;
      const c = row * (j - 1) + i;
      const d = row * j + i;
      idx.push(a, d, b, b, d, c);
    }
  }
  return { vertices: new Float32Array(v), indices: new Uint32Array(idx) };
}
