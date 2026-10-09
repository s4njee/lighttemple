/**
 * Builds the classic "roughness × metalness" reference grid as an in-memory
 * glTF (JSON + base64 buffer), so it goes through exactly the same loader and
 * shading path as any downloaded model.
 */

import { sphere } from '../../gpu/geometry';

const COLUMNS = 6;

export function materialGridGltf(): ArrayBuffer {
  const mesh = sphere(0.22, 48, 32);
  const vertexCount = mesh.vertices.length / 8;
  const positions = new Float32Array(vertexCount * 3);
  const normals = new Float32Array(vertexCount * 3);
  const uvs = new Float32Array(vertexCount * 2);
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < vertexCount; i++) {
    for (let k = 0; k < 3; k++) {
      const p = mesh.vertices[i * 8 + k];
      positions[i * 3 + k] = p;
      min[k] = Math.min(min[k], p);
      max[k] = Math.max(max[k], p);
      normals[i * 3 + k] = mesh.vertices[i * 8 + 3 + k];
    }
    uvs[i * 2] = mesh.vertices[i * 8 + 6];
    uvs[i * 2 + 1] = mesh.vertices[i * 8 + 7];
  }
  const indices = mesh.indices;

  const parts = [positions, normals, uvs, indices];
  const total = parts.reduce((n, p) => n + p.byteLength, 0);
  const bin = new Uint8Array(total);
  const views: { buffer: 0; byteOffset: number; byteLength: number }[] = [];
  let offset = 0;
  for (const p of parts) {
    bin.set(new Uint8Array(p.buffer, p.byteOffset, p.byteLength), offset);
    views.push({ buffer: 0, byteOffset: offset, byteLength: p.byteLength });
    offset += p.byteLength;
  }

  // Row 0: dielectrics (warm off-white), row 1: metals (gold-ish). Roughness 0 → 1 across.
  const materials: object[] = [];
  const meshes: object[] = [];
  const nodes: object[] = [];
  for (let row = 0; row < 2; row++) {
    for (let col = 0; col < COLUMNS; col++) {
      const roughness = col / (COLUMNS - 1);
      materials.push({
        name: `${row ? 'metal' : 'dielectric'}-r${roughness.toFixed(2)}`,
        pbrMetallicRoughness: {
          baseColorFactor: row ? [1.0, 0.78, 0.45, 1] : [0.8, 0.12, 0.08, 1],
          metallicFactor: row,
          roughnessFactor: roughness,
        },
      });
      meshes.push({
        primitives: [{ attributes: { POSITION: 0, NORMAL: 1, TEXCOORD_0: 2 }, indices: 3, material: materials.length - 1 }],
      });
      nodes.push({
        mesh: meshes.length - 1,
        translation: [(col - (COLUMNS - 1) / 2) * 0.55, 0.26 + row * 0.6, 0],
      });
    }
  }

  const base64 = btoa(Array.from(bin, (b) => String.fromCharCode(b)).join(''));
  const json = {
    asset: { version: '2.0', generator: 'lighttemple material grid' },
    scene: 0,
    scenes: [{ nodes: nodes.map((_, i) => i) }],
    nodes,
    meshes,
    materials,
    buffers: [{ byteLength: total, uri: `data:application/octet-stream;base64,${base64}` }],
    bufferViews: views,
    accessors: [
      { bufferView: 0, componentType: 5126, count: vertexCount, type: 'VEC3', min, max },
      { bufferView: 1, componentType: 5126, count: vertexCount, type: 'VEC3' },
      { bufferView: 2, componentType: 5126, count: vertexCount, type: 'VEC2' },
      { bufferView: 3, componentType: 5125, count: indices.length, type: 'SCALAR' },
    ],
  };
  return new TextEncoder().encode(JSON.stringify(json)).buffer as ArrayBuffer;
}
