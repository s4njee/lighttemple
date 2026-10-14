/**
 * A small set of lit meshes ("props") for the 3D shadow scenes (E3).
 *
 * Every prop is one entry in a storage buffer of `Object`s (model matrix, albedo, roughness,
 * metallic), indexed in the shaders by `instance_index`, so a scene binds the buffer once and
 * draws each prop with `firstInstance` set to its index. Splice `PROP_WGSL` into the shader.
 */

import { mat4, type Mat4 } from 'wgpu-matrix';
import { hexToLinear } from '../gpu/color';
import type { Mesh } from '../gpu/geometry';

export interface Prop {
  mesh: Mesh;
  /** Albedo, sRGB hex. */
  color: string;
  /** Perceptual roughness 0..1. */
  roughness: number;
  metallic?: number;
  /** Free per-prop value for the scene's own use (a material id, an emissive strength, …). */
  extra?: number;
  /** Model matrix at animation time `t` (seconds). Omit for the identity. */
  transform?: (t: number) => Mat4;
  /** Leave out of shadow passes (a floor that only receives, a light's own bulb). */
  castsShadow?: boolean;
}

/** Bytes per prop: mat4x4f + 2 × vec4f. */
export const PROP_STRIDE = 96;

export const PROP_WGSL = /* wgsl */ `
struct Object {
  model: mat4x4f,
  // rgb = albedo (linear), a = perceptual roughness
  color: vec4f,
  // x = metallic, y = the prop's \`extra\` value
  material: vec4f,
};
`;

export class PropSet {
  readonly buffer: GPUBuffer;
  private readonly data: Float32Array;

  constructor(
    private readonly device: GPUDevice,
    readonly props: Prop[],
    label = 'props',
  ) {
    this.data = new Float32Array((props.length * PROP_STRIDE) / 4);
    this.buffer = device.createBuffer({
      label,
      size: Math.max(this.data.byteLength, PROP_STRIDE),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  /** Re-evaluate every transform and material at time `t` and upload them. */
  update(t: number): void {
    this.props.forEach((p, i) => {
      const o = (i * PROP_STRIDE) / 4;
      this.data.set(p.transform ? p.transform(t) : mat4.identity(), o);
      this.data.set([...hexToLinear(p.color), p.roughness, p.metallic ?? 0, p.extra ?? 0, 0, 0], o + 16);
    });
    this.device.queue.writeBuffer(this.buffer, 0, this.data);
  }

  /** Draw every prop (or only shadow casters), each as instance `i` so the shader can find it. */
  draw(pass: GPURenderPassEncoder, { castersOnly = false } = {}): void {
    this.props.forEach((p, i) => {
      if (castersOnly && p.castsShadow === false) return;
      pass.setVertexBuffer(0, p.mesh.vertexBuffer);
      pass.setIndexBuffer(p.mesh.indexBuffer, 'uint32');
      pass.drawIndexed(p.mesh.indexCount, 1, 0, 0, i);
    });
  }

  destroy(): void {
    this.buffer.destroy();
  }
}
