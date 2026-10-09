/**
 * Typed uniform blocks with WGSL layout computed on the CPU.
 *
 * Declare fields once in TypeScript, then splice `block.wgslStruct` into the
 * shader. Field order, alignment and padding are guaranteed to match, so there
 * are no hand-maintained byte offsets anywhere.
 *
 *   const u = new UniformBlock(device, 'Frame', { viewProj: 'mat4x4f', time: 'f32', tint: 'vec3f' });
 *   u.set('time', t).set('tint', [1, 0.8, 0.6]).upload();
 */

export type WgslScalarType = 'f32' | 'u32' | 'i32';
export type WgslFieldType = WgslScalarType | 'vec2f' | 'vec3f' | 'vec4f' | 'vec2u' | 'vec4u' | 'mat3x3f' | 'mat4x4f';

interface FieldInfo {
  type: WgslFieldType;
  /** Offset in 4-byte words. */
  offset: number;
  /** Number of meaningful 4-byte components. */
  count: number;
  kind: 'f' | 'u' | 'i';
}

// [alignment, size] in bytes per the WGSL spec (uniform address space).
const LAYOUT: Record<WgslFieldType, [number, number]> = {
  f32: [4, 4],
  u32: [4, 4],
  i32: [4, 4],
  vec2f: [8, 8],
  vec2u: [8, 8],
  vec3f: [16, 12],
  vec4f: [16, 16],
  vec4u: [16, 16],
  mat3x3f: [16, 48],
  mat4x4f: [16, 64],
};

export type FieldValue = number | ArrayLike<number>;

interface StructLayout<S> {
  fields: Record<keyof S, FieldInfo>;
  byteSize: number;
  wgslStruct: string;
}

function layoutStruct<S extends Record<string, WgslFieldType>>(name: string, schema: S): StructLayout<S> {
  let offset = 0;
  let maxAlign = 16; // uniform structs are 16-byte aligned
  const fields = {} as Record<keyof S, FieldInfo>;
  const decl: string[] = [];
  for (const [key, type] of Object.entries(schema) as [keyof S & string, WgslFieldType][]) {
    const [align, size] = LAYOUT[type];
    maxAlign = Math.max(maxAlign, align);
    offset = Math.ceil(offset / align) * align;
    fields[key] = {
      type,
      offset: offset / 4,
      count: size / 4,
      kind: type.endsWith('u') || type === 'u32' ? 'u' : type === 'i32' ? 'i' : 'f',
    };
    decl.push(`  ${key}: ${type},`);
    offset += size;
  }
  return {
    fields,
    byteSize: Math.max(16, Math.ceil(offset / maxAlign) * maxAlign),
    wgslStruct: `struct ${name} {\n${decl.join('\n')}\n};`,
  };
}

/** The WGSL `struct` for a schema, without allocating a buffer (for shared shader code). */
export function uniformStruct(name: string, schema: Record<string, WgslFieldType>): string {
  return layoutStruct(name, schema).wgslStruct;
}

export class UniformBlock<S extends Record<string, WgslFieldType>> {
  readonly buffer: GPUBuffer;
  readonly byteSize: number;
  /** `struct <name> { ... }` declaration matching this block's layout. */
  readonly wgslStruct: string;

  private readonly fields: Record<keyof S, FieldInfo>;
  private readonly data: ArrayBuffer;
  private readonly f32: Float32Array;
  private readonly u32: Uint32Array;
  private readonly i32: Int32Array;
  private dirty = true;

  constructor(
    private readonly device: GPUDevice,
    readonly name: string,
    schema: S,
  ) {
    const layout = layoutStruct(name, schema);
    this.byteSize = layout.byteSize;
    this.fields = layout.fields;
    this.wgslStruct = layout.wgslStruct;
    this.data = new ArrayBuffer(this.byteSize);
    this.f32 = new Float32Array(this.data);
    this.u32 = new Uint32Array(this.data);
    this.i32 = new Int32Array(this.data);
    this.buffer = device.createBuffer({
      label: `uniforms:${name}`,
      size: this.byteSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  set(key: keyof S, value: FieldValue): this {
    const f = this.fields[key];
    const view = f.kind === 'u' ? this.u32 : f.kind === 'i' ? this.i32 : this.f32;
    if (typeof value === 'number') {
      view[f.offset] = value;
    } else if (f.type === 'mat3x3f' && value.length === 9) {
      // Tightly packed 3x3 -> columns padded to vec4.
      for (let c = 0; c < 3; c++) for (let r = 0; r < 3; r++) view[f.offset + c * 4 + r] = value[c * 3 + r];
    } else {
      const n = Math.min(f.count, value.length);
      for (let i = 0; i < n; i++) view[f.offset + i] = value[i];
    }
    this.dirty = true;
    return this;
  }

  /** Writes to the GPU buffer if anything changed since the last upload. */
  upload(): void {
    if (!this.dirty) return;
    this.device.queue.writeBuffer(this.buffer, 0, this.data);
    this.dirty = false;
  }

  destroy(): void {
    this.buffer.destroy();
  }
}
