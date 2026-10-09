/**
 * Radiance .hdr (RGBE) decoding and upload as a filterable rgba16float texture.
 *
 * Supports the common "new" run-length encoding and flat scanlines, with the
 * standard `-Y height +X width` orientation (what Poly Haven and most tools write).
 */

export interface HdrImage {
  width: number;
  height: number;
  /** Linear RGBA float32, alpha = 1. */
  data: Float32Array;
}

export function parseHdr(buffer: ArrayBuffer): HdrImage {
  const bytes = new Uint8Array(buffer);
  let pos = 0;
  const readLine = () => {
    let s = '';
    while (pos < bytes.length && bytes[pos] !== 0x0a) s += String.fromCharCode(bytes[pos++]);
    pos++; // skip \n
    return s;
  };

  const magic = readLine();
  if (!magic.startsWith('#?')) throw new Error('Not a Radiance .hdr file (missing #? header).');
  for (let line = readLine(); line !== ''; line = readLine()) {
    if (line.startsWith('FORMAT=') && line !== 'FORMAT=32-bit_rle_rgbe') {
      throw new Error(`Unsupported .hdr format: ${line}`);
    }
    if (pos >= bytes.length) throw new Error('Truncated .hdr header.');
  }
  const res = readLine().match(/^-Y (\d+) \+X (\d+)$/);
  if (!res) throw new Error('Unsupported .hdr orientation (expected "-Y h +X w").');
  const height = Number(res[1]);
  const width = Number(res[2]);

  const rgbe = new Uint8Array(width * height * 4);
  const scan = new Uint8Array(width * 4);
  for (let y = 0; y < height; y++) {
    const rle = width >= 8 && width < 32768 && bytes[pos] === 2 && bytes[pos + 1] === 2 && !(bytes[pos + 2] & 0x80);
    if (!rle) {
      // Flat (uncompressed) scanline.
      rgbe.set(bytes.subarray(pos, pos + width * 4), y * width * 4);
      pos += width * 4;
      continue;
    }
    if (((bytes[pos + 2] << 8) | bytes[pos + 3]) !== width) throw new Error('Corrupt .hdr scanline header.');
    pos += 4;
    // Each of the 4 channels is run-length encoded separately.
    for (let c = 0; c < 4; c++) {
      let x = 0;
      while (x < width) {
        let count = bytes[pos++];
        if (count > 128) {
          count -= 128;
          const v = bytes[pos++];
          for (let i = 0; i < count; i++) scan[(x++) * 4 + c] = v;
        } else {
          for (let i = 0; i < count; i++) scan[(x++) * 4 + c] = bytes[pos++];
        }
      }
    }
    rgbe.set(scan, y * width * 4);
  }

  const data = new Float32Array(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    const e = rgbe[i * 4 + 3];
    const f = e === 0 ? 0 : Math.pow(2, e - 136); // 2^(e-128) / 256
    data[i * 4] = rgbe[i * 4] * f;
    data[i * 4 + 1] = rgbe[i * 4 + 1] * f;
    data[i * 4 + 2] = rgbe[i * 4 + 2] * f;
    data[i * 4 + 3] = 1;
  }
  return { width, height, data };
}

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

/** IEEE float32 -> float16 bits (round-to-nearest, clamps to ±65504, flushes tiny values). */
export function toHalf(v: number): number {
  f32[0] = v;
  const x = u32[0];
  const sign = (x >>> 16) & 0x8000;
  const exp = ((x >>> 23) & 0xff) - 127 + 15;
  const mant = x & 0x7fffff;
  if (exp >= 31) return sign | 0x7bff; // overflow (and inf/nan) -> max finite
  if (exp <= 0) {
    if (exp < -10) return sign;
    const m = (mant | 0x800000) >>> (1 - exp);
    return sign | ((m + 0x1000) >>> 13);
  }
  // Addition (not OR) so a rounding carry out of the mantissa bumps the exponent.
  const h = (exp << 10) + ((mant + 0x1000) >>> 13);
  return sign | Math.min(h, 0x7bff);
}

/** Uploads linear float RGBA as rgba16float (filterable on every WebGPU device). */
export function uploadHalfFloatTexture(device: GPUDevice, image: HdrImage, label = 'hdr'): GPUTexture {
  const halves = new Uint16Array(image.data.length);
  for (let i = 0; i < halves.length; i++) halves[i] = toHalf(image.data[i]);
  const texture = device.createTexture({
    label,
    size: [image.width, image.height],
    format: 'rgba16float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeTexture({ texture }, halves, { bytesPerRow: image.width * 8 }, [image.width, image.height]);
  return texture;
}
