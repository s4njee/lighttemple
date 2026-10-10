/**
 * Runtime text → signed distance field.
 *
 * Text is rasterised with Canvas 2D (so any CSS font works, including the
 * page's own web fonts, and there is no offline MSDF/atlas toolchain), then
 * converted on the GPU with the jump flooding algorithm. The result is an
 * rgba16float texture whose red channel is the signed distance to the glyph
 * outline in *pixels* (negative inside), ready for extrusion, soft shadows,
 * glows and outlines.
 *
 * Trade-off vs. MSDF: single-channel SDFs round off sharp corners when
 * magnified far beyond their raster size. For lighting effects (soft
 * shadows, bevels, glows) that is invisible; for crisp giant glyph edges,
 * rasterise at a larger size.
 */

import { JumpFlood } from './jump-flood';

export interface TextSdfOptions {
  text: string;
  /** CSS font-family list, e.g. '"Iowan Old Style", Georgia, serif'. */
  family: string;
  /** CSS font-weight. Default 700. */
  weight?: number | string;
  /** Raster font size in px (texture resolution scales with it). Default 160. */
  size?: number;
  /** Empty margin around the glyphs in px, which bounds how far the field reaches. Default size * 0.35. */
  padding?: number;
}

export interface TextSdfLetter {
  char: string;
  /** Index into the original string. */
  index: number;
  /** Horizontal extent in texels. */
  start: number;
  end: number;
}

export interface TextSdf {
  /** rgba16float, r = signed distance in pixels. */
  texture: GPUTexture;
  width: number;
  height: number;
  padding: number;
  /** Baseline y in texels. */
  baseline: number;
  /** Pen origin x in texels (where fillText started; the DOM's text start). */
  originX: number;
  /** Font-wide ascent/descent in texels (CSS line-box metrics, not ink). */
  fontAscent: number;
  fontDescent: number;
  /** Raster font size in px (texels per CSS px = size / CSS font-size). */
  size: number;
  /** Non-whitespace letters with their horizontal extents. */
  letters: TextSdfLetter[];
  destroy(): void;
}

export class TextSdfGenerator {
  private readonly jfa: JumpFlood;

  constructor(private readonly device: GPUDevice) {
    this.jfa = new JumpFlood(device);
  }

  async generate(opts: TextSdfOptions): Promise<TextSdf> {
    const size = opts.size ?? 160;
    const padding = Math.round(opts.padding ?? size * 0.35);
    const fontSpec = `${opts.weight ?? 700} ${size}px ${opts.family}`;
    try {
      await document.fonts.load(fontSpec, opts.text);
    } catch {
      // Fall back to whatever the browser resolves.
    }

    // 1. Rasterise.
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d')!;
    ctx.font = fontSpec;
    const m = ctx.measureText(opts.text || ' ');
    const left = Math.ceil(m.actualBoundingBoxLeft);
    const glyphW = Math.max(1, Math.ceil(m.actualBoundingBoxLeft + m.actualBoundingBoxRight));
    const glyphH = Math.max(1, Math.ceil(m.actualBoundingBoxAscent + m.actualBoundingBoxDescent));
    canvas.width = glyphW + padding * 2;
    canvas.height = glyphH + padding * 2;
    ctx.font = fontSpec; // resizing resets context state
    ctx.fillStyle = '#fff';
    ctx.textBaseline = 'alphabetic';
    const originX = padding + left;
    const baseline = padding + Math.ceil(m.actualBoundingBoxAscent);
    ctx.fillText(opts.text, originX, baseline);
    const { width, height } = canvas;

    // Per-letter horizontal extents (texels), from cumulative advances. Good enough
    // to give each letter its own material; kerned overlaps split down the middle.
    const letters: TextSdfLetter[] = [];
    const chars = [...opts.text];
    let prefix = '';
    let x0 = originX;
    chars.forEach((ch, index) => {
      prefix += ch;
      const x1 = originX + ctx.measureText(prefix).width;
      if (ch.trim()) letters.push({ char: ch, index, start: x0, end: x1 });
      x0 = x1;
    });

    const d = this.device;
    const mask = d.createTexture({
      label: 'text-mask',
      size: [width, height],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    d.queue.copyExternalImageToTexture({ source: canvas }, { texture: mask }, [width, height]);
    const sdf = d.createTexture({
      label: 'text-sdf',
      size: [width, height],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });

    // 2. Jump flood.
    const encoder = d.createCommandEncoder({ label: 'text-sdf' });
    this.jfa.encode(encoder, mask, sdf);
    d.queue.submit([encoder.finish()]);
    await d.queue.onSubmittedWorkDone();
    mask.destroy();
    this.jfa.trim(); // text sizes vary; don't keep scratch around

    return {
      texture: sdf,
      width,
      height,
      padding,
      baseline,
      originX,
      fontAscent: m.fontBoundingBoxAscent ?? m.actualBoundingBoxAscent,
      fontDescent: m.fontBoundingBoxDescent ?? m.actualBoundingBoxDescent,
      size,
      letters,
      destroy: () => sdf.destroy(),
    };
  }

  destroy(): void {
    this.jfa.destroy();
  }
}
