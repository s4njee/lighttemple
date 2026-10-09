/**
 * A canvas bound to the shared device, with device-pixel-exact resizing.
 *
 * Pixel ratio is capped (default 2) because fill-rate is the dominant cost of
 * full-screen lighting effects, and 3x phone screens rarely benefit from it.
 */

export interface SurfaceSize {
  /** Backing-store size in physical pixels. */
  width: number;
  height: number;
  /** Size in CSS pixels. */
  cssWidth: number;
  cssHeight: number;
  /** Effective pixel ratio after the cap/scale is applied. */
  pixelRatio: number;
}

export interface SurfaceOptions {
  maxPixelRatio?: number;
  /** Extra multiplier on top of the pixel ratio (adaptive-quality hook). */
  renderScale?: number;
  alphaMode?: GPUCanvasAlphaMode;
}

export class Surface {
  readonly context: GPUCanvasContext;
  size: SurfaceSize = { width: 1, height: 1, cssWidth: 1, cssHeight: 1, pixelRatio: 1 };
  maxPixelRatio: number;
  renderScale: number;

  private observer: ResizeObserver;
  private devicePixels: { w: number; h: number } | null = null;
  private listeners = new Set<(size: SurfaceSize) => void>();

  constructor(
    readonly canvas: HTMLCanvasElement,
    readonly device: GPUDevice,
    readonly format: GPUTextureFormat,
    options: SurfaceOptions = {},
  ) {
    this.maxPixelRatio = options.maxPixelRatio ?? 2;
    this.renderScale = options.renderScale ?? 1;
    const ctx = canvas.getContext('webgpu');
    if (!ctx) throw new Error('Could not get a WebGPU canvas context.');
    this.context = ctx;
    this.context.configure({ device, format, alphaMode: options.alphaMode ?? 'opaque' });

    this.observer = new ResizeObserver((entries) => {
      const entry = entries[entries.length - 1];
      const dp = entry.devicePixelContentBoxSize?.[0];
      this.devicePixels = dp ? { w: dp.inlineSize, h: dp.blockSize } : null;
      this.update();
    });
    try {
      this.observer.observe(canvas, { box: 'device-pixel-content-box' });
    } catch {
      this.observer.observe(canvas); // Safari: no device-pixel-content-box
    }
    this.update();
  }

  /** Re-evaluates the backing-store size. Called automatically on resize. */
  update(): void {
    const rect = this.canvas.getBoundingClientRect();
    const cssWidth = Math.max(1, rect.width);
    const cssHeight = Math.max(1, rect.height);
    const dpr = window.devicePixelRatio || 1;
    const ratio = Math.min(dpr, this.maxPixelRatio) * this.renderScale;
    // Prefer the exact device-pixel size when the browser reports it and no cap applies.
    const exact = this.devicePixels && ratio === dpr;
    const limit = this.device.limits.maxTextureDimension2D;
    const width = Math.min(limit, Math.max(1, exact ? this.devicePixels!.w : Math.round(cssWidth * ratio)));
    const height = Math.min(limit, Math.max(1, exact ? this.devicePixels!.h : Math.round(cssHeight * ratio)));
    if (width === this.size.width && height === this.size.height) return;
    this.canvas.width = width;
    this.canvas.height = height;
    this.size = { width, height, cssWidth, cssHeight, pixelRatio: width / cssWidth };
    for (const l of this.listeners) l(this.size);
  }

  onResize(listener: (size: SurfaceSize) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  currentTexture(): GPUTexture {
    return this.context.getCurrentTexture();
  }

  destroy(): void {
    this.observer.disconnect();
    this.listeners.clear();
    this.context.unconfigure();
  }
}
