import type { RenderSurface, SurfaceSize } from '../gpu/surface';

export interface TargetSurfaceOptions {
  /** A caller-owned canvas, exposed to the experiment as `surface.canvas`. */
  canvas: HTMLCanvasElement;
  /** Format of the caller's target; the experiment renders its final image to it. */
  format: GPUTextureFormat;
  /** Initial backing-store size in physical pixels. */
  size: SurfaceSize;
  /** The caller's target view for the current frame. */
  getTarget: () => GPUTextureView;
}

/**
 * A `RenderSurface` with no swapchain of its own: `currentTexture()` hands back a view of the
 * caller's render target, so an experiment that draws to `surface.currentTexture()` draws into
 * someone else's frame. This is the bridge the HoloGPU adapter needs (M3 story 3.7): HoloGPU owns
 * the canvas and the frame, and this surface lets a Light Temple experiment land inside it without
 * the experiment knowing anything changed.
 */
export class TargetSurface implements RenderSurface {
  readonly format: GPUTextureFormat;
  size: SurfaceSize;
  private readonly listeners = new Set<(size: SurfaceSize) => void>();

  constructor(private readonly options: TargetSurfaceOptions) {
    this.format = options.format;
    this.size = options.size;
  }

  get canvas(): HTMLCanvasElement {
    return this.options.canvas;
  }

  /** Called by the host when its target size changes; notifies the experiment's `resize`. */
  setSize(size: SurfaceSize): void {
    if (size.width === this.size.width && size.height === this.size.height) {
      this.size = size;
      return;
    }
    this.size = size;
    for (const listener of this.listeners) listener(size);
  }

  onResize(listener: (size: SurfaceSize) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  currentTexture(): GPUTexture {
    const target = this.options.getTarget();
    // A GPUTexture is only ever used here for `.createView()`; hand back the caller's view.
    return { createView: () => target } as unknown as GPUTexture;
  }
}
