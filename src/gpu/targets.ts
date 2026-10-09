/**
 * Size-keyed render-target cache.
 *
 * Ask for a target by name every frame; it is (re)created only when the
 * requested size or description changes. Replaces hand-written
 * "if size changed, destroy and recreate" code in every experiment.
 *
 *   const targets = new TargetPool(device);
 *   const hdr = targets.get('hdr', { size, format: 'rgba16float' });
 *   pass({ view: hdr.view, ... })
 */

export interface TargetDesc {
  /** Full-resolution size [width, height] in pixels; `scale` is applied on top. */
  size: readonly [number, number];
  format: GPUTextureFormat;
  /** Resolution multiplier, e.g. 0.5 for half-res. Defaults to 1. */
  scale?: number;
  sampleCount?: number;
  /** Defaults to RENDER_ATTACHMENT | TEXTURE_BINDING (just RENDER_ATTACHMENT when multisampled). */
  usage?: GPUTextureUsageFlags;
}

export interface Target {
  texture: GPUTexture;
  /** Default full view, cached. */
  view: GPUTextureView;
  width: number;
  height: number;
}

export class TargetPool {
  private targets = new Map<string, { key: string; target: Target }>();

  constructor(private readonly device: GPUDevice) {}

  get(name: string, desc: TargetDesc): Target {
    const scale = desc.scale ?? 1;
    const width = Math.max(1, Math.round(desc.size[0] * scale));
    const height = Math.max(1, Math.round(desc.size[1] * scale));
    const sampleCount = desc.sampleCount ?? 1;
    const usage =
      desc.usage ??
      (sampleCount > 1
        ? GPUTextureUsage.RENDER_ATTACHMENT
        : GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING);
    const key = `${width}x${height}:${desc.format}:${sampleCount}:${usage}`;

    const existing = this.targets.get(name);
    if (existing?.key === key) return existing.target;
    existing?.target.texture.destroy();

    const texture = this.device.createTexture({
      label: `target:${name}`,
      size: [width, height],
      format: desc.format,
      sampleCount,
      usage,
    });
    const target = { texture, view: texture.createView({ label: `target:${name}` }), width, height };
    this.targets.set(name, { key, target });
    return target;
  }

  /** Destroys one target (e.g. a feature was switched off). */
  release(name: string): void {
    this.targets.get(name)?.target.texture.destroy();
    this.targets.delete(name);
  }

  destroy(): void {
    for (const { target } of this.targets.values()) target.texture.destroy();
    this.targets.clear();
  }
}
