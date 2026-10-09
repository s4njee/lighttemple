/**
 * Adapter/device acquisition and capability reporting.
 *
 * One device is shared across the whole gallery; experiments create and destroy
 * their own resources on it. If the device is lost we surface it to the UI and
 * let the user reload rather than trying to transparently rebuild every experiment.
 */

export interface GpuCapabilities {
  /** Optional features we asked for and actually got. */
  features: ReadonlySet<GPUFeatureName>;
  /** Adapter description, best effort (may be empty for privacy reasons). */
  adapterInfo: { vendor: string; architecture: string; description: string };
  /** Whether rgba16float can be used as a render target with blending (always true in core WebGPU). */
  hdrRenderable: boolean;
  /** Whether float32 textures can be filtered (useful for VSM/EVSM shadow maps). */
  float32Filterable: boolean;
  /** Whether GPU timestamp queries are available for the perf HUD. */
  timestampQuery: boolean;
}

export interface GpuContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  /** The canvas format preferred by the platform (bgra8unorm or rgba8unorm). */
  presentationFormat: GPUTextureFormat;
  caps: GpuCapabilities;
}

export class WebGpuUnavailableError extends Error {
  constructor(
    message: string,
    readonly reason: 'no-api' | 'no-adapter' | 'no-device',
  ) {
    super(message);
    this.name = 'WebGpuUnavailableError';
  }
}

/** Features that unlock better versions of some techniques. None are required. */
const OPTIONAL_FEATURES: GPUFeatureName[] = [
  'float32-filterable',
  'timestamp-query',
  'rg11b10ufloat-renderable',
  'depth32float-stencil8',
];

let pending: Promise<GpuContext> | null = null;

export function getGpu(onLost?: (info: GPUDeviceLostInfo) => void): Promise<GpuContext> {
  pending ??= init(onLost).catch((err) => {
    pending = null; // allow a retry after a failure
    throw err;
  });
  return pending;
}

async function init(onLost?: (info: GPUDeviceLostInfo) => void): Promise<GpuContext> {
  if (!('gpu' in navigator) || !navigator.gpu) {
    throw new WebGpuUnavailableError(
      'This browser does not expose navigator.gpu. Try a recent Chrome, Edge, Safari 26+ or Firefox 141+, and make sure the page is served over HTTPS or localhost.',
      'no-api',
    );
  }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    throw new WebGpuUnavailableError(
      'WebGPU is available but no suitable GPU adapter was found. Hardware acceleration may be disabled or the GPU may be blocklisted.',
      'no-adapter',
    );
  }

  const requiredFeatures = OPTIONAL_FEATURES.filter((f) => adapter.features.has(f));
  let device: GPUDevice;
  try {
    device = await adapter.requestDevice({ requiredFeatures });
  } catch (err) {
    throw new WebGpuUnavailableError(`Failed to create a GPU device: ${String(err)}`, 'no-device');
  }

  device.lost.then((info) => {
    pending = null;
    console.error(`[gpu] device lost (${info.reason}): ${info.message}`);
    onLost?.(info);
  });

  device.addEventListener('uncapturederror', (event) => {
    console.error('[gpu] uncaptured error:', (event as GPUUncapturedErrorEvent).error.message);
  });

  const features = new Set(requiredFeatures);
  const info = adapter.info;
  return {
    adapter,
    device,
    presentationFormat: navigator.gpu.getPreferredCanvasFormat(),
    caps: {
      features,
      adapterInfo: {
        vendor: info?.vendor ?? '',
        architecture: info?.architecture ?? '',
        description: info?.description ?? '',
      },
      hdrRenderable: true,
      float32Filterable: features.has('float32-filterable'),
      timestampQuery: features.has('timestamp-query'),
    },
  };
}
