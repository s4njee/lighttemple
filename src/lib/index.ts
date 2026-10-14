/**
 * Light Temple library entry (M3 story 3.7): run one experiment on a canvas the caller owns,
 * with no gallery, panel or DOM around it. The experiment declares its parameters through the
 * headless sink; a host that wants a panel mounts one itself.
 */
import { getGpu } from '../gpu/context';
import { withErrorScope } from '../gpu/shader';
import { Surface } from '../gpu/surface';
import { HeadlessParams } from '../ui/headless-params';
import { PointerTracker } from '../ui/pointer';
import { TargetSurface } from './target-surface';
import type { GpuContext } from '../gpu/context';
import type { SurfaceSize } from '../gpu/surface';
import type { ParamSink } from '../ui/params';
import type { PointerState } from '../ui/pointer';
import type { Experiment, ExperimentFactory, ExperimentMeta } from '../experiments/types';

export { HeadlessParams } from '../ui/headless-params';
export { TargetSurface } from './target-surface';
export type { TargetSurfaceOptions } from './target-surface';
export { hexToLinear, linearToSrgb, srgbToLinear } from '../gpu/color';
export type { ParamSink, ParamSchema, ParamValues } from '../ui/params';
export type { RenderSurface, SurfaceSize } from '../gpu/surface';
export type {
  Experiment,
  ExperimentContext,
  ExperimentFactory,
  ExperimentMeta,
  FrameInfo,
} from '../experiments/types';

const MAX_DT = 1 / 20;

export interface MountExperimentOptions {
  /** Initial parameter overrides (e.g. decoded from a URL). */
  params?: URLSearchParams;
  /** Defaults to the visitor's `prefers-reduced-motion`. */
  reducedMotion?: boolean;
  /** Start playing the render loop. Default true. */
  playing?: boolean;
}

export interface ExperimentHandle {
  readonly canvas: HTMLCanvasElement;
  /** The declared parameter groups and their live values. */
  readonly params: HeadlessParams;
  pause(): void;
  resume(): void;
  dispose(): void;
}

/** Mounts an experiment factory on a caller-owned canvas. */
export async function mountExperiment(
  canvas: HTMLCanvasElement,
  factory: ExperimentFactory,
  options: MountExperimentOptions = {},
): Promise<ExperimentHandle> {
  const gpu = await getGpu(() => {});
  const surface = new Surface(canvas, gpu.device, gpu.presentationFormat);
  const pointer = new PointerTracker(canvas);
  const params = new HeadlessParams(options.params);
  const reducedMotion =
    options.reducedMotion ?? window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const experiment: Experiment = await withErrorScope(gpu.device, () =>
    factory({
      gpu,
      device: gpu.device,
      surface,
      format: gpu.presentationFormat,
      params,
      pointer: pointer.state,
      reducedMotion,
    }),
  );

  const stopResize = surface.onResize((size) => experiment.resize?.(size));
  experiment.resize?.(surface.size);

  let visible = true;
  const observer = new IntersectionObserver(([entry]) => (visible = entry?.isIntersecting ?? true));
  observer.observe(canvas);

  let disposed = false;
  let playing = options.playing ?? true;
  let raf = 0;
  let time = 0;
  let frame = 0;
  let last = performance.now();

  const tick = (now: number) => {
    if (disposed) return;
    raf = requestAnimationFrame(tick);
    const dt = Math.min(MAX_DT, Math.max(0, (now - last) / 1000));
    last = now;
    if (!visible || !playing) return;
    time += dt;
    pointer.update(dt);
    experiment.frame({ time, dt, frame, size: surface.size });
    frame += 1;
  };
  raf = requestAnimationFrame(tick);

  return {
    canvas,
    params,
    pause() {
      playing = false;
    },
    resume() {
      playing = true;
      last = performance.now();
    },
    dispose() {
      disposed = true;
      cancelAnimationFrame(raf);
      observer.disconnect();
      stopResize();
      experiment.dispose();
      pointer.destroy();
      surface.destroy();
    },
  };
}

/** Loads an experiment by its metadata's `load()` and mounts it. */
export async function mountMeta(
  canvas: HTMLCanvasElement,
  meta: ExperimentMeta,
  options: MountExperimentOptions = {},
): Promise<ExperimentHandle> {
  const { default: factory } = await meta.load();
  return mountExperiment(canvas, factory, options);
}

export interface TargetExperimentOptions {
  device: GPUDevice;
  /** Caller-owned canvas, exposed to the experiment as `surface.canvas`. */
  canvas: HTMLCanvasElement;
  /** Format of the caller's render target. */
  format: GPUTextureFormat;
  /** Initial target size in physical pixels. */
  size: SurfaceSize;
  /** Returns the caller's target view for the frame the experiment is about to draw. */
  getTarget: () => GPUTextureView;
  /** Parameter sink the experiment declares into. Defaults to a headless sink. */
  params?: ParamSink;
  /** Initial URL overrides when no explicit sink is given. */
  paramsInitial?: URLSearchParams;
  reducedMotion?: boolean;
  /** Neutral when absent; pass a live state to keep pointer-driven lighting. */
  pointer?: Readonly<PointerState>;
  /** A full GPU context, for experiments that read `ctx.gpu`. */
  gpu?: GpuContext;
}

export interface TargetExperiment {
  readonly experiment: Experiment;
  readonly surface: TargetSurface;
  readonly params: ParamSink;
  /** Updates the reported target size and notifies the experiment's `resize`. */
  setSize(size: SurfaceSize): void;
}

const NEUTRAL_POINTER: PointerState = {
  uv: [0.5, 0.5],
  smoothUv: [0.5, 0.5],
  velocity: [0, 0],
  inside: false,
  down: false,
  idle: Infinity,
};

/**
 * Builds an experiment that draws into a caller-owned target rather than a canvas swapchain.
 * The host owns the loop: it calls `setSize` when its target changes and `experiment.frame(info)`
 * once per frame, having pointed the surface at that frame's target through `getTarget`. This is
 * the "target-based render path" a HoloGPU `EffectDefinition` uses (M3 story 3.7).
 */
export async function createTargetExperiment(
  factory: ExperimentFactory,
  options: TargetExperimentOptions,
): Promise<TargetExperiment> {
  const surface = new TargetSurface({
    canvas: options.canvas,
    format: options.format,
    size: options.size,
    getTarget: options.getTarget,
  });
  const params: ParamSink = options.params ?? new HeadlessParams(options.paramsInitial);
  const gpu =
    options.gpu ??
    ({ device: options.device, presentationFormat: options.format } as unknown as GpuContext);

  const experiment = await withErrorScope(options.device, () =>
    factory({
      gpu,
      device: options.device,
      surface,
      format: options.format,
      params,
      pointer: options.pointer ?? NEUTRAL_POINTER,
      reducedMotion: options.reducedMotion ?? false,
    }),
  );

  surface.onResize((size) => experiment.resize?.(size));
  experiment.resize?.(surface.size);

  return {
    experiment,
    surface,
    params,
    setSize: (size) => surface.setSize(size),
  };
}
