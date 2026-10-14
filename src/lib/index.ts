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
import type { Experiment, ExperimentFactory, ExperimentMeta } from '../experiments/types';

export { HeadlessParams } from '../ui/headless-params';
export { hexToLinear, linearToSrgb, srgbToLinear } from '../gpu/color';
export type { ParamSink, ParamSchema, ParamValues } from '../ui/params';
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
