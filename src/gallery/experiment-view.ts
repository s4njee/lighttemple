/**
 * Hosts one experiment: canvas + parameter panel + notes, and owns the
 * render loop and lifecycle. Returns a disposer for the router.
 */

import { getGpu, WebGpuUnavailableError } from '../gpu/context';
import { ShaderCompileError, withErrorScope } from '../gpu/shader';
import { Surface } from '../gpu/surface';
import type { Experiment, ExperimentMeta } from '../experiments/types';
import { ParamsPanel } from '../ui/params';
import { PointerTracker } from '../ui/pointer';
import { FrameStats } from '../ui/stats';
import { EPIC_LABELS } from './registry';
import { h } from './dom';
import { experimentHash, replaceHash } from './route';
import { reportStatus } from './status';

const MAX_DT = 1 / 20;

export function mountExperiment(root: HTMLElement, meta: ExperimentMeta, initialParams = new URLSearchParams()): () => void {
  document.title = `${meta.title} · Light Temple`;
  reportStatus({ state: 'loading', slug: meta.slug, frames: 0 });

  const canvas = h('canvas', { class: 'stage-canvas', 'aria-label': `${meta.title}, interactive WebGPU canvas` });
  const errorBox = h('div', { class: 'stage-error', hidden: '' });
  const stage = h('div', { class: 'stage' }, canvas, errorBox);
  const panelBody = h('div', { class: 'panel-params' });
  const notes = h(
    'section',
    { class: 'panel-notes' },
    h('h2', {}, 'How it works'),
    ...meta.notes.map((n) => h('p', {}, n)),
  );
  const panel = h(
    'aside',
    { class: 'panel' },
    h(
      'header',
      { class: 'panel-header' },
      h('a', { class: 'back', href: '#/' }, '← Gallery'),
      h('p', { class: 'eyebrow' }, EPIC_LABELS[meta.epic]),
      h('h1', {}, meta.title),
      h('p', { class: 'summary' }, meta.summary),
      h('ul', { class: 'tags' }, ...meta.tags.map((t) => h('li', {}, t))),
    ),
    panelBody,
    notes,
  );
  const view = h('div', { class: 'experiment' }, stage, panel);
  root.replaceChildren(view);

  let disposed = false;
  let failed = false;
  let raf = 0;
  const cleanups: (() => void)[] = [];

  const fail = (err: unknown) => {
    if (failed || disposed) return;
    failed = true;
    cancelAnimationFrame(raf);
    console.error(err);
    reportStatus({ state: 'error', slug: meta.slug, frames: 0, error: String(err) });
    errorBox.hidden = false;
    canvas.hidden = true;
    let title = 'Something went wrong';
    let body = err instanceof Error ? err.message : String(err);
    if (err instanceof WebGpuUnavailableError) title = 'WebGPU is not available';
    if (err instanceof ShaderCompileError) {
      title = `Shader error in ${err.label}`;
      body = err.report;
    }
    errorBox.replaceChildren(h('h2', {}, title), h('pre', {}, body));
  };

  (async () => {
    const gpu = await getGpu((info) => {
      if (!disposed && info.reason !== 'destroyed') fail(new Error(`The GPU device was lost: ${info.message}. Reload to continue.`));
    });
    if (disposed) return;

    const surface = new Surface(canvas, gpu.device, gpu.presentationFormat);
    cleanups.push(() => surface.destroy());
    const pointer = new PointerTracker(canvas);
    cleanups.push(() => pointer.destroy());
    const params = new ParamsPanel(panelBody, initialParams);
    cleanups.push(() => params.destroy());
    const stats = new FrameStats(stage);
    cleanups.push(() => stats.destroy());

    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const mod = await meta.load();
    if (disposed) return;

    const experiment: Experiment = await withErrorScope(gpu.device, () =>
      mod.default({
        gpu,
        device: gpu.device,
        surface,
        format: gpu.presentationFormat,
        params,
        pointer: pointer.state,
        reducedMotion,
      }),
    );
    cleanups.unshift(() => experiment.dispose());
    if (disposed) {
      for (const c of cleanups) c();
      return;
    }

    // Host-level quality controls, shared by every experiment.
    const quality = params.add(
      {
        renderScale: { value: 1, min: 0.25, max: 1, step: 0.05, label: 'Render scale' },
        maxPixelRatio: { value: 2, options: { '1×': 1, '1.5×': 1.5, '2×': 2, '3×': 3 }, label: 'Max pixel ratio' },
      },
      'Renderer',
    );
    const applyQuality = () => {
      surface.renderScale = quality.renderScale;
      surface.maxPixelRatio = quality.maxPixelRatio;
      surface.update();
    };
    applyQuality();

    // Keep the URL in sync with non-default params so any state can be shared.
    let urlTimer = 0;
    params.onChange(() => {
      applyQuality();
      clearTimeout(urlTimer);
      urlTimer = window.setTimeout(() => !disposed && replaceHash(experimentHash(meta.slug, params.serialize())), 150);
    });
    cleanups.push(() => clearTimeout(urlTimer));
    params.button('Copy link to this state', () => {
      const url = new URL(experimentHash(meta.slug, params.serialize()), location.href).href;
      navigator.clipboard?.writeText(url).catch(() => {});
      replaceHash(experimentHash(meta.slug, params.serialize()));
    });
    params.button('Reset parameters', () => params.reset());

    cleanups.push(surface.onResize((size) => experiment.resize?.(size)));
    experiment.resize?.(surface.size);

    // Pause when scrolled out of view (matters once experiments are embedded in long pages).
    let visible = true;
    const io = new IntersectionObserver(([entry]) => (visible = entry.isIntersecting));
    io.observe(canvas);
    cleanups.push(() => io.disconnect());

    let time = 0;
    let frame = 0;
    let last = performance.now();
    const tick = (now: number) => {
      if (disposed || failed) return;
      raf = requestAnimationFrame(tick);
      const dt = Math.min(MAX_DT, Math.max(0, (now - last) / 1000));
      last = now;
      if (!visible) return;
      time += dt;
      pointer.update(dt);
      try {
        experiment.frame({ time, dt, frame, size: surface.size });
      } catch (err) {
        fail(err);
        return;
      }
      frame++;
      stats.push(dt, now, `${surface.size.width}×${surface.size.height}`);
      if (frame % 10 === 0 || frame < 10) reportStatus({ state: 'running', slug: meta.slug, frames: frame });
    };
    raf = requestAnimationFrame(tick);
  })().catch(fail);

  return () => {
    disposed = true;
    cancelAnimationFrame(raf);
    for (const c of cleanups) c();
    cleanups.length = 0;
  };
}
