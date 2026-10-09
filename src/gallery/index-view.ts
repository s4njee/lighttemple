import { h } from './dom';
import { EPIC_LABELS, experiments } from './registry';
import { reportStatus } from './status';

export function mountIndex(root: HTMLElement): () => void {
  document.title = 'Light Temple · WebGPU lighting & shadow experiments';
  reportStatus({ state: 'index', frames: 0 });

  const supported = 'gpu' in navigator && !!navigator.gpu;
  const cards = experiments.map((e) =>
    h(
      'li',
      {},
      h(
        'a',
        { class: 'card', href: `#/x/${e.slug}` },
        h('div', { class: 'card-art', style: `background:${e.cardBackground}` }),
        h(
          'div',
          { class: 'card-body' },
          h('p', { class: 'eyebrow' }, EPIC_LABELS[e.epic]),
          h('h2', {}, e.title),
          h('p', {}, e.summary),
          h('ul', { class: 'tags' }, ...e.tags.map((t) => h('li', {}, t))),
        ),
      ),
    ),
  );

  root.replaceChildren(
    h(
      'main',
      { class: 'index' },
      h(
        'header',
        { class: 'hero' },
        h('p', { class: 'eyebrow' }, 'Research gallery'),
        h('h1', {}, 'Light Temple'),
        h(
          'p',
          { class: 'lede' },
          'Practical WebGPU lighting and shadow techniques for building atmospheric websites. Each experiment is live and tweakable, with notes on how it works and what it costs.',
        ),
        supported
          ? h('p', { class: 'support ok' }, 'WebGPU detected')
          : h(
              'p',
              { class: 'support bad' },
              'WebGPU is not available in this browser. You can browse the notes, but the experiments will not run.',
            ),
      ),
      h('ul', { class: 'grid' }, ...cards),
      h(
        'footer',
        { class: 'foot' },
        h('p', {}, `${experiments.length} experiments · roadmap in plan.md`),
      ),
    ),
  );
  return () => {};
}
