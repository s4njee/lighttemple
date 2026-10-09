import './styles/main.css';
import { mountExperiment, registerHotFactory } from './gallery/experiment-view';
import { mountIndex } from './gallery/index-view';
import { findExperiment } from './gallery/registry';
import { parseRoute } from './gallery/route';

const root = document.getElementById('app')!;
let unmount: () => void = () => {};

function route(): void {
  unmount();
  const r = parseRoute(location.hash);
  const meta = r.view === 'experiment' ? findExperiment(r.slug) : undefined;
  if (r.view === 'experiment' && !meta) {
    location.replace('#/');
    return;
  }
  document.body.dataset.view = meta ? 'experiment' : 'index';
  unmount = meta && r.view === 'experiment' ? mountExperiment(root, meta, r.params) : mountIndex(root);
  window.scrollTo(0, 0);
}

window.addEventListener('hashchange', route);
route();

// Dev: an experiment module was hot-updated; remount it in place if it is showing.
window.addEventListener('lighttemple:hot', (e) => {
  const { slug, factory } = (e as CustomEvent).detail;
  registerHotFactory(slug, factory);
  const r = parseRoute(location.hash);
  if (r.view === 'experiment' && r.slug === slug) route();
});
