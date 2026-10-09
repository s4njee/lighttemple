import './styles/main.css';
import { mountExperiment } from './gallery/experiment-view';
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
