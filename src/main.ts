import './styles/main.css';
import { mountExperiment } from './gallery/experiment-view';
import { mountIndex } from './gallery/index-view';
import { findExperiment } from './gallery/registry';

/**
 * Hash routing keeps the build fully static (no server rewrites needed):
 *   #/            gallery index
 *   #/x/<slug>    a single experiment
 */
const root = document.getElementById('app')!;
let unmount: () => void = () => {};

function route(): void {
  unmount();
  const match = location.hash.match(/^#\/x\/([\w-]+)/);
  const meta = match ? findExperiment(match[1]) : undefined;
  if (match && !meta) {
    location.replace('#/');
    return;
  }
  document.body.dataset.view = meta ? 'experiment' : 'index';
  unmount = meta ? mountExperiment(root, meta) : mountIndex(root);
  window.scrollTo(0, 0);
}

window.addEventListener('hashchange', route);
route();
