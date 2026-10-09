import { defineConfig, type Plugin } from 'vite';

/**
 * Dev-only: make every experiment module self-accepting. When an experiment
 * (or any shader/module it imports) changes, the new factory is handed to the
 * gallery host, which remounts the experiment in place: same GPU device, same
 * params (they live in the URL), no page reload.
 */
function experimentHotSwap(): Plugin {
  return {
    name: 'lighttemple:experiment-hot-swap',
    apply: 'serve',
    transform(code, id) {
      const m = id.match(/\/src\/experiments\/([\w-]+)\/index\.ts$/);
      if (!m) return;
      const slug = JSON.stringify(m[1]);
      return `${code}
if (import.meta.hot) {
  import.meta.hot.accept((mod) => {
    if (mod) window.dispatchEvent(new CustomEvent('lighttemple:hot', { detail: { slug: ${slug}, factory: mod.default } }));
  });
}
`;
    },
  };
}

export default defineConfig({
  // Relative base so the static build can be hosted from any sub-path (e.g. GitHub Pages).
  base: './',
  plugins: [experimentHotSwap()],
  build: {
    target: 'es2022',
  },
});
