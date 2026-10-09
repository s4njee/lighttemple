import { defineConfig } from 'vite';

export default defineConfig({
  // Relative base so the static build can be hosted from any sub-path (e.g. GitHub Pages).
  base: './',
  build: {
    target: 'es2022',
  },
});
