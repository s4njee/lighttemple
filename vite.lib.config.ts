// Library build (M3 story 3.7, packaged in M4 story 4.4): emit plain ESM with WGSL inlined as
// strings (`?raw`), so an experiment can be consumed without Vite. Run with `npm run build:lib`.
import { existsSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

const root = fileURLToPath(new URL('.', import.meta.url));
const experimentsDir = fileURLToPath(new URL('./src/experiments', import.meta.url));

const effectEntries = Object.fromEntries(
  readdirSync(experimentsDir, { withFileTypes: true })
    .filter(
      (entry) =>
        entry.isDirectory() &&
        existsSync(
          fileURLToPath(new URL(`./src/experiments/${entry.name}/index.ts`, import.meta.url)),
        ),
    )
    .map((entry) => [
      `effects/${entry.name}/index`,
      fileURLToPath(new URL(`./src/experiments/${entry.name}/index.ts`, import.meta.url)),
    ]),
);

export default defineConfig({
  build: {
    outDir: 'lib',
    emptyOutDir: true,
    target: 'es2022',
    sourcemap: true,
    minify: false,
    lib: {
      entry: {
        index: fileURLToPath(new URL('./src/lib/index.ts', import.meta.url)),
        ...effectEntries,
      },
      formats: ['es'],
    },
    rollupOptions: {
      // Keep the one runtime dependency external; consumers install it.
      external: (id) => !id.startsWith('.') && !id.startsWith('/') && !id.startsWith('\0'),
    },
  },
  root,
});
