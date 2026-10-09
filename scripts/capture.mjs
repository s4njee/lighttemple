#!/usr/bin/env node
/**
 * Captures screenshots of specific gallery states (headless SwiftShader WebGPU).
 * Serves the production build, so run `npm run build` first.
 *
 *   node scripts/capture.mjs <out-dir> "<hash>" ["<hash>" ...]
 *   node scripts/capture.mjs .smoke/caps "#/x/studio-ibl?model=grid&background=page"
 *
 * Options via env: FRAMES (default 12), WIDTH/HEIGHT (default 1280x800), CANVAS_ONLY=1.
 */

import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { chromium } from 'playwright-core';
import { preview } from 'vite';

const [outDir, ...hashes] = process.argv.slice(2);
if (!outDir || hashes.length === 0) {
  console.error('usage: node scripts/capture.mjs <out-dir> "<hash>" ["<hash>" ...]');
  process.exit(1);
}
const FRAMES = Number(process.env.FRAMES ?? 12);
const viewport = { width: Number(process.env.WIDTH ?? 1280), height: Number(process.env.HEIGHT ?? 800) };

await mkdir(outDir, { recursive: true });
const server = await preview({ preview: { port: 4190, strictPort: false }, logLevel: 'warn' });
const base = server.resolvedUrls?.local[0] ?? 'http://localhost:4190/';
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH || undefined,
  args: [
    '--enable-unsafe-webgpu',
    '--enable-features=Vulkan',
    '--use-vulkan=swiftshader',
    '--use-webgpu-adapter=swiftshader',
    '--use-angle=swiftshader',
    '--disable-vulkan-surface',
  ],
});

let failed = 0;
try {
  const page = await browser.newPage({ viewport });
  const errors = [];
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
  page.on('pageerror', (e) => errors.push(String(e)));
  for (const [i, hash] of hashes.entries()) {
    errors.length = 0;
    await page.goto('about:blank');
    await page.goto(`${base}${hash}`);
    const status = await page
      .waitForFunction(
        (n) => {
          const s = window.__lighttemple;
          return s && (s.state === 'error' || (s.state === 'running' && s.frames >= n)) ? s : null;
        },
        FRAMES,
        { timeout: 180_000, polling: 200 },
      )
      .then((h) => h.jsonValue())
      .catch(() => ({ state: 'timeout' }));
    const file = join(outDir, `${String(i).padStart(2, '0')}.png`);
    const canvas = process.env.CANVAS_ONLY ? await page.$('canvas') : null;
    await (canvas ?? page).screenshot({ path: file });
    const ok = status.state === 'running' && errors.length === 0;
    if (!ok) failed++;
    console.log(`${ok ? 'ok  ' : 'FAIL'} ${file}  ${hash}${status.error ? `\n  ${status.error}` : ''}`);
    for (const e of errors) console.log(`  console: ${e}`);
  }
} finally {
  await browser.close();
  await new Promise((r) => server.httpServer.close(r));
}
process.exit(failed ? 1 : 0);
