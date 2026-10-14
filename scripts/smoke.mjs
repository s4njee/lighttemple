#!/usr/bin/env node
/**
 * Headless smoke test: serves the production build, opens the gallery and every
 * experiment in Chromium with WebGPU on SwiftShader (software Vulkan), waits for
 * frames to render, and fails on shader/validation errors or console errors.
 * Screenshots land in .smoke/ for eyeballing.
 *
 *   npm run smoke                 # build + test
 *   npm run smoke -- --only shadow-map-pcf,studio-ibl   # just these experiments
 *   CHROMIUM_PATH=/path/to/chrome npm run smoke
 */

import { mkdir } from 'node:fs/promises';
import { parseArgs } from 'node:util';
import { chromium } from 'playwright-core';
import { preview } from 'vite';

const { values: args } = parseArgs({ options: { only: { type: 'string' } } });
const only = args.only ? new Set(args.only.split(',').map((s) => s.trim()).filter(Boolean)) : null;
const OUT = '.smoke';
const FRAMES = 8;
const TIMEOUT_MS = 60_000;

const server = await preview({ preview: { port: 4179, strictPort: false, open: false }, logLevel: 'warn' });
const base = server.resolvedUrls?.local[0] ?? 'http://localhost:4179/';
await mkdir(OUT, { recursive: true });

const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH || undefined,
  args: [
    '--enable-unsafe-webgpu',
    '--enable-features=Vulkan',
    '--use-vulkan=swiftshader',
    '--use-webgpu-adapter=swiftshader',
    // Without ANGLE on SwiftShader, presenting a WebGPU canvas drops the Dawn instance.
    '--use-angle=swiftshader',
    '--disable-vulkan-surface',
  ],
});

let failures = 0;
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  const consoleErrors = [];
  page.on('console', (m) => m.type() === 'error' && consoleErrors.push(m.text()));
  page.on('pageerror', (e) => consoleErrors.push(String(e)));

  await page.goto(base);
  await page.waitForSelector('.card');
  const all = await page.$$eval('a.card', (as) => as.map((a) => a.getAttribute('href').replace('#/x/', '')));
  const slugs = only ? all.filter((s) => only.has(s)) : all;
  for (const s of only ?? []) {
    if (!all.includes(s)) {
      failures++;
      console.log(`FAIL ${s} (not in the gallery)`);
    }
  }
  await page.screenshot({ path: `${OUT}/index.png` });
  console.log(`index: ${slugs.length} experiments`);

  for (const slug of slugs) {
    consoleErrors.length = 0;
    const started = Date.now();
    await page.goto(`${base}#/x/${slug}`);
    const status = await page
      .waitForFunction(
        (n) => {
          const s = window.__lighttemple;
          return s && (s.state === 'error' || (s.state === 'running' && s.frames >= n)) ? s : null;
        },
        FRAMES,
        { timeout: TIMEOUT_MS, polling: 100 },
      )
      .then((h) => h.jsonValue())
      .catch(() => ({ state: 'timeout' }));
    await page.screenshot({ path: `${OUT}/${slug}.png` });
    const ms = Date.now() - started;
    const ok = status.state === 'running' && consoleErrors.length === 0;
    if (!ok) failures++;
    console.log(`${ok ? 'PASS' : 'FAIL'} ${slug} (${status.state}, ${ms} ms)`);
    if (status.error) console.log(`  ${status.error.split('\n').join('\n  ')}`);
    for (const e of consoleErrors) console.log(`  console: ${e}`);
  }
} finally {
  await browser.close();
  await new Promise((r) => server.httpServer.close(r));
}

if (failures) {
  console.error(`${failures} experiment(s) failed`);
  process.exit(1);
}
console.log('all experiments rendered');
