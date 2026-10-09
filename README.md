# Light Temple

An experimental gallery of practical **WebGPU lighting and shadow** effects for building aesthetic websites. Each experiment is live and tweakable, with notes on how it works and what it costs.

See **[plan.md](./plan.md)** for the roadmap (epics, stories, sequencing).

## Experiments

| Experiment | Technique |
| --- | --- |
| **Pointer Light** (`#/x/pointer-light-2d`) | 2.5D SDF scene, cursor-following area light, ray-marched soft *and finite* shadows, two coloured lights |
| **Sun & Soft Shadows** (`#/x/shadow-map-pcf`) | Directional shadow map with hard / PCF / Poisson / PCSS filtering, bias controls, debug views |

## Getting started

```sh
npm install
npm run dev        # http://localhost:5173
npm run build      # typecheck + static build to dist/
npm run smoke      # build, then render every experiment headlessly (SwiftShader WebGPU)
```

Requires a WebGPU-capable browser (Chrome/Edge 113+, Safari 26+, Firefox 141+ on Windows).

The smoke test uses `playwright-core` with the Chromium found by Playwright's registry; set `CHROMIUM_PATH` to point at a different binary. Screenshots are written to `.smoke/`.

## Layout

```
src/
  gpu/           device/context, canvas surface, shader helpers, typed uniforms, geometry
  shaders/       shared WGSL chunks: colour pipeline & noise, fullscreen triangle, shadow filters
  scene/         orbit camera
  ui/            params panel, pointer tracker, frame stats
  gallery/       router views, experiment host, registry
  experiments/   one folder per experiment (index.ts + shader.wgsl)
scripts/smoke.mjs  headless render test
```
