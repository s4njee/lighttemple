# Light Temple

An experimental gallery of practical **WebGPU lighting and shadow** effects for building aesthetic websites. Each experiment is live and tweakable, with notes on how it works and what it costs.

See **[plan.md](./plan.md)** for the roadmap (epics, stories, sequencing).

## Experiments

| Experiment | Technique |
| --- | --- |
| **Pointer Light** (`#/x/pointer-light-2d`) | 2.5D SDF scene, cursor-following area light, ray-marched soft *and finite* shadows, two coloured lights |
| **Pointer Light** headline mode | Any text, rendered to a signed distance field at runtime (Canvas 2D + jump flooding), extruded and lit |
| **Sun & Soft Shadows** (`#/x/shadow-map-pcf`) | Directional shadow map with hard / PCF / Poisson / PCSS filtering, PBR materials, MSAA or TAA |
| **Studio Lighting** (`#/x/studio-ibl`) | glTF model lit by image-based lighting (procedural studio/sky or HDRI), soft shadow catcher + contact AO, drag & drop your own `.glb` / `.hdr` |

## Getting started

```sh
npm install
npm run dev        # http://localhost:5173
npm run build      # typecheck + static build to dist/
npm run smoke      # build, then render every experiment headlessly (SwiftShader WebGPU)
npm run new -- my-effect --title "My Effect" --epic shadows   # scaffold + register an experiment
node scripts/capture.mjs .smoke/caps "#/x/studio-ibl?model=grid"  # screenshot specific states (after a build)
```

In `npm run dev`, editing an experiment or any shader it uses hot-swaps it in place without a page reload.

Every parameter change is mirrored into the URL (`#/x/shadow-map-pcf?mode=1&bloom=0.15`), so any state is a shareable link.

Requires a WebGPU-capable browser (Chrome/Edge 113+, Safari 26+, Firefox 141+ on Windows).

The smoke test uses `playwright-core` with the Chromium found by Playwright's registry; set `CHROMIUM_PATH` to point at a different binary. Screenshots are written to `.smoke/`.

## Layout

```
src/
  gpu/           device/context, canvas surface, shader helpers + cache, typed uniforms, geometry, target pool, mipmaps
  post/          shared HDR post chain (bloom, exposure, tonemapping, vignette, grain) and temporal AA
  ibl/           .hdr loader, environment baking (prefiltered specular, SH9, DFG LUT), IBL shading
  gltf/          minimal glTF 2.0 loader + material shader chunk
  sdf/           runtime text -> signed distance field (jump flooding)
  shaders/       shared WGSL chunks: colour & noise, fullscreen triangle, shadow filters, PBR BRDF, SDF primitives
  scene/         orbit camera
  ui/            params panel, pointer tracker, frame stats
  gallery/       router views, experiment host, registry
  experiments/   one folder per experiment (index.ts + shader.wgsl)
public/assets/  CC0 model + HDRI (see public/assets/CREDITS.md)
scripts/smoke.mjs           headless render test
scripts/capture.mjs         screenshot arbitrary gallery states
scripts/new-experiment.mjs  experiment generator
```
