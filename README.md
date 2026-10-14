# Light Temple

An experimental gallery of practical **WebGPU lighting and shadow** effects for building aesthetic websites. Each experiment is live and tweakable, with notes on how it works and what it costs.

Part of the [holofx collection](https://github.com/s4njee/holofx) alongside
[hologpu](https://github.com/s4njee/hologpu),
[firetemple](https://github.com/s4njee/firetemple) and
[watertemple](https://github.com/s4njee/watertemple).

See **[plan.md](./plan.md)** for the roadmap (epics, stories, sequencing).

## Experiments

| Experiment | Technique |
| --- | --- |
| **Pointer Light** (`#/x/pointer-light-2d`) | 2.5D SDF scene or runtime headline, cursor area light, ray-marched soft *and finite* shadows |
| **Embossed Card** (`#/x/embossed-card`) | Procedural paper/steel/leather card, letterpress or emboss + foil, parallax occlusion mapping, relief self-shadows |
| **Holographic Card** (`#/x/holo-card`) | Tilting foil card (pointer or gyro): diffraction grating, thin-film interference, anisotropic GGX |
| **Glass Panels** (`#/x/glass-panels`) | Draggable frosted glass: bevel refraction, mip-based blur, dispersion, rim light |
| **Radiance Cascades** (`#/x/radiance-cascades`) | Paint lights and walls: real-time 2D global illumination (bilinear-fix cascades + JFA SDF), with a brute-force reference view |
| **Long Shadows** (`#/x/long-shadows`) | Flat-design long shadows via exact convex SDF sweeps |
| **Neon Sign** (`#/x/neon-sign`) | Neon tubes from any text, flicker, emission-pyramid light spill on bump-mapped brick |
| **Lit Typography** (`#/x/lit-type`) | Editable DOM heading rendered as aligned, ray-marched 3D letters (oblique projection), per-letter metals |
| **Sun & Soft Shadows** (`#/x/shadow-map-pcf`) | Directional shadow map with hard / PCF / Poisson / PCSS filtering, PBR materials, MSAA or TAA |
| **Studio Lighting** (`#/x/studio-ibl`) | glTF model lit by image-based lighting, soft shadow catcher + contact AO, drag & drop your own `.glb` / `.hdr` |

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

## Embed an experiment

The package is published as `@mk7s/lighttemple`:

```sh
npm install @mk7s/lighttemple
```

`mountExperiment` runs one experiment on a canvas you own, with no gallery or panel:

```ts
import { mountExperiment } from '@mk7s/lighttemple';
import factory from '@mk7s/lighttemple/effects/pointer-light-2d';

const handle = await mountExperiment(document.querySelector('canvas')!, factory);
handle.params.groups; // the parameter groups the experiment declared
handle.pause();
handle.dispose();
```

`createTargetExperiment` renders into a target your host owns — a HoloGPU
`render(frame, encoder, target)`, say — so an experiment can run as a hologpu
`EffectDefinition` (see [ADR-020](https://github.com/s4njee/hologpu/blob/main/docs/adr/020-one-effect-contract.md)).

Parameters are declared through a headless `ParamSink` (`HeadlessParams` in the library,
`ParamsPanel` in the gallery), so an experiment always records its schema and reads live values
without needing any DOM. Both entries ship with types.


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
  sdf/           jump-flood SDFs from any mask; runtime text -> SDF (with per-letter extents and font metrics)
  shaders/       shared WGSL: colour, noise, SDF primitives, PBR (incl. anisotropic), wave optics, shadow filters
  scene/         orbit camera
  ui/            params panel, pointer tracker, frame stats
  gallery/       router views, experiment host, registry
  experiments/   one folder per experiment (index.ts + shader.wgsl)
public/assets/  CC0 model + HDRI (see public/assets/CREDITS.md)
scripts/smoke.mjs           headless render test
scripts/capture.mjs         screenshot arbitrary gallery states
scripts/new-experiment.mjs  experiment generator
```

## Licence

MIT. See [LICENSE](./LICENSE). Third-party assets are credited in
[public/assets/CREDITS.md](./public/assets/CREDITS.md).
