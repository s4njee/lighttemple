# Light Temple: Plan

An experimental, research-oriented gallery of **practical WebGPU lighting and shadow effects** for building aesthetic websites. Every experiment is a live, tweakable page with notes on how it works, what it costs, and when to reach for it.

This document is the roadmap. It is organised as **epics → stories**, each story with acceptance criteria. Status markers:

- `[x]` done · `[~]` in progress / partial · `[ ]` not started
- Size: **S** (≤ ½ day) · **M** (1–2 days) · **L** (3–5 days) · **XL** (needs splitting before starting)

---

## 1. Goals & non-goals

**Goals**

1. **Practical first.** Prefer techniques that ship on real websites: they run at 60 fps on a 2020 integrated GPU at 1080p, degrade gracefully on phones, and don't need a 50 MB asset pipeline.
2. **Aesthetic.** Every experiment should look good by default, with art direction and not just a debug scene, because the point is to inspire site design.
3. **Explain.** Each experiment carries "how it works", its cost, its knobs, and its failure modes. Debug views (shadow masks, depth, normals, SDFs) are first-class.
4. **Reusable.** Techniques graduate from experiment code into small shared modules (`src/gpu`, `src/shaders`) that could later be extracted into an embeddable package.
5. **Raw WebGPU.** No engine dependency. Understanding the GPU work is the research. Keep runtime dependencies to a bare minimum (currently only `wgpu-matrix`).

**Non-goals (for now)**

- A general-purpose engine or scene graph.
- WebGL fallbacks for every experiment. A static poster/video fallback is enough (see E8-S6).
- Photoreal offline-quality GI. We care about believable and fast.

Publication and a shared effect contract are now planned rather than ruled out; see M3 and M4 in the [holofx backlog](../holofx/backlog.md).

## 2. Technical decisions

| Area | Decision | Why |
| --- | --- | --- |
| Build | Vite + TypeScript (strict), static output, relative `base` | Zero-config dev server, `?raw` WGSL imports, deployable to any static host / sub-path. |
| Routing | Hash routes: `#/` and `#/x/<slug>` | Fully static hosting, no server rewrites. |
| Shaders | Plain `.wgsl` files composed by string concatenation (`wgsl(...)`) | Readable error reports, no custom preprocessor to maintain. |
| Uniforms | `UniformBlock` declares fields once in TS and generates the matching WGSL `struct` | Eliminates hand-maintained byte offsets, the #1 source of WebGPU bugs. |
| Colour | Experiments render premultiplied linear HDR (`rgba16float`, alpha = coverage) → shared `PostProcess` (bloom, exposure, tonemap None/ACES/AgX, vignette, grain) → manual sRGB encode + triangular dither | Canvas formats are not sRGB; dither kills banding in the long, smooth gradients lighting produces; one post chain keeps every experiment consistent. |
| Experiments | Each one is a lazy-loaded module exporting an `ExperimentFactory` | Index stays tiny; experiments are isolated and disposable. |
| UI | Tiny in-house params panel (`src/ui/params.ts`), state mirrored into the URL hash | No dependency, styled to match, schema → live values object; any state is a shareable link. |
| Testing | Headless Chromium + SwiftShader WebGPU smoke test (`npm run smoke`) | Catches WGSL/validation errors and blank frames in CI without a GPU. |

**Browser targets:** Chrome/Edge 113+, Safari 26+, Firefox 141+ (Windows; other platforms as they ship). A clear message is shown where WebGPU is unavailable.

**Performance budget (per experiment, default settings):** ≤ 8 ms GPU at 1920×1080 on an integrated GPU (Intel Iris Xe / Apple M1 class); ≤ 16 ms at the capped pixel ratio on a mid-range phone. Every experiment must expose a quality knob that can meet the phone budget.

## 3. Experiment definition of done

An experiment is **stable** (vs. **prototype**) when it has:

- [ ] Good-looking defaults and respects `prefers-reduced-motion`.
- [ ] Parameters for every interesting knob, with hints on non-obvious ones.
- [ ] At least one debug view (mask / depth / normals / buffers).
- [ ] "How it works" notes covering technique, cost, and failure modes.
- [ ] Passes `npm run smoke` with no console errors.
- [ ] Meets the performance budget at default settings (measured with GPU timestamps once E9-S3 lands).
- [ ] A real thumbnail (E9-S1).
- [ ] Disposes all GPU resources (switching experiments 20× shows no growth in GPU memory).

---

## 4. Epics

| ID | Epic | Theme |
| --- | --- | --- |
| E0 | Foundation & infrastructure | Gallery shell, GPU core, tooling |
| E1 | Rendering toolkit | Shared building blocks: HDR, post chain, materials, assets |
| E2 | Interface lighting (2D / 2.5D) | Cursor lights, lit UI surfaces, the "website-native" effects |
| E3 | Real-time shadows | Every practical shadow technique, compared |
| E4 | Global illumination & ambient | AO, bounce light, area lights, environment lighting |
| E5 | Volumetrics & atmosphere | God rays, fog, caustics, sky |
| E6 | Stylised lighting | Toon, halftone, dither, hatching, gradient-mapped light |
| E7 | Many lights | Compute-driven clustered lighting, particles as lights |
| E8 | Web integration | DOM-driven lights/occluders, scroll, embeds, fallbacks |
| E9 | Gallery UX & documentation | Thumbnails, compare mode, sharing, perf HUD |
| E10 | Quality, performance & CI | Automated checks, device tiers, deployment |

---

### E0: Foundation & infrastructure

> Make adding a new experiment a 15-minute job, and make failures loud and legible.

- [x] **E0-S1: Project scaffold** (S). Vite + strict TS, `@webgpu/types`, `?raw` WGSL imports, relative base path.
  *AC:* `npm run dev`, `npm run build`, `npm run typecheck` work on a clean checkout.
- [x] **E0-S2: GPU context** (S). `src/gpu/context.ts`: single shared adapter/device, optional features (`float32-filterable`, `timestamp-query`, `rg11b10ufloat-renderable`, …) requested when available, capability report, device-lost handling, typed `WebGpuUnavailableError`.
  *AC:* Unsupported browsers get a human-readable explanation; device loss shows an in-page error.
- [x] **E0-S3: Surface & resize** (S). `src/gpu/surface.ts`: device-pixel-exact `ResizeObserver`, pixel-ratio cap (default 2), render-scale multiplier.
  *AC:* No blurry canvas on HiDPI; changing render scale live works.
- [x] **E0-S4: Experiment contract & host** (M). `ExperimentFactory → { frame, resize?, dispose }`; host owns loop, dt clamp, pause when off-screen, pointer, params, and errors.
  *AC:* Shader compile errors render in-page with a code excerpt; validation errors during init surface via error scopes.
- [x] **E0-S5: Typed uniforms** (S). `UniformBlock` computes WGSL alignment and emits the struct.
  *AC:* Two experiments use it with mixed `f32/u32/vec*/mat4x4f` fields with no manual offsets.
- [x] **E0-S6: Shared WGSL chunks** (S). `common.wgsl` (tonemappers, sRGB, IGN/hash noise, dithered display encode), `fullscreen.wgsl`, `shadow.wgsl`.
- [x] **E0-S7: Params panel** (S). Number/bool/color/select controls from a schema, reset, buttons.
- [x] **E0-S8: Gallery index + hash router** (S). Lazy-loaded registry, cards, experiment page with panel and notes.
- [x] **E0-S9: Headless smoke test** (M). `npm run smoke`: build, serve, render every experiment in SwiftShader WebGPU, fail on errors, save screenshots to `.smoke/`.
- [x] **E0-S10: First two experiments** (M). `pointer-light-2d` (E2) and `shadow-map-pcf` (E3) as end-to-end proofs of the foundation.
- [x] **E0-S11: Experiment template generator** (S). `npm run new -- <slug> [--title "…"] [--epic …]` scaffolds `index.ts` + `shader.wgsl` from a working pointer-light template on the HDR → post path, and registers it.
  *AC:* A freshly generated experiment type-checks and passes `npm run smoke` unmodified. Slug validation and duplicate detection work.
- [x] **E0-S12: Param state in URL** (S). Non-default params are serialised into the hash (`#/x/slug?intensity=2&color=ffd2a1`) on change (debounced `replaceState`, no history spam), parsed and clamped on load. Unknown keys are ignored. "Copy link to this state" button. Param keys are enforced unique per panel. *(Also enables E9-S4.)*

### E1: Rendering toolkit

> Shared building blocks that later epics lean on. Build each one *when the first experiment needs it*, not ahead of time.

- [x] **E1-S1: HDR render target + post chain** (M). `src/post/post.ts`: `PostProcess.create(ctx, defaults)` → `scene(size)` gives the `rgba16float` target, `render(encoder, out, { time, raw?, background? })` finishes the frame. Adds a standard **Post** params group (exposure, tonemap, bloom, radius, vignette, grain) to every experiment that uses it. That's opt-in per experiment, not host-level, so experiments can bypass it (`raw`) for debug views. Scene alpha is coverage: `background` composites the scene over a *display-space* colour after tonemapping, so 3D content can dissolve into a page colour exactly.
  *AC:* Both experiments use it, and neither does its own tonemapping anymore. ✔
- [x] **E1-S2: Bloom** (M). Up to 6-level pyramid: 13-tap CoD:AW downsample (Karis average on the first level against fireflies), 3×3 tent upsample with additive blending, energy-conserving `mix` with no threshold.
  *Remaining:* measure the ≤ 1 ms @1080p budget once E9-S3 GPU timings exist.
- [x] **E1-S3: Temporal accumulation / TAA-lite** (L). `src/post/taa.ts`: Halton(2,3) clip-space jitter, depth-based reprojection (closest depth in 3×3), 9-tap Catmull-Rom history, YCoCg variance clipping, Karis luma-weighted blending, motion-adaptive feedback, `frameNoiseOffset` for per-frame noise rotation. `post.render({ source })` consumes the resolved image. `shadow-map-pcf` has an *Anti-aliasing* switch (MSAA 4× / TAA / Off); with TAA its Poisson/PCSS grain converges to a smooth result.
  *Remaining:* no velocity buffer yet, so animated objects rely on the neighbourhood clip (fine for slow motion, ghosts on fast motion).
- [x] **E1-S4: PBR material model** (M). `src/shaders/brdf.wgsl`: metal/roughness, GGX + height-correlated Smith + Schlick, Lambert diffuse weighted by (1−F)(1−metallic), and `brdf_ambient` using Karis' analytic split-sum fit so a hemisphere "environment" gives plausible reflections before IBL exists. `shadow-map-pcf` uses it (copper torus, steel pole).
  *Remaining:* `pointer-light-2d` still uses normalised Blinn-Phong (fine for its matte look); the newer E2 experiments use the shared BRDF.
- [x] **E1-S5: Image-based lighting** (M). `src/ibl/`: Radiance `.hdr` (RGBE, RLE) decoder + float16 upload; procedural *studio* (softboxes) and *sky* (sun + gradient) environments; bake to a 512² env cube with mips → GGX-prefiltered specular cube (7 roughness mips, filtered importance sampling) → SH9 irradiance (one compute workgroup, read back to estimate the dominant light direction) → split-sum DFG LUT. `ibl.wgsl` shading includes multiple-scattering energy compensation. Bind group 1 by convention. Rendered by fragment passes (one face/mip per pass) rather than compute, which keeps it on core-only formats.
- [x] **E1-S6: glTF loader (minimal)** (M). `src/gltf/`: `.glb`/`.gltf` (data URIs or relative files), node hierarchies (matrix/TRS), triangle primitives with float or normalized-int attributes, generated normals when missing, metal/roughness materials with base colour, metal-rough, normal, occlusion and emissive textures (sRGB formats where required, mipmaps via `src/gpu/mipmaps.ts`), per-texture samplers, alpha MASK. Normal mapping uses supplied tangents, else a derivative-based cotangent frame. No skinning, morphs, animation or extensions (BLEND renders opaque).
- [x] **E1-S7: Render-target pool** (S). `src/gpu/targets.ts`: `TargetPool.get(name, { size, format, scale?, sampleCount? })` recreates a target only when its description changes. Used by the post chain (scene + bloom levels) and the shadow experiment (MSAA colour/depth).
- [x] **E1-S8: Pipeline/shader cache & hot reload** (S). `createShader` caches compiled modules per device by source, so shared code (post, TAA, IBL bake) compiles once per session. A dev-only Vite plugin makes every `src/experiments/*/index.ts` self-accepting. Editing an experiment or any shader/module it imports remounts it *in place* (same device, same params, no page reload). Changes outside that boundary fall back to a full reload, which still restores state from the URL.
- [x] **E1-S9: SDF toolkit** (M). `src/shaders/sdf.wgsl`: 2D (`sd_circle/box/round_box/segment/capsule/ring/equilateral_triangle`) and 3D (`sd_sphere3/box3/round_box3/torus3/capsule3/cylinder3/plane3`) primitives, `op_extrude`, and (smooth) boolean ops. **Text:** instead of MSDF (offline font/atlas toolchain), `src/sdf/text-sdf.ts` rasterises any CSS font with Canvas 2D and builds a signed distance field with a jump-flood compute pipeline seeded at *sub-pixel* edge positions. It's slightly rounder at extreme magnification, but works with page fonts and arbitrary runtime text, which is what E2-S3/E8-S2 need. `pointer-light-2d` gained a *Headline* scene and an editable text param (new `text` param type).

### E2: Interface lighting (2D / 2.5D)

> The most directly "website-native" category: lights that react to the cursor, scroll and UI, lighting flat or nearly flat content.

- [x] **E2-S1: Pointer light with SDF soft shadows** (M). `pointer-light-2d`: extruded SDF shapes or a runtime headline, bevel normals, ray-marched soft *and finite* shadows, two lights, debug views. Finished: one-pixel material blend + continuous crease occlusion remove the aliased "dashed" rims; shadow rays stop once they climb above every shape, so cost scales with shadow length; touch devices default to fewer steps, and the host defaults to 0.75 render scale on coarse pointers.
- [x] **E2-S2: Normal-mapped surfaces** (M). `embossed-card`: a business card in cotton paper (letterpress + ink), brushed steel, pebbled leather (+ gold foil) or matte stock, all procedural (`src/shaders/noise.wgsl`). A baked height field (text SDFs, border, logo pressed into material microstructure) is ray-cast with parallax occlusion mapping (linear search + secant refinement), normals from the height gradient, relief self-shadows toward the cursor light, studio IBL for metals, and an analytic desk shadow.
  *AC met:* reads as physically debossed/embossed at an angle under a moving light.
- [x] **E2-S3: Lit typography** (M). `lit-type`: a real `contenteditable` DOM heading (transparent ink, selectable, accessible) rendered underneath as ray-marched extruded letters from its *own computed font*. An oblique projection keeps the front faces locked to the DOM glyphs while the pointer reveals the sides. Per-character DOM `Range` rects correct each letter's position (rounding/kerning), per-letter materials (alternating metals / hues / gold), soft SDF shadows, SDF AO, IBL. `pointer-light-2d` keeps a 2.5D headline mode.
- [x] **E2-S4: Holographic / foil cards** (M). `holo-card`: spring-driven tilt from pointer or device orientation (with the iOS permission flow). Foil = diffraction grating (concentric grooves + random-orientation glitter flecks, grating equation over 3 orders → `spectral_rgb`), thin-film interference tint, and radially-brushed anisotropic GGX (`brdf_direct_aniso`), under a printed, laminated (clearcoat) artwork layer. Shared wave-optics in `src/shaders/optics.wgsl`.
- [x] **E2-S5: Glass panels** (M). `glass-panels`: lit background (blobs, grid, headline, soft drop shadows cast away from the cursor) → HDR mip chain. Draggable panels with circular-bevel normals, screen-space refraction along the tilt, RGB dispersion, 8-tap frosted blur one mip below the target, cursor glints, Fresnel rim and hairline, SDF "UI" content.
  *Limitation:* panels refract the background only, not other panels behind them.
- [x] **E2-S6: 2D radiance cascades GI** (L). `radiance-cascades`: paint lights/walls/erase, cursor lamp. Generic `JumpFlood` (`src/sdf/jump-flood.ts`) SDF of the scene. ~6 cascades (spacing 2ⁿ, 4ⁿ⁺¹ directions, 4× intervals) traced and merged top-down with the **bilinear fix**, sky term on the top cascade, fluence from cascade 0. A "Reference (64 rays)" view brute-forces the same field; the cascade result matched it to a few hundredths along measured scanlines.
  *AC:* the 60 fps @1080p check needs a real integrated GPU (E9-S3). Half-res GI is the default.
- [x] **E2-S7: Long-shadow / flat-design lighting** (S). `long-shadows`: letter tiles spell a word. Tile shadows are the *exact* sweep of a convex shape (ternary search over a convex distance function), so they're crisp and anti-aliased at any length. Glyph shadows, clipped to their tile, use dense uniform sampling. Light angle follows the pointer, and tonemapping is off for exact brand colours.
- [x] **E2-S8: Neon & emissive signage** (S). `neon-sign`: tubes from the text SDF (outline or solid), frame and moon icon. White-hot cores, glass highlights when unlit, mounting shadow. Half-res emission mip pyramid summed for ~1/r wall spill + gradient-derived light direction for bump-shaded brick/concrete. Per-letter flicker with one faulty letter (off under reduced motion).

### E3: Real-time shadows

> A side-by-side reference of practical shadow techniques, each with debug views and cost numbers.

- [x] **E3-S1: Directional shadow map + filters** (M). `shadow-map-pcf`: hard (HW 2×2), PCF grid, rotated Poisson, PCSS; depth + normal-offset bias; shadow-map overlay; radial background fade. *Finished:* the light frustum comes from `fitDirectionalShadow` (`src/scene/light-frustum.ts`), fitted to fixed scene bounds or tightly to the camera's frustum slice out to a shadow distance, with texel snapping on a toggle, a *Shadow texels* debug view and a camera-drift toggle that shows the shimmer with snapping off. Snapping fixes camera-motion shimmer only, not a rotating light. The split-screen compare stays with E9-S2.
- [x] **E3-S2: Variance / exponential shadow maps (VSM, EVSM)** (M). VSM and EVSM modes in `shadow-map-pcf`: moments written beside depth, a shared separable Gaussian (`src/post/blur-separable.ts`), Chebyshev visibility and light-bleeding reduction (`src/shaders/shadow-moments.wgsl`). rg/rgba32float with `float32-filterable`, else 16-bit (EVSM exponent clamped to 5.54); a *Precision* param shows the 16-bit result anywhere. The bleeding failure shows at azimuth −150° with the light still (the pole inside the torus's shadow), next to PCSS.
- [x] **E3-S3: Cascaded shadow maps** (L). `cascaded-shadows`: a low sun over a procedural wooded valley (warped heightfield, ~3k/6.5k/12k instanced props, per-pass chunk culling). 1–4 cascades in a `depth32float` 2D array, practical splits (λ), a stable sphere fit with texel snapping against a shimmering tight fit, frusta stretched toward the sun to hold every caster, lerp/dither blend bands and a fade at the shadow distance, per-cascade texel-scaled biases, PCF or a world-radius Poisson. Debug: cascade colours, visibility, texel checker, map thumbnails, freeze cascades.
- [x] **E3-S4: Omnidirectional point-light shadows** (M). `point-light-shadows` (Lantern Room): a swinging, flickering candle lantern in a cut-away room. Six-pass `depth32float` cube map; every filter tap picks its own face and rebuilds its reference depth, so filtering is seamless across faces. Poisson/PCSS in 3D with taps on the receiver's plane, a dual-paraboloid comparison mode, and debug views including the unfolded cube. Shared shape builders in `src/scene/shapes.ts`.
- [x] **E3-S5: Spot light shadows + cookies/gobos** (S). `spot-light-gobo`: a pointer-aimed spot with a cone-fitted perspective shadow map, linear-depth bias in receiver texels and perspective PCSS. Procedural, mipmapped cookies (leaves, blinds, window) that sway, with the mip level as projector focus. Debug: texel density (perspective aliasing), light pattern, light frustum, shadow-map and cookie overlays.
- [x] **E3-S6: Screen-space contact shadows** (S). In `shadow-map-pcf`: a single-sample depth prepass (the main pass is MSAA) and a jittered N-step march toward the sun (`src/shaders/contact-shadow.wgsl`), with a contact-mask debug view. Brings back the small cubes' shadows at a 512 map.
- [x] **E3-S7: SDF ray-marched soft shadows (3D)** (M). `sdf-soft-shadows`: a ray-marched sculpture garden with hard, classic and improved (closest-approach) SDF shadows, a compute-baked `rgba16float` 3D-texture SDF traced with trilinear filtering, and the same scene with a shadow map + PCSS, alone or in a split view. Debug: shadow mask, step heatmaps, a distance-field slice.
- [x] **E3-S8: Transparent-canvas shadow catcher** (M). `studio-ibl` composites its catcher over a page colour; `shadow-catcher` now does it over live HTML (a CSS gradient and a selectable article) on a truly transparent canvas. `ExperimentMeta.transparent` → `alphaMode: 'premultiplied'`, and the post chain gained `output: 'premultiplied'` (valid premultiplied alpha; shadows' alpha remapped in linear light) without changing its opaque output. PCSS page shadow + contact occlusion; text stays selectable through `pointer-events: none` plus an invisible drag handle over the object. The library mounts accept `transparent`. Debug: alpha, a premultiplied check, the shadow mask.
- [x] **E3-S9: Coloured / translucent shadows** (M). `coloured-shadows`: stained-glass lancets and tinted acrylic. An `rgba16float` transmittance map (multiplicative rgb, nearest translucent depth in alpha, depth-tested against the opaque map) beside the depth map, compared with stochastic transparency (grey, or per channel with three maps) converging under TAA. Limit: only the nearest translucent depth is known.
- [x] **E3-S10: Blob & baked shadows** (S). `blob-shadows`: height-aware blob decals; a ground lightmap baked progressively at load (jittered sun + cosine sky samples: soft shadow and AO) with blobs for dynamic objects; and a real-time 2048² shadow map, compared on one scene with each mode's cost. The low-tier fallback for E10-S4.
- [ ] **E3-S11: Ray-traced shadows (stretch)** (XL). Compute-shader BVH ray queries for a small scene. Research only; compare against PCSS. *Split before starting (§ sizes), and deferred past E3:*
  - [ ] **E3-S11a: BVH build** (M). CPU binned-SAH BVH over the scene's triangles, flattened into storage buffers; a debug view of node bounds.
  - [ ] **E3-S11b: Shadow rays** (M). One shadow ray per pixel from a depth prepass, traced through the BVH in a compute pass (stack in registers), any-hit early out, written to a visibility texture the lit pass reads.
  - [ ] **E3-S11c: Soft ray-traced shadows** (M). Jittered rays toward an area/angular light, temporally accumulated with E1-S3, compared against PCSS (E3-S1) on the same scene for quality and cost.

### E4: Global illumination & ambient

- [ ] **E4-S1: SSAO → GTAO** (M). Ground-truth AO with bent normals, half-res + bilateral upsample, temporal accumulation via E1-S3.
- [ ] **E4-S2: Analytic area lights (LTC)** (M). Rect and disc area lights with linearly transformed cosines: soft-box studio lighting for product shots.
- [ ] **E4-S3: Irradiance probes / SH lighting** (M). Baked (or compute-baked at load) SH probe grid for soft bounce light in a small room.
- [ ] **E4-S4: Reflective shadow maps** (M). One-bounce indirect from the sun via RSM sampling; cheap colour bleeding.
- [ ] **E4-S5: Screen-space GI / SSIL** (L). Visibility-bitmask SSIL or similar, half-res with temporal filtering.
- [ ] **E4-S6: 3D radiance cascades / voxel GI (stretch)** (XL). Research spike; split before starting. *Split, and deferred past E4:*
  - [ ] **E4-S6a: Scene voxelisation** (M). Conservative voxelisation of a small room into a 3D texture (albedo, emission, opacity) by a compute or triple-axis raster pass, re-run only when the scene changes; a slice/raymarch debug view.
  - [ ] **E4-S6b: Voxel cone-traced diffuse GI** (M). Mip the voxel volume (anisotropic or isotropic) and cone-trace a handful of diffuse cones per pixel at half resolution, temporally accumulated with E1-S3; compare against the SH probes of E4-S3.
  - [ ] **E4-S6c: 3D radiance cascades** (L). Probe cascades in world space (spacing ×2, rays ×4 per level) traced through the voxel volume and merged top-down, as the 3D sibling of E2-S6; compare quality, memory and cost with S6b.

### E5: Volumetrics & atmosphere

- [ ] **E5-S1: Screen-space god rays** (S). Radial blur of an occlusion mask from the light's screen position. The cheap classic.
- [ ] **E5-S2: Ray-marched volumetric light** (M). March the shadow map in view space with Henyey-Greenstein phase, blue-noise jitter, and temporal reprojection. Shafts through window blinds (pairs with E3-S5).
- [ ] **E5-S3: Froxel volumetric fog** (L). Compute-built froxel grid with per-froxel lighting and integration; supports multiple lights and local fog volumes.
- [ ] **E5-S4: Caustics** (M). Water caustics projected onto a floor from an animated normal field (light-space photon splat or analytic), plus a glass-object caustic.
- [ ] **E5-S5: Dust motes & light-catching particles** (S). GPU particles lit only inside light shafts, the "sunbeam through a window" mood.
- [ ] **E5-S6: Atmospheric sky** (M). Hillaire-style LUT sky (transmittance + multi-scattering) for sunrise/sunset hero backgrounds driven by scroll.

### E6: Stylised lighting

- [ ] **E6-S1: Toon / cel shading** (S). Banded diffuse with a ramp texture, rim light, inverted-hull or screen-space outlines, stylised shadow edges.
- [ ] **E6-S2: Halftone & dithered shadows** (S). Print-style halftone dots, ordered/blue-noise dither, and 1-bit "Obra Dinn" lighting driven by real shadow maps.
- [ ] **E6-S3: Hatching** (M). Tonal art maps for pencil/ink hatching that follow lighting.
- [ ] **E6-S4: Gradient-mapped light** (S). Map luminance through art-directed gradients (duotone/tritone brand palettes). Practical for on-brand sites.
- [ ] **E6-S5: Matcap & fake lighting** (S). Matcaps and hemisphere tricks as near-zero-cost lighting for low-tier devices.
- [ ] **E6-S6: Painterly / Kuwahara post** (M). Anisotropic Kuwahara filter over a lit scene for a painted look.

### E7: Many lights

- [ ] **E7-S1: Clustered forward lighting** (L). Compute-shader light binning into view-space clusters; 1k+ point lights at 60 fps. Cluster heatmap debug view.
- [ ] **E7-S2: Fireflies / lights-as-particles** (M). GPU particle system where each particle is a light, on top of E7-S1.
- [ ] **E7-S3: Light-linked text / UI glow** (S). Hundreds of tiny lights tracing a logo outline that light the surrounding scene.
- [ ] **E7-S4: Deferred vs. forward comparison** (M). Same scene, G-buffer deferred path vs. clustered forward, with timings.

### E8: Web integration

> Techniques that make the effects work *on real pages*. This is what separates this gallery from a generic graphics demo reel.

- [ ] **E8-S1: DOM elements as occluders** (M). Read `getBoundingClientRect()` of real HTML cards/buttons, upload them as SDF rounded-rects, and cast soft light/shadows *from the DOM* onto a WebGPU background. Scroll-synced.
  *AC:* Shadows track elements during scroll and resize with no visible lag (rAF-synchronised).
- [ ] **E8-S2: DOM text as occluders** (M). Read headings' text, font and rects from the DOM, build SDFs with `TextSdfGenerator` (E1-S9), and let real, selectable text cast shadows.
- [ ] **E8-S3: Scroll-driven lighting** (S). Map scroll progress to sun position, time of day, light intensity. Demo: a long-form article that "sets" as you read.
- [ ] **E8-S4: Embeddable component** (M). `<lt-light-surface>` web component wrapping an experiment with attributes for key params; usable outside the gallery.
- [ ] **E8-S5: Transparent overlays & blending with CSS** (S). Premultiplied-alpha canvases composited over HTML, `mix-blend-mode` interplay, pointer-events passthrough.
- [ ] **E8-S6: Graceful fallbacks** (S). Poster image / short video when WebGPU is missing, and when `prefers-reduced-motion` or `Save-Data` is set.
- [ ] **E8-S7: Lifecycle & battery** (S). Pause on `visibilitychange`/off-screen (done in host), throttle to 30 fps when idle, release GPU memory when hidden for a long time.
- [ ] **E8-S8: Multiple canvases, one device** (M). Several lit regions on one page sharing a device and pacing, without janking scroll.

### E9: Gallery UX & documentation

- [ ] **E9-S1: Real thumbnails** (S). Generate PNG/AVIF thumbnails from the smoke-test screenshots; replace CSS gradient placeholders. Optional hover-to-play short loops.
- [ ] **E9-S2: Compare mode** (M). Split-screen slider comparing two parameter presets (e.g. PCF vs. PCSS) in the same experiment.
- [ ] **E9-S3: GPU timing HUD** (M). Per-pass GPU timings via `timestamp-query` when available; CPU frame time otherwise.
- [ ] **E9-S4: Presets** (S). Named art-directed presets per experiment ("Golden hour", "Studio", "Noir"), shareable via E0-S12.
- [ ] **E9-S5: Filtering & search** (S). Filter the index by epic/tag/cost tier.
- [ ] **E9-S6: Long-form write-ups** (M per article). For the best techniques: diagrams, references, and "use it on your site" guidance.
- [ ] **E9-S7: Keyboard & screen-reader pass** (S). Focus states, labelled controls, canvas descriptions, panel collapsible on mobile.
- [ ] **E9-S8: Light theme / art direction pass** (S). Consistent visual identity across experiments.

### E10: Quality, performance & CI

- [ ] **E10-S1: CI workflow** (S). GitHub Actions: typecheck, build, smoke test (SwiftShader); upload `.smoke/` screenshots as artefacts.
- [ ] **E10-S2: Static deployment** (S). GitHub Pages (or Cloudflare Pages) on push to main; preview deploys for PRs.
- [ ] **E10-S3: Visual regression** (M). Deterministic frame mode (fixed time, fixed pointer) + image diff against approved baselines.
- [ ] **E10-S4: Device tiers & adaptive quality** (M). Classify devices (adapter info + a short GPU timing probe) into tiers; auto-pick render scale/steps/map sizes; dynamic resolution if frame time exceeds budget.
- [ ] **E10-S5: Resource leak check** (S). Smoke test cycles every experiment N times and asserts no growth in tracked GPU allocations.
- [ ] **E10-S6: Lint/format** (S). ESLint + Prettier config, and a WGSL formatting convention.

---

## 5. Suggested sequencing

1. **Milestone 1: Foundation** *(done)*. All of E0 and E1.
   **Interface lighting** *(done)*. All of E2: 10 experiments live.
2. **Milestone 2: Ship the gallery.** E10-S1, E10-S2, E9-S1. The site is public, tested and easy to extend.
3. **Milestone 3: Web-native lighting.** E8-S1, E8-S2, E3-S8 (on top of E1 and E2). These are the most immediately useful for real websites.
4. **Milestone 4: Shadow reference.** E3-S1 (finish), E3-S2, E3-S3, E3-S5, E3-S6, E9-S2, E9-S3. *E3 is done apart from the S11 stretch (split, deferred); 17 experiments live. E9-S2 and E9-S3 remain, so GPU timings are still unmeasured.*
5. **Milestone 5: Atmosphere & GI.** E5-S1, E5-S2, E4-S1, E4-S2 (all can lean on E1-S3 TAA and E1-S5 IBL).
6. **Milestone 6: Style & scale.** E6-*, E7-S1, E7-S2, E8-S4.

## 6. Open questions

- Should stable techniques be extracted into a publishable package (`@lighttemple/*`) or stay as copy-paste reference code?
- Do we want a WebGL2 fallback for the 2–3 most practical effects (E2-S1, E3-S8), or are posters enough?
- Thumbnail strategy: static frames vs. short looping video (size budget)?

## 7. Adding an experiment (current workflow)

1. `npm run new -- <slug> --title "Nice Title" --epic shadows` scaffolds `src/experiments/<slug>/` from a working template and registers it. (For a 3D, multi-pass, MSAA starting point, read `shadow-map-pcf`.)
2. Declare params with `ctx.params.add({...}, 'Group')` (keys are unique and become URL keys). Declare uniforms with `new UniformBlock(device, 'Name', {...})` and splice `block.wgslStruct` into the shader.
3. Render premultiplied linear HDR into `post.scene(size)`, then call `post.render(...)`. Use `raw: true` for debug views.
4. Fill in the summary, tags and notes in `src/gallery/registry.ts`.
5. Run `npm run dev`, then `npm run smoke` before committing.
