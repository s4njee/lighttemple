/**
 * Every experiment in the gallery. Code is lazy-loaded per experiment so the
 * index page stays tiny no matter how many effects we add.
 *
 * To add one: create src/experiments/<slug>/index.ts exporting a default
 * ExperimentFactory, then append an entry here.
 */

import type { Epic, ExperimentMeta } from '../experiments/types';

export const EPIC_LABELS: Record<Epic, string> = {
  'ui-lighting': 'Interface lighting',
  shadows: 'Shadows',
  'global-illumination': 'Global illumination',
  volumetrics: 'Volumetrics & atmosphere',
  stylized: 'Stylised lighting',
  'many-lights': 'Many lights',
  'web-integration': 'Web integration',
};

export const experiments: ExperimentMeta[] = [
  {
    slug: 'pointer-light-2d',
    title: 'Pointer Light',
    summary: 'A cursor-following area light casting soft, finite shadows from extruded 2D shapes.',
    epic: 'ui-lighting',
    tags: ['sdf', 'soft shadows', 'pointer', 'text', 'fullscreen'],
    status: 'prototype',
    cardBackground:
      'radial-gradient(circle at 62% 38%, #ffe2bd 0%, #c9b9a5 22%, #6c625a 58%, #2a2623 100%)',
    notes: [
      'The whole scene is one fullscreen fragment shader. Shapes are a 2D signed distance field (SDF), extruded to a fixed height with a smooth bevel, so every pixel knows its height and surface normal without any geometry.',
      'Shadows are ray-marched through the SDF toward each light. The tightest clearance-to-distance ratio along the ray gives a penumbra that widens with distance from the occluder, the classic Quilez soft-shadow trick. The ray also climbs toward the light, so once it clears the top of a shape it is no longer blocked. That is why shadows end, and why they lengthen as you lower the light.',
      'Lighting uses inverse-square falloff, regularised by the light radius, with Lambert diffuse and normalised Blinn-Phong. Output goes through AgX or ACES, then sRGB encoding with triangular dither so the long, smooth gradients do not band on 8-bit displays.',
      'The headline is real text: Canvas 2D rasterises it in any CSS font, then a jump-flood compute pass turns the coverage mask into a signed distance field with sub-pixel edge seeds. The shapes come from the shared SDF toolkit. Edit the text in the panel and the field rebuilds in a few milliseconds.',
      'Cost: one fullscreen pass with steps × lights SDF evaluations per pixel. At 48 steps this is comfortable on integrated GPUs at 1080p. Lower the steps or render scale for mobile.',
    ],
    load: () => import('../experiments/pointer-light-2d'),
  },
  {
    slug: 'shadow-map-pcf',
    title: 'Sun & Soft Shadows',
    summary: 'Directional shadow mapping with hard, PCF, Poisson and contact-hardening PCSS filters side by side.',
    epic: 'shadows',
    tags: ['shadow map', 'pcf', 'pcss', '3d', 'msaa'],
    status: 'prototype',
    cardBackground:
      'linear-gradient(160deg, #c9c5bf 0%, #b9b1a7 45%, #7d7268 46%, #a79c90 70%, #d6cfc6 100%)',
    notes: [
      'A depth-only pass renders the scene from the sun into a 32-bit depth texture through an orthographic projection. The main pass reprojects each fragment into that space and compares depths.',
      'Filters: "Hard" is a single hardware comparison, which with a linear comparison sampler is already 2×2 bilinear PCF. "PCF grid" averages a (2r+1)² block of those. "Poisson" takes 16 taps in a disc rotated per pixel by interleaved gradient noise, trading banding for fine grain. "PCSS" first searches for blockers, then scales the filter by receiver–blocker distance, so shadows are crisp at contact and soften with distance. Watch the thin pole.',
      'Materials use the shared metal/roughness PBR model (GGX, height-correlated Smith, Schlick Fresnel). The "environment" is just a sky/ground hemisphere, fed through Karis\' analytic split-sum fit, so the copper torus and steel pole read as metal without any cubemaps.',
      'Acne control combines a small constant depth bias with normal-offset sampling scaled by shadow-map texel size, which avoids most peter-panning. Shadows also fade out on surfaces turning away from the light.',
      'A radial fade is applied after tonemapping so it matches the clear colour exactly, and the floor dissolves into the background with no visible edge. Useful when a 3D scene has to sit inside a flat-coloured page section.',
    ],
    load: () => import('../experiments/shadow-map-pcf'),
  },
  {
    slug: 'studio-ibl',
    title: 'Studio Lighting',
    summary: 'A glTF product shot lit only by its environment, with a soft shadow catcher that melts into the page.',
    epic: 'global-illumination',
    tags: ['ibl', 'pbr', 'gltf', 'shadow catcher', 'hdr'],
    status: 'prototype',
    cardBackground: 'radial-gradient(ellipse at 50% 35%, #fff3e0 0%, #9c8b78 30%, #2b2622 70%, #141210 100%)',
    notes: [
      'Image-based lighting: the environment (a procedural studio with softboxes, a procedural sky, or a real HDRI) is baked once into a 512² cubemap with mips. From that come a GGX-prefiltered specular cube (one mip per roughness step, using filtered importance sampling so 96 samples per texel are enough), nine spherical-harmonic coefficients for diffuse irradiance (one compute workgroup), and the split-sum DFG lookup table.',
      'Shading adds multiple-scattering energy compensation (Fdez-Agüera), so rough metals keep their brightness instead of going dark. Compare the rough end of the metal row in the material grid.',
      'The floor is a shadow catcher: it renders nothing but darkness, blended over whatever is behind it, so it works on the environment and on a flat page colour. The soft shadow comes from the environment\'s dominant light, estimated from the SH band-1 vector, so rotating the environment swings the shadow. Contact occlusion comes from a depth map rendered looking up from under the floor: each texel stores how high the underside of the model is.',
      'The lantern is a CC0 Khronos sample model (textures shrunk to 1K JPEG, about 1 MB). Drag your own .glb or equirectangular .hdr onto the canvas to try them. Bake cost is a one-off few milliseconds on a discrete GPU. Per frame it is one model pass plus two small depth passes.',
    ],
    load: () => import('../experiments/studio-ibl'),
  },
  {
    slug: 'embossed-card',
    title: 'Embossed Card',
    summary: 'A letterpress, embossed or foil-stamped business card on paper, steel or leather, raked by your cursor.',
    epic: 'ui-lighting',
    tags: ['parallax occlusion', 'normal mapping', 'procedural materials', 'foil', 'pointer'],
    status: 'prototype',
    cardBackground: 'radial-gradient(ellipse at 40% 40%, #f4efe6 0%, #d9d0c2 30%, #4a372b 62%, #22180f 100%)',
    notes: [
      'The card is a baked height field: the text (runtime SDFs), a border and a small mark are pressed into procedural microstructure. Cotton fibres, brushed-steel scratches or pebbled leather grain are all made from noise, so no textures are downloaded. Pressed areas keep less grain, as under a real die. The bake only re-runs when the design changes.',
      'Every frame is one fullscreen pass with no geometry. The camera ray hits the relief "ceiling", then parallax occlusion mapping walks down through depth layers to the surface and refines the hit with a secant step. Depth only shows when you tilt the card, so it is viewed at an angle with a little pointer parallax.',
      'Normals come from the height field\'s gradient. Lighting is the cursor light (GGX, inverse-square) with relief self-shadows, marched across the height field toward the light, plus studio IBL so steel and foil have something to reflect. The desk gets the card\'s soft shadow and contact occlusion analytically.',
      'Cost: POM steps + 12 shadow steps + 6 normal taps per card pixel. Lower "POM steps" for phones; at very low light heights, raise it to avoid stair-stepping.',
    ],
    load: () => import('../experiments/embossed-card'),
  },
  {
    slug: 'holo-card',
    title: 'Holographic Card',
    summary: 'A foil trading card that tilts toward your cursor (or phone): diffraction rainbows, thin-film colour and brushed highlights.',
    epic: 'ui-lighting',
    tags: ['iridescence', 'diffraction', 'thin film', 'anisotropic', 'gyro'],
    status: 'prototype',
    cardBackground: 'conic-gradient(from 210deg at 55% 45%, #ff7ab8, #ffd36e, #7affc8, #6ec3ff, #b98bff, #ff7ab8)',
    notes: [
      'The card is ray-cast analytically. It is a rotated plane, so there is no mesh. A spring drives its rotation from the pointer, or from device orientation after "Use device tilt" is tapped (iOS asks for permission).',
      'The foil is where the effect lives. A diffraction grating with grooves running radially around the sun splits the light by the grating equation λ = spacing·|t·(l+v)|/m. As the card tilts, different wavelengths reach your eye and the rainbow sweeps across. Glitter cells have random groove directions, so individual flecks flash. A thin-film term (two-beam interference across RGB wavelengths) tints the reflections like oil on water, and anisotropic GGX brushed radially stretches highlights into streaks.',
      'The printed artwork sits under a glossy laminate (a clearcoat GGX lobe plus IBL reflection). In "Reverse holo" and "Full art" layouts the ink partially covers the foil and tints it, which is how real holo cards are printed.',
      'The card casts a soft shadow onto the backdrop, found by intersecting the shadow ray with the card plane. All of this is one fullscreen pass.',
    ],
    load: () => import('../experiments/holo-card'),
  },
  {
    slug: 'glass-panels',
    title: 'Glass Panels',
    summary: 'Draggable frosted-glass UI cards: refraction at the bevels, real blur, colour fringing and rim light from your cursor.',
    epic: 'ui-lighting',
    tags: ['glassmorphism', 'refraction', 'dispersion', 'blur', 'drag'],
    status: 'prototype',
    cardBackground: 'radial-gradient(circle at 30% 40%, #ff8a3d 0%, transparent 45%), radial-gradient(circle at 75% 35%, #d83cc0 0%, transparent 40%), radial-gradient(circle at 55% 80%, #2a7dff 0%, transparent 45%), #0b0b18',
    notes: [
      'The background (light blobs, a grid, the headline, and the panels\' soft drop shadows cast away from the cursor light) is rendered to an HDR texture, then mip-mapped. That gives every panel a cheap, correct blur at any radius.',
      'Each panel is a rounded-rectangle SDF with a circular bevel profile, so the surface normal is flat in the middle and tilts steeply at the rim. The background lookup is offset along that tilt: the centre stays undistorted while the edges bend the scene like a thick slab of glass. Red, green and blue use slightly different offsets (dispersion), so the edges fringe.',
      'Frost is eight rotated taps one mip below the requested blur level, which hides the blockiness of a single mip tap. The cursor light adds a sharp glint and a broad sheen on the bevel, a Fresnel rim and a 1 px hairline. Drag the panels around; the "UI" on the front card is drawn as SDF shapes.',
      'Unlike CSS backdrop-filter, the blur, refraction and lighting are all physically linked and in HDR, so bright blobs bloom through the glass. Cost: one background pass plus mip generation, then nine frosted lookups per glass pixel.',
    ],
    load: () => import('../experiments/glass-panels'),
  },
  {
    slug: 'radiance-cascades',
    title: 'Radiance Cascades',
    summary: 'Paint lights and walls and watch real-time 2D global illumination: soft shadows, colour bleeding and light leaking through gaps.',
    epic: 'ui-lighting',
    tags: ['global illumination', 'radiance cascades', 'jump flood', 'compute', 'paint'],
    status: 'prototype',
    cardBackground: 'radial-gradient(circle at 25% 35%, #ffb36b 0%, transparent 30%), radial-gradient(circle at 75% 30%, #4cc3ff 0%, transparent 28%), radial-gradient(circle at 50% 85%, #ff4fc8 0%, transparent 30%), #0a0b10',
    notes: [
      'Radiance cascades (Alexander Sannikov) trade angular for spatial resolution. Near a probe you need many probes but few directions; far away you need many directions but few probes. Cascade n has probes every 2ⁿ texels, 4ⁿ⁺¹ directions, and only traces the interval [l₀(4ⁿ−1)/3, l₀(4ⁿ⁺¹−1)/3], so every cascade costs the same and together they cover the screen.',
      'Each cascade texel ray-marches its short interval through a jump-flood distance field of the painted walls and lights, which is rebuilt only when you paint. If it hits nothing, it continues with cascade n+1: the four child directions, averaged, from the four nearest upper probes. Doing this top-down leaves cascade 0 holding full radiance at every probe; its four directions average to the fluence that lights the floor.',
      'Everything is light here: painted strokes are emitters, walls block and darken, and the cursor carries a lamp. Shadows soften naturally with distance, colours mix where lights overlap, and light spills through the gap in the ring. Rays that escape the screen pick up a faint "sky".',
      'Merging uses the "bilinear fix": each texel marches a separate segment to the interval start of each of the four upper probes, so segments meet exactly and there are no leaks or bands between cascades. The "Reference (64 rays)" view brute-forces the same field per pixel. The cascade result matches it to within a few hundredths, at a fraction of the cost.',
      'Cost: about 6 cascades × 4·W·H texels at GI resolution (half-res by default), four short marches each, plus a jump flood whenever you paint.',
    ],
    load: () => import('../experiments/radiance-cascades'),
  },
  {
    slug: 'long-shadows',
    title: 'Long Shadows',
    summary: 'Flat-design long shadows for icons and type, cast in real time from your cursor using SDF sweeps.',
    epic: 'ui-lighting',
    tags: ['flat design', 'sdf sweep', 'text', 'pointer'],
    status: 'prototype',
    cardBackground: 'linear-gradient(135deg, #f4a259 0%, #f4a259 48%, #c97d42 48%, #c97d42 100%)',
    notes: [
      'A long shadow is the shape dragged along the light direction. Point p is in shadow if p − dir·t lies inside the shape for some t ≤ L. Instead of stamping many offset copies (the CSS text-shadow trick, which costs one layer per pixel of length), the shader sphere-traces backwards along −dir through the shape\'s SDF. Each step jumps by the distance to the shape, so a 500 px shadow takes a dozen steps.',
      'Two levels, as in classic flat icons: the tiles cast shadows onto the page, and each letter casts its own shadow clipped to its tile. Letters come from one runtime text SDF, and each tile samples its letter\'s slice using the per-letter extents the text generator reports.',
      'Move the pointer to swing the light; the shadow angle follows it. "Fade" turns the shadow from solid to a gradient. Tonemapping is off here, because flat design wants exact brand colours.',
    ],
    load: () => import('../experiments/long-shadows'),
  },
  {
    slug: 'neon-sign',
    title: 'Neon Sign',
    summary: 'A flickering neon sign from any text: glass tubes, white-hot cores and coloured light spilling across brick.',
    epic: 'ui-lighting',
    tags: ['emissive', 'bloom', 'flicker', 'text', 'light spill'],
    status: 'prototype',
    cardBackground: 'radial-gradient(ellipse at 45% 50%, #ff3d8b 0%, #5a1030 30%, #170a10 65%), #0e0b0d',
    notes: [
      'The tubes are distance fields. The text is a runtime SDF (any font; script by default), followed along its outline or inflated into solid letters. Around it sit a rounded-rectangle frame and a moon made from two circles. Each tube is drawn with a white-hot core, a saturated rim and a glassy specular stripe, which stays visible when a tube is off.',
      'Light spill: the tubes\' emission is rendered at half resolution and mip-mapped. Summing several mip levels makes a falloff close to the 1/r of a long light source, so the wall glows in the right colours with no light list. The slope of a very blurred level points toward the tubes and gives a light direction, which bump-shades the bricks or concrete so their relief catches the glow.',
      'Each letter flickers on its own phase, and one is "faulty" and drops out more often with a fast buzz. The post chain\'s bloom does the rest. Flicker defaults to off when the OS asks for reduced motion.',
      'Cost: a half-res emission pass, mip generation, and one composite pass with 7 blurred taps and a few SDF evaluations per pixel.',
    ],
    load: () => import('../experiments/neon-sign'),
  },
  {
    slug: 'lit-type',
    title: 'Lit Typography',
    summary: 'A real, editable HTML heading rendered as lit 3D extruded letters, with per-letter metals and shadows from your cursor.',
    epic: 'ui-lighting',
    tags: ['text', 'ray marching', 'dom', 'oblique projection', 'ibl'],
    status: 'prototype',
    cardBackground: 'linear-gradient(160deg, #e2d9cc 0%, #d0c4b3 55%, #b89c6a 56%, #e9c27a 70%, #c9b49a 100%)',
    notes: [
      'The heading is real DOM text. It is selectable, editable (click and type), and readable by screen readers, but its ink is transparent. Underneath, WebGPU draws the same text as solid extruded letters. Every frame reads the heading\'s computed font and measures its pen origin and baseline with a DOM Range. The text SDF is generated from that exact font, so the 3D letters line up with the DOM glyphs. Toggle "Show DOM text" to check.',
      'The letters are the text SDF extruded with rounded edges and sphere-traced together with the wall plane. The projection is oblique: each ray is p + shift·(depth − z). The letters\' front faces always land exactly on the DOM glyphs, while moving the pointer skews the rays to reveal the extruded sides like a tilt, without breaking selection.',
      'Shading: cursor light (GGX) with SDF soft shadows on the wall and between letters, five-tap SDF ambient occlusion, and studio IBL so the metals have something to reflect. Each letter gets its own material from the per-letter extents the text generator measures.',
      'Cost: up to 96 march steps plus 40 shadow steps per pixel, and the march only runs where rays can hit letters, so it is fine at 1080p on integrated GPUs. Lower the render scale on phones.',
    ],
    load: () => import('../experiments/lit-type'),
  },
];

export function findExperiment(slug: string): ExperimentMeta | undefined {
  return experiments.find((e) => e.slug === slug);
}
