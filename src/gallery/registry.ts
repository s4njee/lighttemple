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
    tags: ['sdf', 'soft shadows', 'pointer', 'fullscreen'],
    status: 'prototype',
    cardBackground:
      'radial-gradient(circle at 62% 38%, #ffe2bd 0%, #c9b9a5 22%, #6c625a 58%, #2a2623 100%)',
    notes: [
      'The whole scene is one fullscreen fragment shader. Shapes are a 2D signed distance field (SDF), extruded to a fixed height with a smooth bevel, so every pixel knows its height and surface normal without any geometry.',
      'Shadows are ray-marched through the SDF toward each light. The tightest clearance-to-distance ratio along the ray gives a penumbra that widens with distance from the occluder, the classic Quilez soft-shadow trick. The ray also climbs toward the light, so once it clears the top of a shape it is no longer blocked. That is why shadows end, and why they lengthen as you lower the light.',
      'Lighting uses inverse-square falloff, regularised by the light radius, with Lambert diffuse and normalised Blinn-Phong. Output goes through AgX or ACES, then sRGB encoding with triangular dither so the long, smooth gradients do not band on 8-bit displays.',
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
];

export function findExperiment(slug: string): ExperimentMeta | undefined {
  return experiments.find((e) => e.slug === slug);
}
