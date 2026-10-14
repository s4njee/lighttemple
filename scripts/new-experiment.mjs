#!/usr/bin/env node
/**
 * Scaffolds a new experiment and registers it in the gallery.
 *
 *   npm run new -- <slug> [--title "Nice Title"] [--epic ui-lighting]
 *
 * Creates src/experiments/<slug>/{index.ts,shader.wgsl} from a working
 * fullscreen-shader template (pointer light → HDR → shared post chain) and
 * appends an entry to src/gallery/registry.ts.
 */

import { existsSync } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const EPICS = ['ui-lighting', 'shadows', 'global-illumination', 'volumetrics', 'stylized', 'many-lights', 'web-integration'];

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: { title: { type: 'string' }, epic: { type: 'string', default: 'ui-lighting' } },
});

const slug = positionals[0];
const fail = (msg) => {
  console.error(`error: ${msg}\nusage: npm run new -- <slug> [--title "Nice Title"] [--epic ${EPICS.join('|')}]`);
  process.exit(1);
};
if (!slug) fail('missing slug');
if (!/^[a-z0-9]+(-[a-z0-9]+)*$/.test(slug)) fail(`slug must be kebab-case, got "${slug}"`);
if (!EPICS.includes(values.epic)) fail(`unknown epic "${values.epic}"`);

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const dir = join(root, 'src/experiments', slug);
const registryPath = join(root, 'src/gallery/registry.ts');
if (existsSync(dir)) fail(`src/experiments/${slug} already exists`);

const registry = await readFile(registryPath, 'utf8');
if (registry.includes(`slug: '${slug}'`)) fail(`"${slug}" is already registered`);

const title = values.title ?? slug.replace(/(^|-)(\w)/g, (_, sep, c) => (sep ? ' ' : '') + c.toUpperCase());
const q = (s) => `'${s.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}'`;

const indexTs = `import commonWgsl from '../../shaders/common.wgsl?raw';
import fullscreenWgsl from '../../shaders/fullscreen.wgsl?raw';
import { createShader, wgsl } from '../../gpu/shader';
import { UniformBlock } from '../../gpu/uniforms';
import { HDR_FORMAT, PostProcess } from '../../post/post';
import { hexToLinear } from '../../gpu/color';
import type { ExperimentFactory } from '../types';
import shaderWgsl from './shader.wgsl?raw';

const create: ExperimentFactory = async (ctx) => {
  const { device, surface, params, pointer } = ctx;

  // Param keys must be unique across the panel (they are also the URL keys).
  const light = params.add(
    {
      intensity: { value: 1.5, min: 0, max: 6, step: 0.01 },
      color: { value: '#ffd2a1', type: 'color' },
      height: { value: 0.3, min: 0.02, max: 1.5, step: 0.01 },
    },
    'Light',
  );
  const post = await PostProcess.create(ctx);

  // Declared once here; \`uniforms.wgslStruct\` is spliced into the shader.
  const uniforms = new UniformBlock(device, 'Params', {
    resolution: 'vec2f',
    time: 'f32',
    intensity: 'f32',
    light: 'vec4f',
    lightColor: 'vec4f',
  });

  const module = await createShader(device, ${q(slug)}, wgsl(commonWgsl, fullscreenWgsl, uniforms.wgslStruct, shaderWgsl));
  const pipeline = await device.createRenderPipelineAsync({
    label: ${q(slug)},
    layout: 'auto',
    vertex: { module, entryPoint: 'vs_fullscreen' },
    fragment: { module, entryPoint: 'fs_main', targets: [{ format: HDR_FORMAT }] },
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniforms.buffer } }],
  });

  return {
    frame({ time, size }) {
      const aspect = size.width / size.height;
      const [u, v] = pointer.smoothUv;
      uniforms
        .set('resolution', [size.width, size.height])
        .set('time', time)
        .set('intensity', light.intensity)
        .set('light', [(u - 0.5) * 2 * aspect, (0.5 - v) * 2, light.height, 0])
        .set('lightColor', [...hexToLinear(light.color), 0])
        .upload();

      const encoder = device.createCommandEncoder({ label: ${q(slug)} });
      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: post.scene(size).view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(3);
      pass.end();
      post.render(encoder, surface.currentTexture().createView(), { time });
      device.queue.submit([encoder.finish()]);
    },
    dispose() {
      post.destroy();
      uniforms.destroy();
    },
  };
};

export default create;
`;

const shaderWgsl = `// ---------------------------------------------------------------------------
// ${title}
// Outputs linear HDR (alpha = coverage) into the shared post chain.
// Requires (prepended): common.wgsl, fullscreen.wgsl, the \`Params\` struct.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> u: Params;

@fragment
fn fs_main(v: FullscreenOut) -> @location(0) vec4f {
  // World space: y in [-1, 1] (up), x in [-aspect, aspect].
  let p = (v.position.xy - 0.5 * u.resolution) / u.resolution.y * vec2f(2.0, -2.0);

  // A point light hovering over a flat floor: inverse-square falloff x Lambert.
  let to_light = vec3f(u.light.xy - p, u.light.z);
  let dist2 = dot(to_light, to_light);
  let ndl = u.light.z * inverseSqrt(dist2);
  let albedo = vec3f(0.6, 0.57, 0.53);
  let col = albedo / PI * u.lightColor.rgb * u.intensity * ndl / dist2 + albedo * 0.02;
  return vec4f(col, 1.0);
}
`;

const entry = `  {
    slug: ${q(slug)},
    title: ${q(title)},
    summary: 'TODO: one-line pitch.',
    epic: ${q(values.epic)},
    tags: [],
    status: 'prototype',
    cardBackground: 'radial-gradient(circle at 50% 45%, #ffe2bd 0%, #6c625a 60%, #2a2623 100%)',
    notes: ['TODO: how it works, what it costs, when to use it.'],
    load: () => import('../experiments/${slug}'),
  },
`;

const marker = '\n];\n\nexport function findExperiment';
if (!registry.includes(marker)) fail('could not find the end of the experiments array in registry.ts');

await mkdir(dir, { recursive: true });
await writeFile(join(dir, 'index.ts'), indexTs);
await writeFile(join(dir, 'shader.wgsl'), shaderWgsl);
await writeFile(registryPath, registry.replace(marker, `\n${entry}];\n\nexport function findExperiment`));

console.log(`created src/experiments/${slug}/index.ts
created src/experiments/${slug}/shader.wgsl
registered "${title}" in src/gallery/registry.ts

next: npm run dev, then open #/x/${slug}`);
