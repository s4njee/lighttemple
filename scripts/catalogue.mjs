// Emits catalogue.json for the gallery's experiments (M3 stories 3.9 and 3.10).
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const ts = require('typescript');
const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));

const source = ts.transpileModule(fs.readFileSync(path.join(root, 'src/gallery/registry.ts'), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
}).outputText;
const registry = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));

const experiments = registry.experiments.map((e) => ({
  id: e.slug,
  title: e.title,
  summary: e.summary,
  tags: [...e.tags],
  category: registry.EPIC_LABELS[e.epic],
  background: e.cardBackground,
  // Every experiment is a prototype today; only a promoted `stable` one would be `ready`.
  status: e.status === 'stable' ? 'ready' : 'research',
}));

const catalogue = { project: 'lighttemple', experiments };
fs.writeFileSync(path.join(root, 'catalogue.json'), JSON.stringify(catalogue, null, 2) + '\n');
console.log(`${experiments.length} experiments, ${experiments.filter((e) => e.status === 'ready').length} ready`);
