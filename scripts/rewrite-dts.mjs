// Rewrite relative import specifiers in the emitted declarations to the paths the library build
// emits, so a consumer's resolver (which looks for `*.d.ts` beside `*.js`) finds them. This handles
// both `.ts` specifiers and extensionless ones (Light Temple writes extensionless relative imports,
// resolving a sibling file to `<name>.js` and a directory to `<name>/index.js`).
import { existsSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';

const root = 'types';

const resolveSpecifier = (fromFile, spec) => {
  if (spec.endsWith('.ts')) return `${spec.slice(0, -3)}.js`;
  if (/\.[a-z]+$/.test(spec)) return spec; // already .js, .json, ...
  const base = resolve(dirname(fromFile), spec);
  if (existsSync(`${base}.d.ts`)) return `${spec}.js`;
  if (existsSync(join(base, 'index.d.ts'))) return `${spec}/index.js`;
  return spec;
};

let changed = 0;
const walk = (dir) => {
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry);
    if (statSync(path).isDirectory()) walk(path);
    else if (entry.endsWith('.d.ts')) {
      const source = readFileSync(path, 'utf8');
      // A bundled `?raw` default import cannot resolve outside a bundler; expose it as the string
      // value it is (e.g. `import iblWgsl from './ibl.wgsl?raw'` -> `declare const iblWgsl: string`).
      const rewritten = source
        .replace(
          /^import\s+([A-Za-z_$][\w$]*)\s+from\s+(['"])[^'"]*\?raw\2;\s*$/gm,
          'declare const $1: string;',
        )
        .replace(
          /(['"])(\.\.?\/[^'"]*)\1/g,
          (_match, quote, spec) => `${quote}${resolveSpecifier(path, spec)}${quote}`,
        );
      if (rewritten !== source) {
        writeFileSync(path, rewritten);
        changed += 1;
      }
    }
  }
};
walk(root);
console.log(`rewrote relative import specifiers in ${changed} declaration files`);
