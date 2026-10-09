/**
 * WGSL module helpers.
 *
 * Shaders are composed by plain string concatenation of reusable chunks
 * (see src/shaders/*.wgsl, imported with Vite's `?raw`). That keeps the
 * toolchain trivial and the generated source easy to read in error reports.
 */

export class ShaderCompileError extends Error {
  constructor(
    readonly label: string,
    readonly report: string,
  ) {
    super(`WGSL compilation failed for "${label}"\n${report}`);
    this.name = 'ShaderCompileError';
  }
}

/** Joins WGSL chunks with blank lines so line numbers in errors stay readable. */
export function wgsl(...chunks: string[]): string {
  return chunks.join('\n\n');
}

/** Compiled modules per device, keyed by source. Experiments that share code
 * (e.g. the post chain) compile it once per session, not once per mount. */
const moduleCache = new WeakMap<GPUDevice, Map<string, Promise<GPUShaderModule>>>();

/**
 * Creates (or reuses) a shader module and awaits its compilation info. Errors
 * throw a ShaderCompileError with a short code excerpt around each failing
 * line; warnings are logged.
 */
export function createShader(device: GPUDevice, label: string, code: string): Promise<GPUShaderModule> {
  let cache = moduleCache.get(device);
  if (!cache) moduleCache.set(device, (cache = new Map()));
  let pending = cache.get(code);
  if (!pending) {
    pending = compile(device, label, code);
    cache.set(code, pending);
    pending.catch(() => cache.delete(code)); // don't cache failures
  }
  return pending;
}

async function compile(device: GPUDevice, label: string, code: string): Promise<GPUShaderModule> {
  const module = device.createShaderModule({ label, code });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  for (const m of info.messages) {
    if (m.type === 'warning') console.warn(`[wgsl:${label}] ${m.lineNum}:${m.linePos} ${m.message}`);
  }
  if (errors.length > 0) {
    throw new ShaderCompileError(label, formatMessages(code, errors));
  }
  return module;
}

function formatMessages(code: string, messages: readonly GPUCompilationMessage[]): string {
  const lines = code.split('\n');
  return messages
    .map((m) => {
      const from = Math.max(0, m.lineNum - 3);
      const to = Math.min(lines.length, m.lineNum + 2);
      const excerpt = lines
        .slice(from, to)
        .map((l, i) => {
          const n = from + i + 1;
          const marker = n === m.lineNum ? '>' : ' ';
          return `${marker} ${String(n).padStart(4)} | ${l}`;
        })
        .join('\n');
      return `error at ${m.lineNum}:${m.linePos}: ${m.message}\n${excerpt}`;
    })
    .join('\n\n');
}

/**
 * Runs `fn` inside validation + out-of-memory error scopes and rethrows the
 * first captured error. Use around resource/pipeline creation so validation
 * problems show up in the UI instead of only in the console.
 */
export async function withErrorScope<T>(device: GPUDevice, fn: () => Promise<T> | T): Promise<T> {
  device.pushErrorScope('out-of-memory');
  device.pushErrorScope('validation');
  let result: T;
  try {
    result = await fn();
  } catch (err) {
    // Keep the scopes balanced, but report the original (usually more specific) error.
    await device.popErrorScope();
    await device.popErrorScope();
    throw err;
  }
  const validation = await device.popErrorScope();
  const oom = await device.popErrorScope();
  const err = validation ?? oom;
  if (err) throw new Error(`GPU ${validation ? 'validation' : 'out-of-memory'} error: ${err.message}`);
  return result;
}
